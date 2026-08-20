"""
Batch independence is an INVARIANT of every recognition model in this repo.

Measured defect (2026-08-20): paddle's TransformerEncoder is batch-first
([batch, seq, dim]), but the encoders transposed to [T, B, C] - the PyTorch
convention - so self-attention ran ACROSS BATCH SAMPLES at each timestep.
On stage-3b epoch-44 weights, one image's logits changed by up to 7.4
depending on its batch companions; 97/102 val predictions differed between
batch-1 and batch-16; the trainer's batched eval reported 0.7549 char
accuracy while the single-image (deployment) score of the same weights was
0.6661. Neither number was wrong - they measured different functions, which
is exactly why batch composition must never be a model input.

Test design note (itself measured): at random initialization the untrained
backbone collapses every input to ~1e-18 features in eval mode, so
FULL-MODEL tests cannot discriminate - all companions look identical and
even the defective forward shows no leak. A test that cannot fail under
the defect is worse than none. These tests therefore drive the
transformer stage directly with synthetic non-zero features, where the
defect lives and where it is provably visible (leak magnitude ~1 at these
scales, threshold 1e-3, margin >100x).
"""

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

from src.vin_ocr.training.finetune_paddleocr import (  # noqa: E402
    PPOCRv5RecognitionModel,
    SVTREncoder,
    VINRecognitionModel,
)

CONFIG = {
    "Architecture": {
        "Neck": {
            "hidden_dim": 64,
            "Transformer": {"num_heads": 4, "num_layers": 1, "dropout": 0.1},
        },
        "Head": {"dropout": 0.1},
    }
}


def _feature_batch(seed: int, n: int, channels: int) -> np.ndarray:
    """Synthetic non-zero backbone features [n, C, H=4, W=40]."""
    return np.random.RandomState(seed).rand(n, channels, 4, 40).astype("float32")


def _neck_out_for_sample0(neck, companions: np.ndarray, channels: int) -> np.ndarray:
    sample0 = _feature_batch(seed=99, n=1, channels=channels)
    batch = np.concatenate([sample0, companions], axis=0)
    with paddle.no_grad():
        return neck(paddle.to_tensor(batch)).numpy()[0]


class TestSVTREncoderBatchIndependence:
    CHANNELS = 32

    def _make(self, legacy: bool) -> SVTREncoder:
        paddle.seed(0)
        neck = SVTREncoder(
            in_channels=self.CHANNELS, hidden_dim=64, num_heads=4,
            num_layers=1, dropout=0.1,
            legacy_batch_axis_attention=legacy,
        )
        neck.eval()
        return neck

    def test_default_forward_is_batch_independent(self):
        neck = self._make(legacy=False)
        alone = _neck_out_for_sample0(
            neck, _feature_batch(1, 0, self.CHANNELS), self.CHANNELS)
        with_a = _neck_out_for_sample0(
            neck, _feature_batch(1, 3, self.CHANNELS), self.CHANNELS)
        with_b = _neck_out_for_sample0(
            neck, _feature_batch(2, 3, self.CHANNELS), self.CHANNELS)
        np.testing.assert_allclose(with_a, with_b, atol=1e-5)
        np.testing.assert_allclose(alone, with_a, atol=1e-5)

    def test_legacy_flag_reproduces_companion_dependence(self):
        """The defect-compat mode must still leak; that leakage IS the
        semantics archived checkpoints were trained and measured under.
        If this ever passes silently under legacy=True, the compat mode
        no longer reproduces pre-fix checkpoints and must not be used."""
        neck = self._make(legacy=True)
        with_a = _neck_out_for_sample0(
            neck, _feature_batch(1, 3, self.CHANNELS), self.CHANNELS)
        with_b = _neck_out_for_sample0(
            neck, _feature_batch(2, 3, self.CHANNELS), self.CHANNELS)
        assert float(np.abs(with_a - with_b).max()) > 1e-3, (
            "legacy mode no longer attends across the batch axis - it "
            "cannot reproduce pre-fix checkpoint behaviour any more"
        )

    def test_batched_equals_singles(self):
        """The property the metrics layer depends on: evaluating N inputs
        in one batch must equal evaluating them one at a time."""
        neck = self._make(legacy=False)
        feats = _feature_batch(7, 4, self.CHANNELS)
        with paddle.no_grad():
            batched = neck(paddle.to_tensor(feats)).numpy()
            singles = np.stack([
                neck(paddle.to_tensor(feats[i:i + 1])).numpy()[0]
                for i in range(4)
            ])
        np.testing.assert_allclose(batched, singles, atol=1e-5)


class TestV5TransformerStageBatchIndependence:
    """PPOCRv5RecognitionModel keeps its transformer inline in forward();
    this drives the identical post-backbone stage (neck_conv -> pool ->
    squeeze/transpose -> transformer) with synthetic features."""

    def _stage(self, model, feats: np.ndarray) -> np.ndarray:
        x = paddle.to_tensor(feats)
        with paddle.no_grad():
            x = model.neck_conv(x)
            x = model.pool(x).squeeze(2).transpose([0, 2, 1])
            x = model.transformer(x)
        return x.numpy()

    def test_v5_transformer_stage_is_batch_independent(self):
        paddle.seed(0)
        model = PPOCRv5RecognitionModel(CONFIG, num_classes=34)
        model.eval()
        hidden = model.neck_conv[0].weight.shape[1]
        sample0 = _feature_batch(99, 1, hidden)
        with_a = self._stage(model, np.concatenate(
            [sample0, _feature_batch(1, 3, hidden)]))[0]
        with_b = self._stage(model, np.concatenate(
            [sample0, _feature_batch(2, 3, hidden)]))[0]
        np.testing.assert_allclose(with_a, with_b, atol=1e-5)


class TestSemanticsPlumbing:
    def test_loader_and_evaluator_expose_the_basis(self):
        import inspect

        from src.vin_ocr.tracking import model_registry as mr

        for fn in (mr._load_recognizer, mr.evaluate_checkpoint,
                   mr.register_checkpoint_version, mr.traced_recognize):
            assert "legacy_batch_axis_attention" in inspect.signature(fn).parameters, (
                f"{fn.__name__} must expose the semantics basis explicitly"
            )

    def test_model_constructor_exposes_the_basis(self):
        import inspect
        params = inspect.signature(VINRecognitionModel.__init__).parameters
        assert "legacy_batch_axis_attention" in params

    def test_pyfunc_defaults_to_legacy_when_semantics_artifact_absent(self):
        """Models logged before the fix carry no semantics.json; loading
        them under the fixed forward would serve garbage measured at
        0.1113 char accuracy. Absence must therefore mean legacy."""
        import inspect

        from src.vin_ocr.tracking.model_registry import VINRecognizerPyfunc

        source = inspect.getsource(VINRecognizerPyfunc.load_context)
        assert "legacy = True" in source and "semantics" in source

    def test_onnx_export_tools_use_canonical_architecture_with_legacy_basis(self):
        """The re-export tools exist solely for pre-fix stranded weights;
        they must build the single canonical class under legacy semantics,
        never a drifted inline copy."""
        import ast
        from pathlib import Path

        for script in ("scripts/reexport_and_convert_onnx.py",
                       "scripts/convert_all_to_onnx.py"):
            source = Path(script).read_text()
            assert "legacy_batch_axis_attention=True" in source, script
            tree = ast.parse(source)
            inline_classes = [n.name for n in ast.walk(tree)
                              if isinstance(n, ast.ClassDef)
                              and n.name in ("SVTREncoder", "VINRecognitionModel",
                                             "PPLCNetV3Backbone", "CTCHead")]
            assert inline_classes == [], (
                f"{script} re-declares architecture inline: {inline_classes}"
            )
