"""
Apple-GPU (torch/MPS) training stack: invariants and cross-stack parity.

All tests run on CPU so CI without MPS exercises them; the MPS-specific
facts (device availability, aten::_ctc_loss gap, measured speedup) are
runtime-verified by the trainer and the smoke script on the target machine.
"""

import json
import math

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from src.vin_ocr.training.finetune_torch import (  # noqa: E402
    TorchVINTrainer,
    ctc_loss_cpu_bridge,
    evaluate_torch_checkpoint,
    lr_at_epoch,
    validate_config,
)
from src.vin_ocr.training.torch_rosetta import (  # noqa: E402
    MIN_CTC_TIMESTEPS,
    OUTPUT_TIMESTEPS,
    RosettaResNet34Torch,
)


def _scratch_model() -> RosettaResNet34Torch:
    torch.manual_seed(0)
    model = RosettaResNet34Torch(num_classes=34, pretrained_backbone=False)
    model.eval()
    return model


class TestGeometry:
    def test_output_shape_and_ctc_bound(self):
        model = _scratch_model()
        x = torch.rand(2, 3, 48, 320)
        with torch.no_grad():
            logits = model(x)
        assert tuple(logits.shape) == (2, OUTPUT_TIMESTEPS, 34)
        assert OUTPUT_TIMESTEPS >= MIN_CTC_TIMESTEPS

    def test_parameter_budget_is_resnet34_class(self):
        model = _scratch_model()
        n = sum(p.numel() for p in model.parameters())
        assert 20_000_000 < n < 23_000_000, f"{n:,}"


class TestBatchIndependence:
    def test_eval_forward_is_batch_independent_with_power(self):
        model = _scratch_model()
        rng = np.random.RandomState(7)
        sample0 = torch.from_numpy(rng.rand(1, 3, 48, 320).astype("float32"))
        comp_a = torch.from_numpy(
            np.random.RandomState(1).rand(3, 3, 48, 320).astype("float32"))
        comp_b = torch.from_numpy(
            np.random.RandomState(2).rand(3, 3, 48, 320).astype("float32"))
        with torch.no_grad():
            alone = model(sample0).numpy()[0]
            with_a = model(torch.cat([sample0, comp_a])).numpy()[0]
            with_b = model(torch.cat([sample0, comp_b])).numpy()[0]
        assert float(np.abs(alone).std()) > 1e-3, (
            "features collapsed at random init; test has no power")
        np.testing.assert_allclose(with_a, with_b, atol=1e-4)
        np.testing.assert_allclose(alone, with_a, atol=1e-4)

    def test_no_sequence_module(self):
        model = _scratch_model()
        forbidden = (torch.nn.TransformerEncoder,
                     torch.nn.TransformerEncoderLayer,
                     torch.nn.MultiheadAttention,
                     torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)
        offenders = [name for name, module in model.named_modules()
                     if isinstance(module, forbidden)]
        assert offenders == []


class TestSchedule:
    def test_warmup_then_cosine(self):
        peak, warmup, total = 2e-4, 3, 30
        lrs = [lr_at_epoch(e, peak, warmup, total) for e in range(1, total + 1)]
        assert lrs[0] < lrs[1] < lrs[2] <= peak * 1.0001
        assert abs(lrs[warmup - 1] - peak) / peak < 0.01
        assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(warmup, total - 1))
        assert lrs[-1] < peak * 0.01

    def test_no_warmup_starts_at_peak(self):
        assert lr_at_epoch(1, 1e-4, 0, 10) == pytest.approx(1e-4, rel=1e-6)


class TestConfigGates:
    BASE = {
        "Global": {"epoch_num": 10, "character_dict_path": "configs/vin_dict.txt"},
        "Optimizer": {"lr": {"learning_rate": 2e-4, "warmup_epoch": 3}},
        "Train": {"dataset": {"label_file_list": ["x"]}},
        "Eval": {"dataset": {"label_file_list": ["y"]}},
    }

    def _config(self, **overrides):
        import copy
        config = copy.deepcopy(self.BASE)
        for dotted, value in overrides.items():
            node = config
            *parents, leaf = dotted.split(".")
            for key in parents:
                node = node[key]
            node[leaf] = value
        return config

    def test_accepts_valid(self):
        validate_config(self._config())

    def test_rejects_unsafe_learning_rate(self):
        with pytest.raises(ValueError, match="measured safe bound"):
            validate_config(self._config(**{"Optimizer.lr.learning_rate": 3e-3}))

    def test_rejects_warmup_not_below_epochs(self):
        with pytest.raises(ValueError, match="warmup_epoch"):
            validate_config(self._config(**{"Optimizer.lr.warmup_epoch": 10}))

    def test_rejects_missing_dict(self):
        with pytest.raises(ValueError, match="character_dict_path"):
            validate_config(self._config(**{"Global.character_dict_path": "nope.txt"}))


class TestCTCBridge:
    def test_finite_loss_and_gradient_flow(self):
        torch.manual_seed(0)
        logits = torch.randn(4, OUTPUT_TIMESTEPS, 34, requires_grad=True)
        labels = torch.randint(1, 34, (4, 17))
        lengths = torch.full((4,), 17, dtype=torch.long)
        loss = ctc_loss_cpu_bridge(logits, labels, lengths,
                                   valid_widths=[320, 280, 200, 320])
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_parity_with_paddle_ctc(self):
        """Cross-stack golden: identical inputs -> per-sample CTC losses
        agree between torch (training stack) and paddle (frozen evaluator
        stack). This is the equivalence evidence for comparing trajectories
        across the two trainers."""
        paddle = pytest.importorskip("paddle")

        rng = np.random.RandomState(0)
        batch, timesteps, classes, label_len = 3, 40, 34, 17
        logits_np = rng.randn(batch, timesteps, classes).astype("float32")
        labels_np = rng.randint(1, classes, (batch, label_len)).astype("int32")

        log_probs_torch = torch.from_numpy(logits_np).log_softmax(-1).permute(1, 0, 2)
        torch_losses = torch.nn.functional.ctc_loss(
            log_probs_torch, torch.from_numpy(labels_np.astype("int64")),
            torch.full((batch,), timesteps, dtype=torch.long),
            torch.full((batch,), label_len, dtype=torch.long),
            blank=0, reduction="none",
        ).numpy()

        log_probs_paddle = paddle.nn.functional.log_softmax(
            paddle.to_tensor(logits_np), axis=-1).transpose([1, 0, 2])
        paddle_losses = paddle.nn.functional.ctc_loss(
            log_probs_paddle, paddle.to_tensor(labels_np),
            paddle.to_tensor(np.full(batch, timesteps, dtype="int64")),
            paddle.to_tensor(np.full(batch, label_len, dtype="int64")),
            blank=0, reduction="none", norm_by_times=False,
        ).numpy().reshape(-1)

        np.testing.assert_allclose(torch_losses, paddle_losses, rtol=1e-3)


class TestEndToEndOnSynthetic:
    """Full-loop: dataset parity, checkpoint roundtrip, canonical eval."""

    @pytest.fixture()
    def synthetic(self, tmp_path):
        from scripts.train_smoke_test import build_dataset
        build_dataset(tmp_path, n_train=2, n_val=2)
        config = {
            "Global": {
                "epoch_num": 1, "device": "cpu", "seed": 42,
                "character_dict_path": "configs/vin_dict.txt",
                "max_text_length": 17,
                "save_model_dir": str(tmp_path / "out"),
            },
            "Architecture": {"algorithm": "RosettaTorch",
                             "pretrained_backbone": False,
                             "Head": {"dropout": 0.1}},
            "Optimizer": {"lr": {"learning_rate": 1e-4, "warmup_epoch": 0}},
            "Train": {"dataset": {"data_dir": str(tmp_path),
                                  "label_file_list": [str(tmp_path / "train_labels.txt")]},
                      "loader": {"batch_size_per_card": 2}},
            "Eval": {"dataset": {"data_dir": str(tmp_path),
                                 "label_file_list": [str(tmp_path / "val_labels.txt")]},
                     "loader": {"batch_size_per_card": 2}},
        }
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.safe_dump(config))
        return config, config_path, tmp_path

    def test_dataset_parity_with_paddle_path(self, synthetic):
        pytest.importorskip("paddle")
        from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset
        from src.vin_ocr.training.finetune_torch import TorchVINDataset
        from src.vin_ocr.core.charset import load_char_dict

        config, _, tmp_path = synthetic
        char_to_idx, _ = load_char_dict("configs/vin_dict.txt")
        torch_dataset = TorchVINDataset(
            data_dir=str(tmp_path),
            label_file=str(tmp_path / "val_labels.txt"),
            char_dict=char_to_idx, is_training=False)
        paddle_dataset = VINRecognitionDataset(
            data_dir=str(tmp_path),
            label_file=str(tmp_path / "val_labels.txt"),
            char_dict=char_to_idx, is_training=False)
        torch_item = torch_dataset[0]
        paddle_item = paddle_dataset[0]
        np.testing.assert_array_equal(
            torch_item["image"].numpy(), paddle_item["image"])
        assert torch_item["text"] == paddle_item["text"]
        assert torch_item["valid_width"] == int(paddle_item["valid_width"][0])

    def test_train_checkpoint_evaluate_roundtrip(self, synthetic):
        config, config_path, tmp_path = synthetic
        trainer = TorchVINTrainer(config)
        result = trainer.train()
        assert math.isfinite(result["final_train_loss"])

        out = tmp_path / "out"
        assert (out / "latest.pt").is_file()
        info = json.loads((out / "latest_info.json").read_text())
        assert info["epoch"] == 1 and info["framework"].startswith("torch-")

        measured = evaluate_torch_checkpoint(
            str(out / "latest.pt"), str(tmp_path / "val_labels.txt"),
            data_dir=str(tmp_path), config_path=str(config_path))
        assert measured["n"] == 2.0
        assert 0.0 <= measured["char_accuracy"] <= 1.0
        assert 0.0 <= measured["checksum_valid_rate"] <= 1.0
