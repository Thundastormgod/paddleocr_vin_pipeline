"""
Regression tests for the fine-tune training loop, from the enterprise
training-readiness sweeps of 2026-08-18. Every defect here was confirmed by
executing real training runs on a synthetic micro-dataset:

1. CTC dtype contract: paddle 3.3's warpctc requires labels int32 but BOTH
   length tensors int64. The trainer passed all-int32 and crashed on the
   first batch - the trainer could not run at all on paddle 3.3.x.
2. Raw-logits contract: warpctc applies softmax internally ("aliased as
   softmax with CTC"); the trainer fed log_softmax output.
3. Early stopping was unconditionally fatal: best_accuracy was updated
   BEFORE the improvement test, which therefore compared
   "val > val + min_delta" (always False). The patience counter never
   reset and every run was killed after exactly patience+1 epochs
   regardless of progress - including every historical Optuna trial
   (patience 3-20).
4. The second is_best computation in the same loop was always False for
   the same reason, making save_checkpoint's best branch dead code.
5. `latest` was only written inside the conditional save (crash before the
   first improvement or save_epoch_step boundary lost all progress), had
   no `latest_info.json` (resume-from-latest silently restarted the epoch
   counter and LR warmup), and all checkpoint writes were non-atomic.
"""

import ast
import inspect
import json
from pathlib import Path

import pytest

from src.vin_ocr.training.finetune_paddleocr import (
    VINFineTuner,
    update_early_stopping,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = REPO_ROOT / "src" / "vin_ocr" / "training" / "finetune_paddleocr.py"


class TestEarlyStoppingLogic:
    """Pure decision logic - the guillotine bug and its boundaries."""

    def test_improvement_resets_the_counter(self):
        improved, counter = update_early_stopping(
            val_accuracy=0.30, previous_best=0.20,
            epochs_without_improvement=7, min_delta=0.001,
        )
        assert improved is True
        assert counter == 0

    def test_no_improvement_increments(self):
        improved, counter = update_early_stopping(
            val_accuracy=0.20, previous_best=0.20,
            epochs_without_improvement=2, min_delta=0.001,
        )
        assert improved is False
        assert counter == 3

    def test_min_delta_boundary_is_strict(self):
        """Exactly min_delta better does NOT count as improvement."""
        improved, _ = update_early_stopping(
            val_accuracy=0.201, previous_best=0.200,
            epochs_without_improvement=0, min_delta=0.001,
        )
        assert improved is False

    def test_the_historical_bug_shape_never_improves(self):
        """
        The old code passed the ALREADY-UPDATED best as the comparison
        target. This documents why the signature demands previous_best:
        with val==previous_best (the updated value), improvement can never
        register, whatever the actual progress was.
        """
        improved, counter = update_early_stopping(
            val_accuracy=0.99, previous_best=0.99,   # already updated
            epochs_without_improvement=0, min_delta=0.001,
        )
        assert improved is False
        assert counter == 1


class TestTrainLoopStructure:
    """AST/source pins on the fixed epoch-end block."""

    def test_single_is_best_computed_from_previous_best(self):
        """
        Counted in the AST, not the raw text: the explanatory comments
        legitimately quote the old duplicated assignment.
        """
        source = inspect.getsource(VINFineTuner.train)
        tree = ast.parse("class _W:\n" + source.replace("\n", "\n    "))
        assigns = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "is_best"
                for t in node.targets
            )
        ]
        assert len(assigns) == 1, (
            "the duplicate (always-False) is_best computation is back"
        )
        assert "previous_best = self.best_accuracy" in source
        assert "update_early_stopping(" in source

    def test_latest_is_saved_every_epoch(self):
        """_save_latest must be called unconditionally in the epoch loop."""
        source = inspect.getsource(VINFineTuner.train)
        assert "self._save_latest(epoch)" in source

    def test_early_stop_break_happens_after_checkpointing(self):
        """
        The stopping epoch must still be logged and checkpointed: the break
        is driven by the stop_early flag AFTER the save block, not inline
        in the early-stopping check.
        """
        source = inspect.getsource(VINFineTuner.train)
        assert "stop_early = True" in source
        save_pos = source.find("self._save_latest(epoch)")
        break_pos = source.find("if stop_early:")
        assert 0 < save_pos < break_pos

    def test_ctc_branches_use_the_mixed_dtype_contract(self):
        """
        labels int32; input_lengths and target_lengths int64; and no
        log_softmax fed to the criterion (warpctc normalises internally).
        Applies to BOTH the training and validation loops.
        """
        for method in (VINFineTuner.train_epoch, VINFineTuner.validate):
            source = inspect.getsource(method)
            assert "paddle.to_tensor(batch['label'], dtype='int32')" in source, method
            assert "dtype='int64').reshape([-1])" in source, method
            # The exact input_lengths construction (checked as the full
            # assignment expression - comments also mention input_lengths).
            assert (
                "input_lengths = paddle.full([logits.shape[0]], "
                "logits.shape[1], dtype='int64')" in source
            ), method
            ctc_branch = source.split("if self.use_ctc:")[1].split("else:")[0]
            assert "log_softmax(" not in ctc_branch, (
                f"{method.__name__} feeds normalised probabilities to warpctc"
            )

    def test_no_bare_excepts_in_trainer(self):
        tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
        bare = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        ]
        assert bare == [], f"bare except at {bare}"


class TestCheckpointDurability:
    """Atomic writes and a resumable `latest` (paddle required)."""

    @pytest.fixture()
    def tiny_trainer(self, tmp_path):
        paddle = pytest.importorskip("paddle")
        trainer = VINFineTuner.__new__(VINFineTuner)
        trainer.output_dir = tmp_path
        trainer.model = paddle.nn.Linear(4, 2)
        trainer.optimizer = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=trainer.model.parameters()
        )
        trainer.global_step = 123
        trainer.best_accuracy = 0.5
        trainer.config = {"probe": True}
        return trainer

    def test_save_latest_writes_weights_optimizer_and_resume_info(
        self, tiny_trainer, tmp_path
    ):
        tiny_trainer._save_latest(epoch=7)

        assert (tmp_path / "latest.pdparams").is_file()
        assert (tmp_path / "latest.pdopt").is_file()
        info = json.loads((tmp_path / "latest_info.json").read_text())
        assert info["epoch"] == 7
        assert info["global_step"] == 123
        assert info["best_accuracy"] == 0.5

    def test_saves_leave_no_temp_files_behind(self, tiny_trainer, tmp_path):
        tiny_trainer._save_latest(epoch=1)
        tiny_trainer.save_checkpoint(epoch=1, is_best=True)
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []
        assert (tmp_path / "best_accuracy.pdparams").is_file()
        assert (tmp_path / "epoch_1.pdparams").is_file()
        assert json.loads((tmp_path / "epoch_1_info.json").read_text())["epoch"] == 1

    def test_resume_from_latest_restores_epoch_counter(self, tiny_trainer, tmp_path):
        """
        WAS BROKEN: `latest` had no info json, so load_checkpoint left
        current_epoch at 0 and a resume silently restarted from epoch 1
        (re-running warmup). The info file must round-trip.
        """
        tiny_trainer._save_latest(epoch=9)

        fresh = VINFineTuner.__new__(VINFineTuner)
        fresh.model = tiny_trainer.model
        fresh.optimizer = tiny_trainer.optimizer
        fresh.current_epoch = 0
        fresh.global_step = 0
        fresh.best_accuracy = 0.0
        fresh.load_checkpoint(str(tmp_path / "latest"))

        assert fresh.current_epoch == 9
        assert fresh.global_step == 123
        assert fresh.best_accuracy == 0.5


class TestLossAwareEarlyStopping:
    """
    Readiness item C3: accuracy alone is a defective stopping signal for
    CTC (it sits at 0 through the whole blank-collapse phase - observed
    live: val_loss fell 13.6 -> 0.91 over 17 epochs with exact-match 0.0).
    The counter must also reset while validation loss keeps falling.
    """

    def test_falling_loss_resets_counter_while_accuracy_flat(self):
        improved, counter = update_early_stopping(
            val_accuracy=0.0, previous_best=0.0,
            epochs_without_improvement=16, min_delta=0.001,
            val_loss=0.91, best_val_loss=0.95, loss_min_delta=0.005,
        )
        assert improved is True
        assert counter == 0

    def test_flat_loss_and_flat_accuracy_increment(self):
        improved, counter = update_early_stopping(
            val_accuracy=0.0, previous_best=0.0,
            epochs_without_improvement=3, min_delta=0.001,
            val_loss=0.95, best_val_loss=0.95, loss_min_delta=0.005,
        )
        assert improved is False
        assert counter == 4

    def test_loss_min_delta_boundary_is_strict(self):
        improved, _ = update_early_stopping(
            val_accuracy=0.0, previous_best=0.0,
            epochs_without_improvement=0, min_delta=0.001,
            val_loss=0.945, best_val_loss=0.95, loss_min_delta=0.005,
        )
        assert improved is False  # exactly loss_min_delta is not enough

    def test_without_loss_arguments_behaviour_is_accuracy_only(self):
        """Backward compatibility: the 4-argument form still works."""
        improved, counter = update_early_stopping(
            val_accuracy=0.5, previous_best=0.4,
            epochs_without_improvement=9, min_delta=0.001,
        )
        assert improved is True and counter == 0


class TestConfigValidation:
    """Readiness item C9: named errors at startup, all problems at once."""

    def test_repo_default_config_validates(self):
        import yaml
        from src.vin_ocr.training.finetune_paddleocr import validate_config
        config = yaml.safe_load(
            (REPO_ROOT / "configs" / "vin_finetune_config.yml").read_text()
        )
        validate_config(config)  # must not raise

    def test_all_problems_reported_at_once(self):
        from src.vin_ocr.training.finetune_paddleocr import (
            ConfigValidationError,
            validate_config,
            _REQUIRED_CONFIG_KEYS,
        )
        with pytest.raises(ConfigValidationError) as excinfo:
            validate_config({})
        # every required key is reported, not just the first
        assert str(excinfo.value).count("missing key") == len(_REQUIRED_CONFIG_KEYS)

    def test_type_errors_are_named(self):
        from src.vin_ocr.training.finetune_paddleocr import (
            ConfigValidationError,
            validate_config,
        )
        with pytest.raises(ConfigValidationError) as excinfo:
            validate_config({'Global': {'epoch_num': 'thirty'}})
        assert "Global.epoch_num" in str(excinfo.value)
        assert "str" in str(excinfo.value)

    def test_collapse_region_learning_rate_is_rejected(self):
        """Measured: >~2e-3 pins CTC loss at ln(num_classes) permanently."""
        import yaml
        from src.vin_ocr.training.finetune_paddleocr import (
            ConfigValidationError,
            validate_config,
        )
        config = yaml.safe_load(
            (REPO_ROOT / "configs" / "vin_finetune_config.yml").read_text()
        )
        config['Optimizer']['lr']['learning_rate'] = 0.003
        with pytest.raises(ConfigValidationError) as excinfo:
            validate_config(config)
        assert "blank basin" in str(excinfo.value)


class TestDatasetSkipsAreBounded:
    """Readiness item C6: unreadable images must not recurse forever."""

    @pytest.fixture()
    def dataset_factory(self, tmp_path):
        paddle = pytest.importorskip("paddle")
        import numpy as np
        import cv2
        from src.vin_ocr.core.charset import load_char_dict
        from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset

        char_to_idx, _ = load_char_dict("configs/vin_dict.txt")

        def build(good: int, corrupt: int):
            lines = []
            for i in range(good):
                name = f"good_{i}.jpg"
                img = np.full((64, 320, 3), 128, np.uint8)
                cv2.putText(img, "SAL1A2A40SA606662", (5, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imwrite(str(tmp_path / name), img)
                lines.append(f"{name}\tSAL1A2A40SA606662")
            for i in range(corrupt):
                name = f"corrupt_{i}.jpg"
                (tmp_path / name).write_bytes(b"this is not a jpeg")
                lines.append(f"{name}\tSAL1A2A40SA606662")
            labels = tmp_path / "labels.txt"
            labels.write_text("\n".join(lines) + "\n")
            return VINRecognitionDataset(
                data_dir=str(tmp_path), label_file=str(labels),
                char_dict=char_to_idx, is_training=False,
            )

        return build

    def test_corrupt_image_skips_to_next_readable(self, dataset_factory):
        ds = dataset_factory(good=1, corrupt=1)
        # index 1 is the corrupt file; item must come from the good one
        item = ds[1]
        assert item['text'] == "SAL1A2A40SA606662"
        assert item['image'].shape[0] == 3  # CHW, real tensor

    def test_all_corrupt_raises_instead_of_recursing(self, dataset_factory):
        ds = dataset_factory(good=0, corrupt=3)
        with pytest.raises(RuntimeError) as excinfo:
            ds[0]
        assert "No readable image" in str(excinfo.value)


class TestDecodeConfidence:
    """Readiness item C5: confidence covers emitted timesteps only."""

    @pytest.fixture()
    def decoder(self):
        paddle = pytest.importorskip("paddle")
        from src.vin_ocr.core.charset import load_char_dict
        trainer = VINFineTuner.__new__(VINFineTuner)
        char_to_idx, idx_to_char = load_char_dict("configs/vin_dict.txt")
        trainer.idx_to_char = idx_to_char
        trainer.char_dict = char_to_idx
        return trainer, char_to_idx

    def test_all_blank_output_has_zero_confidence(self, decoder):
        import numpy as np
        import paddle
        trainer, _ = decoder
        logits = np.full((1, 10, 34), -10.0, dtype='float32')
        logits[:, :, 0] = 10.0  # blank everywhere, very confidently
        texts, confs = trainer._ctc_greedy_decode_with_confidence(
            paddle.to_tensor(logits)
        )
        assert texts == ['']
        assert confs == [0.0], (
            "an empty decode must report 0 confidence, not the confidence "
            "of predicting nothing"
        )

    def test_emitted_sequence_confidence_comes_from_kept_steps(self, decoder):
        import numpy as np
        import paddle
        trainer, char_to_idx = decoder
        seq = [0, char_to_idx['1'], 0, char_to_idx['M'], 0, char_to_idx['8'], 0]
        logits = np.full((1, len(seq), 34), -10.0, dtype='float32')
        for t, idx in enumerate(seq):
            logits[0, t, idx] = 10.0
        texts, confs = trainer._ctc_greedy_decode_with_confidence(
            paddle.to_tensor(logits)
        )
        assert texts == ['1M8']
        assert confs[0] > 0.99


class TestConfigValidationHoles:
    """
    Found by adversarial self-audit (2026-08-18): two confirmed-by-execution
    holes in validate_config as first shipped.
    """

    def _base(self):
        import yaml
        return yaml.safe_load(
            (REPO_ROOT / "configs" / "vin_finetune_config.yml").read_text()
        )

    def test_none_value_is_rejected_not_skipped(self):
        """A null VALUE used to pass: only absent keys were reported."""
        from src.vin_ocr.training.finetune_paddleocr import (
            ConfigValidationError, validate_config,
        )
        config = self._base()
        config['Global']['epoch_num'] = None
        with pytest.raises(ConfigValidationError) as excinfo:
            validate_config(config)
        assert "Global.epoch_num" in str(excinfo.value)

    def test_bool_is_not_an_acceptable_int(self):
        """bool subclasses int; `epoch_num: true` used to validate."""
        from src.vin_ocr.training.finetune_paddleocr import (
            ConfigValidationError, validate_config,
        )
        config = self._base()
        config['Global']['epoch_num'] = True
        with pytest.raises(ConfigValidationError) as excinfo:
            validate_config(config)
        assert "bool" in str(excinfo.value)


class TestCorruptionThreshold:
    """C6 as SPECIFIED: warn once per corrupt path, abort past 5%."""

    def test_each_corrupt_path_warned_once_and_threshold_aborts(self, tmp_path, caplog):
        paddle = pytest.importorskip("paddle")
        import numpy as np
        import cv2
        from src.vin_ocr.core.charset import load_char_dict
        from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset

        char_to_idx, _ = load_char_dict("configs/vin_dict.txt")
        lines = []
        for i in range(10):
            name = f"good_{i}.jpg"
            img = np.full((64, 320, 3), 128, np.uint8)
            cv2.imwrite(str(tmp_path / name), img)
            lines.append(f"{name}\tSAL1A2A40SA606662")
        for i in range(2):  # 2/12 corrupt > 5% threshold
            name = f"corrupt_{i}.jpg"
            (tmp_path / name).write_bytes(b"junk")
            lines.append(f"{name}\tSAL1A2A40SA606662")
        (tmp_path / "labels.txt").write_text("\n".join(lines) + "\n")
        ds = VINRecognitionDataset(
            data_dir=str(tmp_path), label_file=str(tmp_path / "labels.txt"),
            char_dict=char_to_idx, is_training=False,
        )

        # Accessing the first corrupt index scans forward, discovering BOTH
        # corrupt files before reaching a readable one - so the threshold
        # (max(1, 5% of 12) = 1) is exceeded within this single access.
        with pytest.raises(RuntimeError) as excinfo:
            ds[10]
        assert "unreadable" in str(excinfo.value)

        # each corrupt path was warned exactly once during the scan
        for name in ("corrupt_0", "corrupt_1"):
            warnings = [r for r in caplog.records if name in r.getMessage()]
            assert len(warnings) == 1, name


class TestTrackingFallback:
    """_run_tracked must run untracked - loudly - when tracking is absent."""

    def test_training_proceeds_without_tracking(self, monkeypatch, capsys):
        import builtins
        from src.vin_ocr.training import finetune_paddleocr as ft

        real_import = builtins.__import__

        def no_tracking(name, *args, **kwargs):
            if name.startswith("src.vin_ocr.tracking"):
                raise ImportError("tracking extra not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_tracking)

        calls = {}

        class StubTrainer:
            def train(self, resume_from=None):
                calls['trained'] = True

        class Args:
            resume = None
            config = "configs/vin_finetune_config.yml"

        ft._run_tracked(StubTrainer(), {}, Args())

        assert calls.get('trained') is True
        assert "TRACKING DISABLED" in capsys.readouterr().out


class TestCharMetricsInvariantsSurviveOptimization:
    """Conservation checks must be explicit raises, not -O-strippable asserts."""

    def test_invariants_are_not_bare_asserts(self):
        source = (REPO_ROOT / "src" / "vin_ocr" / "core" / "char_metrics.py").read_text()
        import ast as astmod
        tree = astmod.parse(source)
        fn = next(
            n for n in astmod.walk(tree)
            if isinstance(n, astmod.FunctionDef) and n.name == "alignment_counts"
        )
        asserts = [n for n in astmod.walk(fn) if isinstance(n, astmod.Assert)]
        raises = [n for n in astmod.walk(fn) if isinstance(n, astmod.Raise)]
        assert asserts == [], "conservation invariants would vanish under -O"
        assert len(raises) >= 2

    def test_metric_zips_are_strict(self):
        """A pred/ref length mismatch must raise, never silently truncate."""
        from src.vin_ocr.evaluation.evaluate import calculate_character_metrics
        with pytest.raises(ValueError):
            calculate_character_metrics(["ABC"], ["ABC", "DEF"])
