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
            # input lengths are flag-gated: full-T by default (padding
            # learns blanks; width-free inference contract), per-sample
            # content-width lengths under Global.ctc_mask_padding.
            assert "ctc_mask_padding" in source, method
            assert "ctc_input_lengths(" in source, method
            assert "dtype='int64'" in source, method
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
        # M15: checkpoint info persists the best-metric baselines and the
        # early-stop patience counter, so they are part of trainer state.
        trainer.best_val_loss = 0.9
        trainer.best_val_char_accuracy = 0.4
        trainer.epochs_without_improvement = 2
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
        (re-running warmup). The info file must round-trip - including the
        best-metric baselines and patience counter (M15), and identically
        for BOTH the extensionless and explicit .pdparams forms (M16: the
        latter used to derive `latest.pdparams_info.json`, never written,
        silently resetting the epoch counter).
        """
        tiny_trainer._save_latest(epoch=9)

        for checkpoint_arg in ("latest", "latest.pdparams"):
            fresh = VINFineTuner.__new__(VINFineTuner)
            fresh.model = tiny_trainer.model
            fresh.optimizer = tiny_trainer.optimizer
            fresh.current_epoch = 0
            fresh.global_step = 0
            fresh.best_accuracy = 0.0
            restored = fresh.load_checkpoint(str(tmp_path / checkpoint_arg))

            assert fresh.current_epoch == 9, checkpoint_arg
            assert fresh.global_step == 123, checkpoint_arg
            assert fresh.best_accuracy == 0.5, checkpoint_arg
            # M15: best baselines and patience survive the resume.
            assert fresh.best_val_loss == 0.9, checkpoint_arg
            assert fresh.best_val_char_accuracy == 0.4, checkpoint_arg
            assert fresh.epochs_without_improvement == 2, checkpoint_arg
            # load_checkpoint reports what it restored (H6 relies on this
            # being checkable instead of assumed).
            assert restored["resume_state_restored"] is True
            assert restored["optimizer_restored"] is True
            assert restored["epoch"] == 9


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
        pytest.importorskip("paddle")
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
        pytest.importorskip("paddle")
        from src.vin_ocr.core.charset import load_char_dict
        trainer = VINFineTuner.__new__(VINFineTuner)
        char_to_idx, idx_to_char = load_char_dict("configs/vin_dict.txt")
        trainer.idx_to_char = idx_to_char
        trainer.char_dict = char_to_idx
        trainer.config = {}  # decoder reads Global.image_width with a default
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
        pytest.importorskip("paddle")
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


class TestImageReadsAreStreamCompatible:
    """
    DagsHub streaming regression (2026-08-22): install_hooks patches only
    builtins.open / io.open / os.stat. cv2.imread opens files in native
    OpenCV code, so under streaming a remote-only image passed the hooked
    Path.exists() check (phantom stat) and then read as None - the sample
    was misclassified as corrupt and the run aborted at the 5% threshold.
    The dataset must therefore read image BYTES through Python's open()
    and decode with cv2.imdecode.
    """

    @pytest.fixture()
    def small_dataset(self, tmp_path):
        pytest.importorskip("paddle")
        import numpy as np
        import cv2
        from src.vin_ocr.core.charset import load_char_dict
        from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset

        char_to_idx, _ = load_char_dict("configs/vin_dict.txt")
        lines = []
        for i in range(2):
            name = f"good_{i}.jpg"
            img = np.full((64, 320, 3), 128, np.uint8)
            cv2.putText(img, "SAL1A2A40SA606662", (5, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imwrite(str(tmp_path / name), img)
            lines.append(f"{name}\tSAL1A2A40SA606662")
        (tmp_path / "corrupt.jpg").write_bytes(b"this is not a jpeg")
        lines.append("corrupt.jpg\tSAL1A2A40SA606662")
        (tmp_path / "labels.txt").write_text("\n".join(lines) + "\n")
        return VINRecognitionDataset(
            data_dir=str(tmp_path), label_file=str(tmp_path / "labels.txt"),
            char_dict=char_to_idx, is_training=False,
        )

    def test_read_image_matches_cv2_imread(self, small_dataset):
        """Same bytes, same decoder: pixel-identical to cv2.imread."""
        import numpy as np
        import cv2
        good = next(p for p, _ in small_dataset.samples if "good_0" in p)
        assert np.array_equal(small_dataset._read_image(good), cv2.imread(good))

    def test_read_image_keeps_imread_none_contract(self, small_dataset, tmp_path):
        """Corrupt bytes and missing files both map to None, like imread."""
        corrupt = next(p for p, _ in small_dataset.samples if "corrupt" in p)
        assert small_dataset._read_image(corrupt) is None
        assert small_dataset._read_image(str(tmp_path / "absent.jpg")) is None
        (tmp_path / "empty.jpg").write_bytes(b"")
        assert small_dataset._read_image(str(tmp_path / "empty.jpg")) is None

    def test_getitem_reads_bytes_through_builtins_open(self, small_dataset, monkeypatch):
        """The image byte read must be interceptable by install_hooks."""
        import builtins
        opened = []
        real_open = builtins.open

        def counting_open(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", counting_open)
        item = small_dataset[0]
        assert item["image"].shape[0] == 3
        assert any(p.endswith("good_0.jpg") for p in opened), (
            "image bytes were not read through builtins.open - native "
            "file I/O cannot be intercepted by dagshub install_hooks"
        )

    def test_remote_only_file_served_by_hooked_open(self, small_dataset, monkeypatch):
        """
        The install_hooks scenario end-to-end: the file is DELETED from
        disk and served from a patched builtins.open, exactly the way the
        hooks materialize remote-only files. cv2.imread returns None here;
        the dataset must still produce a real tensor.
        """
        import builtins
        import io as _io
        img_path = Path(small_dataset.samples[0][0])
        data = img_path.read_bytes()
        img_path.unlink()
        real_open = builtins.open

        def hooked_open(file, mode="r", *args, **kwargs):
            if str(file) == str(img_path):
                return _io.BytesIO(data)
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", hooked_open)
        item = small_dataset[0]
        assert item["image"].shape[0] == 3
        assert item["text"] == "SAL1A2A40SA606662"

    def test_missing_file_skips_like_unreadable(self, small_dataset):
        """A file that vanishes after load resolves to the next sample."""
        Path(small_dataset.samples[0][0]).unlink()
        item = small_dataset[0]
        assert item["text"] == "SAL1A2A40SA606662"
        assert item["image"].shape[0] == 3

    def test_no_native_imread_left_in_dataset(self):
        """cv2.imread must not reappear inside VINRecognitionDataset."""
        import textwrap
        from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset
        src = textwrap.dedent(inspect.getsource(VINRecognitionDataset))
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Attribute) and node.attr == "imread":
                pytest.fail(
                    "cv2.imread found in VINRecognitionDataset: native "
                    "file I/O bypasses dagshub streaming hooks"
                )


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


class TestCTCInputLengths:
    """
    Deep-dive fix: per-sample CTC input lengths from content width. Full-T
    lengths force the loss to demand emissions over black padding - the
    mechanism behind the observed 40-char digit tail over an 80-timestep
    sequence.
    """

    def test_lengths_follow_content_width(self):
        from src.vin_ocr.training.finetune_paddleocr import ctc_input_lengths
        # stride = 320/80 = 4px per timestep
        lengths = ctc_input_lengths(
            valid_widths=[320, 160, 68, 4],
            total_timesteps=80,
            image_width=320,
            label_lengths=[17, 17, 17, 17],
        )
        assert lengths == [80, 40, 17, 17]  # last two clamped up to label len

    def test_never_exceeds_total_timesteps(self):
        from src.vin_ocr.training.finetune_paddleocr import ctc_input_lengths
        assert ctc_input_lengths([9999], 80, 320, [17]) == [80]

    def test_never_below_label_length(self):
        """CTC requires input_length >= label_length."""
        from src.vin_ocr.training.finetune_paddleocr import ctc_input_lengths
        assert ctc_input_lengths([1], 80, 320, [17]) == [17]

    def test_mismatched_batch_raises(self):
        from src.vin_ocr.training.finetune_paddleocr import ctc_input_lengths
        with pytest.raises(ValueError):
            ctc_input_lengths([320, 160], 80, 320, [17])

    def test_dataset_emits_valid_width(self, tmp_path):
        pytest.importorskip("paddle")
        import numpy as np
        import cv2
        from src.vin_ocr.core.charset import load_char_dict
        from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset

        char_to_idx, _ = load_char_dict("configs/vin_dict.txt")
        img = np.full((96, 640, 3), 128, np.uint8)  # wide plate crop
        cv2.imwrite(str(tmp_path / "a.jpg"), img)
        (tmp_path / "labels.txt").write_text("a.jpg\tSAL1A2A40SA606662\n")
        ds = VINRecognitionDataset(
            data_dir=str(tmp_path), label_file=str(tmp_path / "labels.txt"),
            char_dict=char_to_idx, is_training=False,
        )
        item = ds[0]
        assert 'valid_width' in item
        width = int(item['valid_width'][0])
        assert 0 < width <= 320


class TestBestValLossCheckpoint:
    """Selection must not depend on exact-match becoming non-zero."""

    def test_best_val_loss_checkpoint_written_with_provenance(self, tmp_path):
        paddle = pytest.importorskip("paddle")
        trainer = VINFineTuner.__new__(VINFineTuner)
        trainer.output_dir = tmp_path
        trainer.model = paddle.nn.Linear(4, 2)
        trainer.optimizer = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=trainer.model.parameters())
        trainer.global_step = 7
        trainer.current_epoch = 3
        trainer.best_accuracy = 0.0
        trainer.best_val_loss = 0.862
        trainer.best_val_char_accuracy = 0.0
        trainer.epochs_without_improvement = 0
        trainer.config = {"probe": True}

        trainer._save_best_val_loss_model()

        assert (tmp_path / "best_val_loss.pdparams").is_file()
        info = json.loads((tmp_path / "best_val_loss_info.json").read_text())
        assert info["selection_metric"] == "val_loss"
        assert info["selection_value"] == 0.862
        assert info["epoch"] == 3

    def test_best_accuracy_checkpoint_carries_provenance_too(self, tmp_path):
        """The six-month-stale-best hazard: 'best' must say when/what."""
        paddle = pytest.importorskip("paddle")
        trainer = VINFineTuner.__new__(VINFineTuner)
        trainer.output_dir = tmp_path
        trainer.model = paddle.nn.Linear(4, 2)
        trainer.optimizer = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=trainer.model.parameters())
        trainer.global_step = 9
        trainer.current_epoch = 5
        trainer.best_accuracy = 0.25
        trainer.best_val_loss = 1.0
        trainer.best_val_char_accuracy = 0.0
        trainer.epochs_without_improvement = 0
        trainer.config = {"probe": True}

        trainer._save_best_model()

        info = json.loads((tmp_path / "best_accuracy_info.json").read_text())
        assert info["selection_metric"] == "val_exact_match_accuracy"
        assert info["selection_value"] == 0.25


class TestTrainerMetricsAreCanonical:
    """
    Deep-dive finding: _calculate_detailed_metrics - the producer of
    training_metrics.json, which the Optuna tuner scores from - inlined one
    more positional char loop with the invalid-char FP hole, plus a fifth
    inline Levenshtein.
    """

    def _metrics(self, predictions, gts):
        trainer = VINFineTuner.__new__(VINFineTuner)
        return VINFineTuner._calculate_detailed_metrics(
            trainer, predictions, gts, [0.9] * len(predictions)
        )

    def test_char_metrics_match_canonical(self):
        pytest.importorskip("paddle")
        from src.vin_ocr.core.char_metrics import char_level_metrics

        gt = "SAL1A2A40SA606662"
        preds = [gt[:5], "*" + gt[:-1], gt]
        out = self._metrics(preds, [gt] * 3)
        canonical = char_level_metrics([(p[:17], gt) for p in preds])
        cl = out['character_level']
        assert cl['f1_micro'] == pytest.approx(canonical.f1_micro)
        assert cl['character_accuracy'] == pytest.approx(canonical.char_accuracy)
        assert cl['micro_precision'] == pytest.approx(canonical.precision)

    def test_shift_probe_no_longer_collapses(self):
        pytest.importorskip("paddle")
        gt = "SAL1A2A40SA606662"
        out = self._metrics(["*" + gt[:-1]], [gt])
        # positional scoring gave ~0.059 for this 94%-correct prediction
        assert out['character_level']['character_accuracy'] > 0.8

    def test_no_inline_levenshtein_remains(self):
        source = TRAINER_PATH.read_text(encoding="utf-8")
        assert "_levenshtein_distance" not in source

    def test_no_debug_prints_in_metrics_path(self):
        """AST check: comments legitimately DESCRIBE the removed hack."""
        import inspect as _inspect
        import textwrap
        source = textwrap.dedent(
            _inspect.getsource(VINFineTuner._calculate_comprehensive_metrics)
        )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == 'print':
                pytest.fail(f"print() in the metrics path at line {node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr == 'path' \
                    and isinstance(node.value, ast.Name) and node.value.id == 'sys':
                pytest.fail(f"sys.path manipulation at line {node.lineno}")


class TestPaddingMaskIsOptInAndLegacyCompatible:
    """
    Measured on the legacy checkpoint: it emits 10/17 characters INSIDE the
    padded region (timesteps 64-79 of 80 at valid_T=64) - full-T-trained
    models anchor emissions anywhere. Masking must therefore be opt-in:
    default full-T supervision keeps warm starts valid and teaches blanks
    over padding; the mask flag exists for fresh anchored-emission runs.
    """

    def test_default_config_does_not_mask(self):
        import yaml
        config = yaml.safe_load(
            (REPO_ROOT / "configs" / "vin_finetune_config.yml").read_text()
        )
        assert not config['Global'].get('ctc_mask_padding', False)

    def test_decode_is_unmasked_when_widths_none(self):
        paddle = pytest.importorskip("paddle")
        import numpy as np
        from src.vin_ocr.core.charset import load_char_dict

        trainer = VINFineTuner.__new__(VINFineTuner)
        char_to_idx, idx_to_char = load_char_dict("configs/vin_dict.txt")
        trainer.idx_to_char = idx_to_char
        trainer.config = {}
        # emission at timestep 70 of 80 - inside would-be padding
        logits = np.full((1, 80, 34), -10.0, dtype='float32')
        logits[0, 70, char_to_idx['M']] = 10.0
        texts, _ = trainer._ctc_greedy_decode_with_confidence(
            paddle.to_tensor(logits), valid_widths=None
        )
        assert texts == ['M'], "unmasked decode must see late emissions"

    def test_decode_masks_when_widths_given(self):
        paddle = pytest.importorskip("paddle")
        import numpy as np
        from src.vin_ocr.core.charset import load_char_dict

        trainer = VINFineTuner.__new__(VINFineTuner)
        char_to_idx, idx_to_char = load_char_dict("configs/vin_dict.txt")
        trainer.idx_to_char = idx_to_char
        trainer.config = {}
        logits = np.full((1, 80, 34), -10.0, dtype='float32')
        logits[0, 70, char_to_idx['M']] = 10.0  # beyond valid_T=64
        logits[0, 10, char_to_idx['S']] = 10.0  # inside content
        texts, _ = trainer._ctc_greedy_decode_with_confidence(
            paddle.to_tensor(logits), valid_widths=[253]  # ceil(253/4)=64
        )
        assert texts == ['S'], "masked decode must ignore pad-region emissions"


class TestTracingIsOptionalAndArmed:
    """Tracing must be a no-op without enable_tracing(), loud when broken."""

    def test_traced_is_passthrough_when_disabled(self):
        from src.vin_ocr.tracking import tracing
        tracing.disable_tracing()

        calls = {}

        @tracing.traced(name="probe", span_type="CHAIN")
        def fn(x):
            calls['ran'] = True
            return x + 1

        assert fn(1) == 2 and calls['ran']
        assert not tracing.is_enabled()

    def test_enable_tracing_reports_failure_not_crash(self, monkeypatch):
        import builtins
        from src.vin_ocr.tracking import tracing
        tracing.disable_tracing()

        real_import = builtins.__import__

        def no_mlflow(name, *args, **kwargs):
            if name == 'mlflow' or name.startswith('mlflow.'):
                raise ImportError('mlflow not installed')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', no_mlflow)
        assert tracing.enable_tracing() is False
        assert not tracing.is_enabled()

    def test_pipeline_imports_without_tracing_enabled(self):
        """The decorated pipeline must work untraced (mlflow-free path)."""
        from src.vin_ocr.tracking import tracing
        tracing.disable_tracing()
        from src.vin_ocr.pipeline.vin_pipeline import VINPostProcessor
        result = VINPostProcessor().process("*SAL1A2A40SA606662*")
        assert result['vin'] == "SAL1A2A40SA606662"


class TestCharAccuracyAwareSelection:
    """
    Measured on stage-3b: val loss selected epoch 19 while char accuracy
    improved through epoch 44 and won on held-out test. Selection and
    stopping must therefore track char accuracy as well.
    """

    def test_char_improvement_resets_counter_when_loss_flat(self):
        improved, counter = update_early_stopping(
            val_accuracy=0.0, previous_best=0.0,
            epochs_without_improvement=12, min_delta=0.001,
            val_loss=0.85, best_val_loss=0.80, loss_min_delta=0.005,
            val_char_accuracy=0.71, best_val_char_accuracy=0.70,
            char_min_delta=0.002,
        )
        assert improved is True and counter == 0

    def test_char_min_delta_boundary_is_strict(self):
        improved, _ = update_early_stopping(
            val_accuracy=0.0, previous_best=0.0,
            epochs_without_improvement=0, min_delta=0.001,
            val_char_accuracy=0.702, best_val_char_accuracy=0.700,
            char_min_delta=0.002,
        )
        assert improved is False

    def test_omitting_char_args_is_backward_compatible(self):
        improved, counter = update_early_stopping(
            val_accuracy=0.0, previous_best=0.0,
            epochs_without_improvement=1, min_delta=0.001,
            val_loss=0.70, best_val_loss=0.80, loss_min_delta=0.005,
        )
        assert improved is True and counter == 0

    def test_best_char_accuracy_checkpoint_carries_provenance(self, tmp_path):
        paddle = pytest.importorskip("paddle")
        trainer = VINFineTuner.__new__(VINFineTuner)
        trainer.output_dir = tmp_path
        trainer.model = paddle.nn.Linear(4, 2)
        trainer.optimizer = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=trainer.model.parameters())
        trainer.global_step = 11
        trainer.current_epoch = 44
        trainer.best_accuracy = 0.0
        trainer.best_val_loss = 0.82
        trainer.best_val_char_accuracy = 0.7549
        trainer.epochs_without_improvement = 0
        trainer.config = {"probe": True}

        trainer._save_best_char_accuracy_model()

        assert (tmp_path / "best_char_accuracy.pdparams").is_file()
        info = json.loads((tmp_path / "best_char_accuracy_info.json").read_text())
        assert info["selection_metric"] == "val_char_accuracy"
        assert info["selection_value"] == 0.7549
        assert info["epoch"] == 44

    def test_validate_returns_char_accuracy(self):
        import inspect
        source = inspect.getsource(VINFineTuner.validate)
        assert "return avg_loss, accuracy, val_char_accuracy" in source


class TestInterruptSemantics:
    """C7: an interrupted run must be distinguishable from a completed one,
    and resuming an already-finished run must terminate cleanly."""

    def test_shutdown_paths_set_interrupted_flag(self):
        import ast
        import inspect
        import sys
        module_ast = ast.parse(
            inspect.getsource(sys.modules[VINFineTuner.__module__]))
        cls = next(n for n in ast.walk(module_ast)
                   if isinstance(n, ast.ClassDef) and n.name == "VINFineTuner")
        marked = []
        for func in (n for n in ast.walk(cls)
                     if isinstance(n, ast.FunctionDef)
                     and n.name in ("train", "train_epoch")):
            for node in ast.walk(func):
                if (isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Attribute)
                                and t.attr == "interrupted"
                                for t in node.targets)):
                    marked.append(func.name)
        assert set(marked) == {"train", "train_epoch"}, (
            f"interrupted flag must be set on both shutdown paths, got {marked}"
        )

    def test_tracked_run_tags_interruption(self):
        import inspect
        import sys
        module = sys.modules[VINFineTuner.__module__]
        source = inspect.getsource(module._run_tracked)
        assert "interrupted" in source and "set_tags" in source, (
            "_run_tracked must tag interrupted runs in the tracking store"
        )

    def test_completion_variables_exist_when_epoch_loop_runs_zero_times(self):
        """Resuming a checkpoint that already reached epoch_num historically
        raised NameError: the completion path read train_loss/val_loss/
        val_accuracy, which were only assigned inside the epoch loop."""
        import inspect
        import textwrap
        source = textwrap.dedent(inspect.getsource(VINFineTuner.train))
        loop_pos = source.find("for epoch in range(self.current_epoch + 1")
        assert loop_pos > 0
        prefix = source[:loop_pos]
        for name in ("train_loss", "val_loss", "val_accuracy"):
            assert f"{name} = " in prefix, (
                f"{name} must be initialized before the epoch loop; "
                f"a zero-iteration resume otherwise crashes at completion"
            )
