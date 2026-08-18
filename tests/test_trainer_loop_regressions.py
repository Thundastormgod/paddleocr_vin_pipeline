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
