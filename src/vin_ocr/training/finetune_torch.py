"""
Apple-GPU (MPS) training for VIN recognition - PyTorch trainer.

This is the long-term training stack for this machine (decision record in
torch_rosetta.py: paddle has no Metal backend; measured paddle-CPU ceiling
~1.1 of 8 cores). It carries over every discipline the paddle trainer
earned the hard way, by importing the same single-source implementations
wherever they are framework-agnostic:

- data: wraps `VINRecognitionDataset` (identical preprocessing bytes);
- per-sample CTC input lengths: `ctc_input_lengths` (pure function);
- early stopping: `update_early_stopping` (pure function);
- decode + metrics: `core.charset.ctc_greedy_decode` +
  `core.char_metrics.char_level_metrics` (the canonical implementations);
- provenance: `src.vin_ocr.tracking.start_run` (commit, env, data hashes);
- MLflow 3 LoggedModel workflow on completion: the trained network is
  logged as a LoggedModel with its params, evaluation metrics are linked
  to `model_id` + a named Dataset entity built from the val label file,
  and the model is registered - the same pattern `model_registry.py`
  applies to paddle checkpoints.

MPS specifics, all measured on this machine (2026-08-20):
- `aten::_ctc_loss` is NOT implemented on MPS -> the network runs on MPS
  and CTC loss runs on CPU log-probs; autograd bridges devices, so the
  backward still updates MPS parameters. The bridge tensors are tiny
  ([T=80, B, 34] floats).
- Non-finite losses RAISE (repo rule: silent inf/NaN averaging is how a
  broken geometry once hid inside epoch means).

Usage:
    python -m src.vin_ocr.training.finetune_torch \
        --config configs/vin_rosetta_torch_config.yml [--resume PATH.pt]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.vin_ocr.core.char_metrics import char_level_metrics
from src.vin_ocr.core.charset import ctc_greedy_decode, load_char_dict, num_classes
from src.vin_ocr.training.torch_rosetta import (
    INPUT_WIDTH,
    OUTPUT_TIMESTEPS,
    RosettaResNet34Torch,
)

MAX_SAFE_LEARNING_RATE = 2e-3  # measured: 3e-3 permanently collapsed CTC


# =============================================================================
# CONFIG
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML training config."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    config = yaml.safe_load(path.read_text())
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Reject configs that measurement has shown cannot train.

    Raises:
        ValueError: On any out-of-range or missing setting.
    """
    global_config = config.get("Global", {})
    optimizer_config = config.get("Optimizer", {})
    lr_config = optimizer_config.get("lr", {})

    epochs = global_config.get("epoch_num")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"Global.epoch_num must be a positive int, got {epochs!r}")

    learning_rate = lr_config.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError(f"lr.learning_rate must be > 0, got {learning_rate!r}")
    if learning_rate > MAX_SAFE_LEARNING_RATE:
        raise ValueError(
            f"lr.learning_rate={learning_rate} exceeds the measured safe bound "
            f"{MAX_SAFE_LEARNING_RATE} (3e-3 collapsed CTC permanently; "
            f"LOGBOOK 2026-08-18)"
        )

    warmup = lr_config.get("warmup_epoch", 0)
    if not isinstance(warmup, int) or warmup < 0 or warmup >= epochs:
        raise ValueError(f"warmup_epoch must be in [0, epoch_num), got {warmup!r}")

    for section in ("Train", "Eval"):
        label_files = (config.get(section, {}).get("dataset", {})
                       .get("label_file_list", []))
        if not label_files:
            raise ValueError(f"{section}.dataset.label_file_list is required")

    dict_path = global_config.get("character_dict_path")
    if not dict_path or not Path(dict_path).is_file():
        raise ValueError(f"character_dict_path missing or absent: {dict_path!r}")


def resolve_device(config: Dict[str, Any]) -> torch.device:
    """
    Resolve the training device with an honest failure mode.

    Config Global.device: 'mps' (default) requires a working MPS build;
    'cpu' is allowed for tests/smoke. There is no CUDA on this machine.
    """
    requested = str(config.get("Global", {}).get("device", "mps")).lower()
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Global.device=mps but MPS is unavailable on this host; "
                "set Global.device=cpu explicitly if CPU training is intended"
            )
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported Global.device={requested!r} (mps|cpu)")


def lr_at_epoch(epoch: int, peak_lr: float, warmup_epochs: int,
                total_epochs: int, warmup_start_lr: float = 1e-6,
                eta_min: float = 1e-7) -> float:
    """
    Linear warmup to peak, then cosine decay to eta_min. Pure function.

    Epoch is 1-based (epoch 1 is the first training epoch), matching the
    paddle trainer's schedule semantics so trajectories are comparable.
    """
    assert epoch >= 1 and total_epochs >= 1
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        fraction = epoch / warmup_epochs
        return warmup_start_lr + (peak_lr - warmup_start_lr) * fraction
    # First post-warmup epoch sits at progress 0 (exactly peak); the final
    # epoch reaches progress 1 (exactly eta_min).
    remaining = max(1, total_epochs - warmup_epochs - 1)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs - 1) / remaining))
    return eta_min + (peak_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * progress))


# =============================================================================
# DATA
# =============================================================================

class TorchVINDataset(Dataset):
    """
    torch view over `VINRecognitionDataset`: identical bytes, torch tensors.

    The paddle class is imported for its preprocessing/augmentation ONLY
    (its __getitem__ returns numpy); duplicating that code is how this
    repository once ended up with four drifted metric implementations.
    """

    def __init__(self, data_dir: str, label_file: str,
                 char_dict: Dict[str, int], is_training: bool):
        from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset
        self._inner = VINRecognitionDataset(
            data_dir=data_dir, label_file=label_file,
            char_dict=char_dict, is_training=is_training,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._inner[idx]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(item["image"])),
            "label": torch.from_numpy(np.asarray(item["label"], dtype=np.int64)),
            "length": int(item["length"][0]),
            "valid_width": int(item["valid_width"][0]),
            "text": item["text"],
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "valid_width": [b["valid_width"] for b in batch],
        "text": [b["text"] for b in batch],
    }


# =============================================================================
# LOSS - the MPS/CPU bridge
# =============================================================================

def ctc_loss_cpu_bridge(logits: torch.Tensor, labels: torch.Tensor,
                        label_lengths: torch.Tensor,
                        valid_widths: List[int]) -> torch.Tensor:
    """
    CTC loss for MPS training: log-probs move to CPU, gradient bridges back.

    `aten::_ctc_loss` is not implemented on MPS (verified 2026-08-20,
    torch 2.13.0), so the loss runs on CPU while the network stays on the
    GPU. Per-sample input lengths come from the shared `ctc_input_lengths`
    (full-T supervision is NOT used here: unlike the legacy SVTR
    checkpoints, this model has no emission-anchoring history, and padded
    timesteps carry no signal for a fresh network).

    Raises:
        RuntimeError: If the loss is non-finite (repo rule: never average
            inf/NaN into an epoch mean).
    """
    from src.vin_ocr.training.finetune_paddleocr import ctc_input_lengths

    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2).cpu()  # [T,B,C]
    input_lengths = torch.tensor(
        ctc_input_lengths(
            valid_widths=valid_widths,
            total_timesteps=logits.shape[1],
            image_width=INPUT_WIDTH,
            label_lengths=[int(n) for n in label_lengths],
        ),
        dtype=torch.long,
    )
    loss = F.ctc_loss(
        log_probs, labels.cpu(), input_lengths, label_lengths.cpu(),
        blank=0, reduction="mean", zero_infinity=False,
    )
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"non-finite CTC loss ({float(loss)!r}); refusing to average it "
            f"into the epoch mean - check input lengths vs label lengths"
        )
    return loss


# =============================================================================
# TRAINER
# =============================================================================

class TorchVINTrainer:
    """MPS training loop with the repo's full checkpoint/stopping discipline."""

    _shutdown_requested = False

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = resolve_device(config)
        self.output_dir = Path(config["Global"]["save_model_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        char_to_idx, idx_to_char = load_char_dict(
            config["Global"]["character_dict_path"])
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.num_classes = num_classes(char_to_idx)

        arch = config.get("Architecture", {})
        self.model = RosettaResNet34Torch(
            num_classes=self.num_classes,
            pretrained_backbone=bool(arch.get("pretrained_backbone", True)),
            dropout=float(arch.get("Head", {}).get("dropout", 0.1)),
        ).to(self.device)

        optimizer_config = config.get("Optimizer", {})
        lr_config = optimizer_config.get("lr", {})
        self.peak_lr = float(lr_config.get("learning_rate"))
        self.warmup_epochs = int(lr_config.get("warmup_epoch", 0))
        self.total_epochs = int(config["Global"]["epoch_num"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.peak_lr,
            betas=(float(optimizer_config.get("beta1", 0.9)),
                   float(optimizer_config.get("beta2", 0.999))),
        )

        self.grad_clip_norm = 5.0
        self.global_step = 0
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.best_val_loss = float("inf")
        self.best_val_char_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.interrupted = False
        self.train_losses: List[float] = []

        self._build_dataloaders()
        self._install_signal_handlers()

    # -- setup ----------------------------------------------------------------

    def _build_dataloaders(self) -> None:
        train_dataset_config = self.config["Train"]["dataset"]
        eval_dataset_config = self.config["Eval"]["dataset"]
        train_loader_config = self.config["Train"].get("loader", {})
        eval_loader_config = self.config["Eval"].get("loader", {})

        self.train_dataset = TorchVINDataset(
            data_dir=train_dataset_config["data_dir"],
            label_file=train_dataset_config["label_file_list"][0],
            char_dict=self.char_to_idx, is_training=True,
        )
        self.val_dataset = TorchVINDataset(
            data_dir=eval_dataset_config["data_dir"],
            label_file=eval_dataset_config["label_file_list"][0],
            char_dict=self.char_to_idx, is_training=False,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(train_loader_config.get("batch_size_per_card", 16)),
            shuffle=bool(train_loader_config.get("shuffle", True)),
            drop_last=bool(train_loader_config.get("drop_last", True)),
            num_workers=0, collate_fn=_collate,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=int(eval_loader_config.get("batch_size_per_card", 16)),
            shuffle=False, drop_last=False,
            num_workers=0, collate_fn=_collate,
        )

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            TorchVINTrainer._shutdown_requested = True
            print(f"\nshutdown signal {signum} received; finishing the "
                  f"current batch then checkpointing", flush=True)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    # -- checkpoints ----------------------------------------------------------

    def _atomic_save(self, payload: Dict[str, Any], path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _write_info(self, path: Path, extra: Dict[str, Any]) -> None:
        info = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
            "best_val_loss": (None if math.isinf(self.best_val_loss)
                              else self.best_val_loss),
            "best_val_char_accuracy": self.best_val_char_accuracy,
            "epochs_without_improvement": self.epochs_without_improvement,
            "framework": f"torch-{torch.__version__}",
            "device": str(self.device),
            "config": self.config,
            **extra,
        }
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(info, indent=1, default=str))
        os.replace(tmp_path, path)

    def save_checkpoint(self, name: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }
        self._atomic_save(payload, self.output_dir / f"{name}.pt")
        self._write_info(self.output_dir / f"{name}_info.json", extra or {})

    def resume(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu",
                             weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.current_epoch = int(payload["epoch"])
        self.global_step = int(payload["global_step"])
        info_path = Path(checkpoint_path).with_name(
            Path(checkpoint_path).stem + "_info.json")
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            self.best_accuracy = float(info.get("best_accuracy", 0.0))
            self.best_val_char_accuracy = float(
                info.get("best_val_char_accuracy", 0.0))
            stored_loss = info.get("best_val_loss")
            self.best_val_loss = (float("inf") if stored_loss is None
                                  else float(stored_loss))
            # Early-stop patience must survive resume: a counter silently
            # reset to 0 grants every resumed run a fresh patience budget.
            self.epochs_without_improvement = int(
                info.get("epochs_without_improvement", 0))
        print(f"Resumed from epoch {self.current_epoch}", flush=True)

    # -- core loops -----------------------------------------------------------

    def _set_lr(self, epoch: int) -> float:
        lr = lr_at_epoch(epoch, self.peak_lr, self.warmup_epochs,
                         self.total_epochs)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, batches = 0.0, 0
        for batch_index, batch in enumerate(self.train_loader):
            if TorchVINTrainer._shutdown_requested:
                self.interrupted = True
                break
            images = batch["image"].to(self.device)
            logits = self.model(images)
            loss = ctc_loss_cpu_bridge(
                logits, batch["label"], batch["length"], batch["valid_width"])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            total_loss += float(loss.detach())
            batches += 1
            self.global_step += 1
            if batch_index % 10 == 0:
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch [{epoch}] Batch [{batch_index}/"
                      f"{len(self.train_loader)}] Loss: "
                      f"{float(loss.detach()):.4f} LR: {lr_now:.6f}",
                      flush=True)
        return total_loss / max(1, batches)

    def validate(self) -> Tuple[float, float, float]:
        """Returns (val_loss, exact_match, char_accuracy) - canonical metrics."""
        from src.vin_ocr.training.finetune_paddleocr import ctc_input_lengths

        self.model.eval()
        total_loss, batches = 0.0, 0
        pairs: List[Tuple[str, str]] = []
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                logits = self.model(images)
                loss = ctc_loss_cpu_bridge(
                    logits, batch["label"], batch["length"],
                    batch["valid_width"])
                total_loss += float(loss)
                batches += 1
                ids = logits.argmax(-1).cpu().numpy()
                # Decode only the timesteps training supervises: the SAME
                # per-sample input lengths the CTC loss uses mask the pad
                # region, so padding can never emit characters here while
                # being unsupervised in the loss (repo decode contract).
                input_lengths = ctc_input_lengths(
                    valid_widths=batch["valid_width"],
                    total_timesteps=int(logits.shape[1]),
                    image_width=INPUT_WIDTH,
                    label_lengths=[int(n) for n in batch["length"]],
                )
                for row, n_valid, ground_truth in zip(
                        ids, input_lengths, batch["text"], strict=True):
                    text, _ = ctc_greedy_decode(row[:n_valid], self.idx_to_char)
                    pairs.append((text[:17], ground_truth))
        metrics = char_level_metrics(pairs)
        exact = sum(1 for p, g in pairs if p == g) / max(1, len(pairs))
        return total_loss / max(1, batches), exact, metrics.char_accuracy

    def train(self, resume_from: Optional[str] = None) -> Dict[str, float]:
        if resume_from:
            self.resume(resume_from)

        early_config = self.config["Global"]
        patience = int(early_config.get("early_stopping_patience", 25))
        min_delta = float(early_config.get("early_stopping_min_delta", 1e-4))
        min_epochs = int(early_config.get("early_stopping_min_epochs", 10))

        from src.vin_ocr.training.finetune_paddleocr import update_early_stopping

        start_time = time.time()
        train_loss = self.train_losses[-1] if self.train_losses else 0.0
        val_loss = (self.best_val_loss
                    if not math.isinf(self.best_val_loss) else 0.0)
        val_exact = self.best_accuracy
        val_char = self.best_val_char_accuracy

        for epoch in range(self.current_epoch + 1, self.total_epochs + 1):
            if TorchVINTrainer._shutdown_requested:
                self.interrupted = True
                break
            self.current_epoch = epoch
            lr_now = self._set_lr(epoch)
            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            if self.interrupted:
                break
            self.train_losses.append(train_loss)
            val_loss, val_exact, val_char = self.validate()

            improved, self.epochs_without_improvement = update_early_stopping(
                val_accuracy=val_exact,
                previous_best=self.best_accuracy,
                epochs_without_improvement=self.epochs_without_improvement,
                min_delta=min_delta,
                val_loss=val_loss,
                best_val_loss=(None if math.isinf(self.best_val_loss)
                               else self.best_val_loss),
                val_char_accuracy=val_char,
                best_val_char_accuracy=self.best_val_char_accuracy,
            )
            if val_exact > self.best_accuracy:
                self.best_accuracy = val_exact
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_val_loss", {
                    "selection_metric": "val_loss",
                    "selection_value": val_loss,
                })
            if val_char > self.best_val_char_accuracy:
                self.best_val_char_accuracy = val_char
                self.save_checkpoint("best_char_accuracy", {
                    "selection_metric": "val_char_accuracy",
                    "selection_value": val_char,
                })
            self.save_checkpoint("latest")

            print(f"Epoch [{epoch}/{self.total_epochs}] "
                  f"Train Loss: {train_loss:.4f} "
                  f"Val Loss: {val_loss:.4f} "
                  f"Val Exact: {val_exact:.4f} "
                  f"Val CharAcc: {val_char:.4f} "
                  f"LR: {lr_now:.6f} "
                  f"({time.time() - epoch_start:.0f}s)", flush=True)

            self._log_epoch_metrics(epoch, train_loss, val_loss,
                                    val_exact, val_char, lr_now)

            if (epoch >= min_epochs
                    and self.epochs_without_improvement >= patience):
                print(f"Early stopping at epoch {epoch}: "
                      f"{self.epochs_without_improvement} epochs without "
                      f"improvement (patience {patience})", flush=True)
                break

        if self.interrupted:
            self.save_checkpoint("latest")
            print("Training interrupted; latest checkpoint saved.", flush=True)
        total_seconds = time.time() - start_time
        print(f"Training finished in {total_seconds / 3600:.2f}h | "
              f"best val char accuracy {self.best_val_char_accuracy:.4f}",
              flush=True)
        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "final_val_exact": val_exact,
            "final_val_char_accuracy": val_char,
            "best_val_char_accuracy": self.best_val_char_accuracy,
            "seconds": total_seconds,
        }

    # -- tracking hook (assigned by _run_tracked; no-op untracked) -------------

    def _log_epoch_metrics(self, epoch: int, train_loss: float,
                           val_loss: float, val_exact: float,
                           val_char: float, lr_now: float) -> None:
        pass


# =============================================================================
# EVALUATION (comparison-grade, canonical)
# =============================================================================

def evaluate_torch_checkpoint(checkpoint_path: str, label_file: str,
                              data_dir: str = "finetune_data",
                              config_path: str = "configs/vin_rosetta_torch_config.yml",
                              postprocess: bool = False,
                              device: str = "cpu") -> Dict[str, float]:
    """
    Single-image evaluation of a torch checkpoint with canonical metrics.

    Mirrors `model_registry.evaluate_checkpoint`'s contract exactly
    (single-image basis; same metric keys) so comparison rows line up.
    """
    from src.vin_ocr.core.vin_utils import validate_vin
    from src.vin_ocr.training.finetune_paddleocr import ctc_input_lengths

    config = load_config(config_path)
    char_to_idx, idx_to_char = load_char_dict(
        config["Global"]["character_dict_path"])
    arch = config.get("Architecture", {})
    model = RosettaResNet34Torch(
        num_classes=num_classes(char_to_idx),
        pretrained_backbone=False,  # weights come from the checkpoint
        dropout=float(arch.get("Head", {}).get("dropout", 0.1)),
    )
    payload = torch.load(checkpoint_path, map_location="cpu",
                         weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    post = None
    if postprocess:
        from src.vin_ocr.pipeline.vin_pipeline import VINPostProcessor
        post = VINPostProcessor()

    dataset = TorchVINDataset(data_dir=data_dir, label_file=label_file,
                              char_dict=char_to_idx, is_training=False)
    pairs, checksum_ok = [], 0
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            logits = model(item["image"][None].to(device))
            # Same pad-timestep masking as training/validate (M19): only
            # the sample's valid timesteps may emit characters.
            n_valid = ctc_input_lengths(
                valid_widths=[item["valid_width"]],
                total_timesteps=int(logits.shape[1]),
                image_width=INPUT_WIDTH,
                label_lengths=[item["length"]],
            )[0]
            ids = logits.argmax(-1).cpu().numpy()[0]
            text, _ = ctc_greedy_decode(ids[:n_valid], idx_to_char)
            pred = (post.process(text)["vin"] or "") if post else text[:17]
            pairs.append((pred, item["text"]))
            checksum_ok += validate_vin(pred).checksum_valid

    n = len(pairs)
    metrics = char_level_metrics(pairs)
    exact = sum(1 for p, g in pairs if p == g)
    return {
        "exact_match": exact / n if n else 0.0,
        "char_accuracy": metrics.char_accuracy,
        "f1_micro": metrics.f1_micro,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "cer": metrics.cer,
        "checksum_valid_rate": checksum_ok / n if n else 0.0,
        "n": float(n),
    }


# =============================================================================
# TRACKED ENTRYPOINT - MLflow 3 LoggedModel workflow
# =============================================================================

def _parse_label_file(label_file: str) -> List[Tuple[str, str]]:
    """
    Parse (path, vin) pairs with the loaders' tab-OR-space contract.

    Mirrors `VINRecognitionDataset._load_samples` (finetune_paddleocr.py):
    tab-separated lines split on tab, otherwise on any whitespace (first
    field only). A tab-only parser here fed whole space-separated lines
    into the MLflow Dataset entity's `path` column and None into `vin`.
    """
    pairs: List[Tuple[str, str]] = []
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def _run_tracked(trainer: TorchVINTrainer, config: Dict[str, Any],
                 args: argparse.Namespace) -> None:
    """
    Train inside a provenance-tracked run; on completion, apply the MLflow 3
    LoggedModel pattern: log the network as a LoggedModel with its params,
    measure it with the canonical evaluator, link those metrics to
    model_id + a named Dataset entity, and register a version.
    """
    try:
        from src.vin_ocr.tracking import start_run
    except Exception as exc:
        print(f"TRACKING DISABLED ({exc}); this run will have no provenance "
              f"record", flush=True)
        trainer.train(resume_from=args.resume)
        return

    import mlflow
    import pandas as pd

    arch = config.get("Architecture", {})
    val_label_file = config["Eval"]["dataset"]["label_file_list"][0]
    run_params = {
        "framework": f"torch-{torch.__version__}",
        "device": str(trainer.device),
        "algorithm": "Rosetta-ResNet34-torch",
        "pretrained_backbone": str(arch.get("pretrained_backbone", True)),
        "config_path": str(args.config),
        "learning_rate": trainer.peak_lr,
        "warmup_epochs": trainer.warmup_epochs,
        "epochs": trainer.total_epochs,
        "ctc": "CPU bridge (aten::_ctc_loss unimplemented on MPS)",
    }
    dataset_roots = []
    for section in ("Train", "Eval"):
        data_dir = config.get(section, {}).get("dataset", {}).get("data_dir")
        if data_dir and Path(data_dir).is_dir():
            dataset_roots.append(Path(data_dir))

    with start_run(
        f"train/Rosetta-ResNet34-torch-"
        f"{'in1k' if arch.get('pretrained_backbone', True) else 'scratch'}",
        experiment="vin_finetune",
        dataset_roots=dataset_roots,
        params=run_params,
    ) as run:
        trainer._log_epoch_metrics = (
            lambda epoch, train_loss, val_loss, val_exact, val_char, lr_now:
            run.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "val_exact_match": val_exact, "val_char_accuracy": val_char,
                "learning_rate": lr_now,
            }, step=epoch)
        )
        final = trainer.train(resume_from=args.resume)

        if trainer.interrupted:
            run.set_tags({"interrupted": "true"})
            return
        run.log_metrics({k: v for k, v in final.items()})

        # MLflow 3 LoggedModel: model + params, metrics linked to
        # model_id + Dataset entity, then a registry version. The label
        # file is parsed with the loaders' tab-OR-space contract so the
        # `vin` column matches what training actually consumed.
        labels_df = pd.DataFrame(
            _parse_label_file(val_label_file), columns=["path", "vin"],
        ).astype(str)
        eval_dataset = mlflow.data.from_pandas(
            labels_df, name=Path(val_label_file).stem, targets="vin")
        mlflow.log_input(eval_dataset, context="evaluation")

        best_path = trainer.output_dir / "best_char_accuracy.pt"
        weights_path = (best_path if best_path.is_file()
                        else trainer.output_dir / "latest.pt")
        eval_model = RosettaResNet34Torch(
            num_classes=trainer.num_classes, pretrained_backbone=False,
            dropout=float(arch.get("Head", {}).get("dropout", 0.1)))
        eval_model.load_state_dict(
            torch.load(weights_path, map_location="cpu",
                       weights_only=False)["model"])
        eval_model.eval()

        model_info = mlflow.pytorch.log_model(
            eval_model,
            name="rosetta-resnet34-torch",
            # mlflow>=3 + torch>=2.13 default to the 'pt2' traced-graph
            # format, which REQUIRES an input example to trace forward()
            # (measured failure 2026-08-20: omitting it crashed the
            # post-training export and FAILED the run).
            input_example=np.zeros((1, 3, 48, 320), dtype=np.float32),
            params={
                "pretrained_backbone": str(arch.get("pretrained_backbone", True)),
                "num_classes": str(trainer.num_classes),
                "input": "3x48x320",
                "timesteps": str(OUTPUT_TIMESTEPS),
                "weights_file": weights_path.name,
            },
        )
        measured = evaluate_torch_checkpoint(
            str(weights_path), val_label_file,
            data_dir=config["Eval"]["dataset"]["data_dir"].rstrip("/"),
            config_path=str(args.config))
        mlflow.log_metrics(measured, model_id=model_info.model_id,
                           dataset=eval_dataset)
        registered = mlflow.register_model(
            model_info.model_uri, "vin-rosetta-resnet34-torch")
        client = mlflow.MlflowClient()
        for key, value in {
            "semantics": "batch-first (torch, batch-independent)",
            "framework": f"torch-{torch.__version__}",
            "device_trained_on": str(trainer.device),
            "weights_file": weights_path.name,
            "measurement_basis": "single-image (batch-1), canonical char metrics",
        }.items():
            client.set_model_version_tag(
                "vin-rosetta-resnet34-torch", registered.version, key, value)
        print(f"LoggedModel {model_info.model_id} registered as "
              f"vin-rosetta-resnet34-torch v{registered.version}; "
              f"val char accuracy {measured['char_accuracy']:.4f}",
              flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apple-GPU (MPS) VIN recognition training")
    parser.add_argument("--config", "-c",
                        default="configs/vin_rosetta_torch_config.yml")
    parser.add_argument("--resume", "-r", default=None,
                        help="Checkpoint .pt path to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["Global"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = TorchVINTrainer(config)
    print(f"Device: {trainer.device} | params: "
          f"{sum(p.numel() for p in trainer.model.parameters()):,}",
          flush=True)
    _run_tracked(trainer, config, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
