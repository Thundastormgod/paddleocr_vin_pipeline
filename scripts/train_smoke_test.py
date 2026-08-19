#!/usr/bin/env python3
"""
Training smoke test: prove the trainer can RUN and LEARN on this install.

Rationale (C8): CI historically installed a paddle-less test subset, so
the class of failure where the trainer cannot execute at all - e.g. the
paddle 3.3 warpctc dtype contract that crashed every run on first batch,
found only when a human finally executed training - was invisible to CI.
This script is the executable guard: it generates a synthetic
micro-dataset of rendered VIN plates (valid ISO-3779 check digits), runs
a short real training, and ASSERTS the mechanics.

Asserts:
  1. training completes (no crash - dtype contracts, geometry, loaders);
  2. loss FELL materially from the cold-start value;
  3. per-epoch `latest` checkpoint + resume info exist;
  4. best-by-val-loss checkpoint exists with provenance;
  5. resume-from-latest starts at the recorded epoch.

Runtime: ~2-3 minutes on CPU. Usage:
    python scripts/train_smoke_test.py [--workdir DIR]
Exit code 0 on success; any assertion failure is a real regression.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.vin_ocr.core.vin_utils import calculate_check_digit  # noqa: E402

CHARS = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"


def make_vin(rng: random.Random) -> str:
    body = [rng.choice(CHARS) for _ in range(17)]
    body[8] = "0"
    check = calculate_check_digit("".join(body))
    body[8] = check
    return "".join(body)


def render_plate(vin: str, path: Path, seed: int) -> None:
    rstate = np.random.RandomState(seed)
    img = np.full((96, 640, 3), 120, np.uint8)
    noise = rstate.randint(-25, 25, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.putText(img, vin, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.15,
                (30, 30, 30), 3, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def build_dataset(base: Path, n_train: int = 12, n_val: int = 4) -> None:
    (base / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    vins: list = []
    while len(vins) < n_train + n_val:
        vin = make_vin(rng)
        if vin not in vins:
            vins.append(vin)
    train_lines, val_lines = [], []
    for i, vin in enumerate(vins):
        name = f"{vin}_{i}.jpg"
        render_plate(vin, base / "images" / name, seed=i)
        (train_lines if i < n_train else val_lines).append(f"images/{name}\t{vin}")
    (base / "train_labels.txt").write_text("\n".join(train_lines) + "\n")
    (base / "val_labels.txt").write_text("\n".join(val_lines) + "\n")


def write_config(base: Path) -> Path:
    template = (REPO / "configs" / "vin_finetune_config.yml").read_text()
    config = template
    config = config.replace("epoch_num: 30", "epoch_num: 3")
    config = config.replace("T_max: 30", "T_max: 3")
    config = config.replace("warmup_epoch: 5", "warmup_epoch: 1")
    config = config.replace("batch_size_per_card: 16", "batch_size_per_card: 4")
    config = config.replace("save_model_dir: ./output/vin_rec_finetune",
                            f"save_model_dir: {base / 'output'}")
    config = config.replace("data_dir: ./finetune_data/", f"data_dir: {base}/")
    config = config.replace("- ./finetune_data/train_labels.txt",
                            f"- {base / 'train_labels.txt'}")
    config = config.replace("- ./finetune_data/val_labels.txt",
                            f"- {base / 'val_labels.txt'}")
    path = base / "smoke_config.yml"
    path.write_text(config)
    return path


def run_training(config_path: Path, base: Path, resume: str = "") -> str:
    import os
    env = dict(os.environ)
    # Isolate provenance: smoke runs must never pollute the real MLflow store.
    env["MLFLOW_TRACKING_URI"] = f"sqlite:///{base / 'smoke_mlflow.db'}"
    cmd = [sys.executable, "-u", "-m", "src.vin_ocr.training.finetune_paddleocr",
           "--config", str(config_path)]
    if resume:
        cmd += ["--resume", resume]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(REPO), timeout=1200, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        raise AssertionError(f"training exited {proc.returncode}")
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=None,
                        help="Reuse a directory instead of a fresh tempdir")
    args = parser.parse_args()

    base = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix="vin_smoke_"))
    print(f"workdir: {base}")

    build_dataset(base)
    config_path = write_config(base)

    stdout = run_training(config_path, base)

    losses = [float(m) for m in re.findall(
        r"Train Loss: ([0-9.]+)", stdout)]
    assert losses, "no epoch loss lines in trainer output"
    assert losses[0] > 5.0, f"cold-start loss suspiciously low: {losses[0]}"
    assert min(losses) < losses[0] * 0.8, (
        f"loss did not fall materially: {losses}"
    )
    print(f"loss trajectory OK: {losses}")

    out = base / "output"
    for artifact in ("latest.pdparams", "latest.pdopt", "latest_info.json",
                     "best_val_loss.pdparams", "best_val_loss_info.json"):
        assert (out / artifact).is_file(), f"missing checkpoint artifact: {artifact}"
    info = json.loads((out / "latest_info.json").read_text())
    assert info["epoch"] == 3, f"latest_info epoch {info['epoch']} != 3"
    print("checkpoint artifacts OK")

    # Resume with exactly one epoch of budget left: proves optimizer/epoch
    # restore and that training continues rather than restarts.
    resume_config = base / "smoke_config_resume.yml"
    resume_config.write_text(
        config_path.read_text()
        .replace("epoch_num: 3", "epoch_num: 4")
        .replace("T_max: 3", "T_max: 4")
    )
    resume_out = run_training(resume_config, base, resume=str(out / "latest"))
    assert "Resumed from epoch 3" in resume_out, "resume did not restore epoch"
    info = json.loads((out / "latest_info.json").read_text())
    assert info["epoch"] == 4, f"post-resume epoch {info['epoch']} != 4"
    print("resume-from-latest OK")

    # Resume with ZERO epochs of budget left: pins the empty-loop completion
    # path (historically a NameError on train_loss - the completion handler
    # read variables only assigned inside the epoch loop).
    noop_out = run_training(resume_config, base, resume=str(out / "latest"))
    assert "Resumed from epoch 4" in noop_out, "no-op resume did not restore epoch"
    assert "Training Complete" in noop_out, (
        "zero-remaining-epochs resume did not reach clean completion"
    )
    print("zero-epoch resume OK")

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
