#!/usr/bin/env python3
"""
Model evaluation regression gate.

Evaluates a Paddle inference model on a labeled split and compares
exact-match rate and corpus CER against a committed baseline. This is
the ML half of the CI gate: code changes cannot silently degrade the
model contract, and model updates must beat (or match within tolerance)
the recorded baseline before promotion.

Usage:
    # Gate (CI): fail when the model regresses vs the baseline
    python scripts/model_eval_gate.py \
        --model-dir output/vin_rec_finetune_stage3/inference \
        --labels finetune_data/val_labels.txt \
        --data-root finetune_data \
        --baseline configs/model_baseline.json

    # Record a new baseline (run locally after a verified improvement;
    # commit the JSON in the same PR as the model/registry change)
    python scripts/model_eval_gate.py ... --write-baseline

Exit codes:
    0  metrics within tolerance of the baseline (or baseline written)
    1  regression: exact-match fell or CER rose beyond tolerance
    2  usage/environment error (missing model, labels, or baseline)

The label file format is `<relative-or-absolute path>\t<VIN>` per line,
matching finetune_data/*_labels.txt. Relative paths resolve against
--data-root.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_labels(labels_path: Path, data_root: Path) -> List[Tuple[Path, str]]:
    """Parse `<path>\t<label>` lines; skip and count malformed ones."""
    if not labels_path.is_file():
        print(f"ERROR: label file not found: {labels_path}", file=sys.stderr)
        raise SystemExit(2)
    samples: List[Tuple[Path, str]] = []
    malformed = 0
    missing = 0
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        path_str, sep, label = line.partition("\t")
        if not sep:
            path_str, _, label = line.partition(" ")
        path_str, label = path_str.strip(), label.strip()
        if not path_str or not label:
            malformed += 1
            continue
        img = Path(path_str)
        if not img.is_absolute():
            img = data_root / img
        if not img.is_file():
            missing += 1
            continue
        samples.append((img, label.upper()))
    if malformed:
        print(f"WARNING: skipped {malformed} malformed label line(s)")
    if missing:
        print(f"WARNING: skipped {missing} sample(s) with missing image files")
    if not samples:
        print("ERROR: no usable samples in the label file", file=sys.stderr)
        raise SystemExit(2)
    return samples


def evaluate(model_dir: Path, samples: List[Tuple[Path, str]]) -> Dict[str, float]:
    """Run the model over samples; return exact-match rate and corpus CER."""
    from rapidfuzz.distance import Levenshtein

    from src.vin_ocr.inference.paddle_inference import VINInference

    inference = VINInference(str(model_dir))
    exact = 0
    edit_sum = 0
    ref_len_sum = 0
    errors = 0
    for img, gt in samples:
        result = inference.recognize(str(img))
        if result.get("error"):
            errors += 1
        pred = (result.get("vin") or "").upper()
        if pred == gt:
            exact += 1
        edit_sum += Levenshtein.distance(pred, gt)
        ref_len_sum += len(gt)
    n = len(samples)
    if errors:
        print(f"WARNING: {errors}/{n} samples returned an inference error")
    return {
        "exact_match": exact / n,
        "cer": (edit_sum / ref_len_sum) if ref_len_sum else 0.0,
        "n_samples": n,
        "inference_errors": errors,
    }


def git_commit() -> Optional[str]:
    """Current commit SHA, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=REPO_ROOT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model-dir", required=True,
                        help="Paddle inference dir (inference.json/.pdmodel + .pdiparams)")
    parser.add_argument("--labels", required=True, help="TSV label file: path<TAB>VIN")
    parser.add_argument("--data-root", default="finetune_data",
                        help="Base directory for relative image paths")
    parser.add_argument("--baseline", default="configs/model_baseline.json",
                        help="Committed baseline JSON path")
    parser.add_argument("--write-baseline", action="store_true",
                        help="Record current metrics as the new baseline instead of gating")
    parser.add_argument("--tol-exact", type=float, default=0.005,
                        help="Allowed exact-match drop (fraction, default 0.005 = 0.5pp)")
    parser.add_argument("--tol-cer", type=float, default=0.005,
                        help="Allowed CER rise (absolute, default 0.005)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate at most N samples (deterministic head)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    required_any = ["inference.json", "inference.pdmodel"]
    if not model_dir.is_dir() or not any((model_dir / f).is_file() for f in required_any):
        found = sorted(p.name for p in model_dir.glob("*")) if model_dir.is_dir() else []
        print(
            f"ERROR: {model_dir} is not a Paddle inference dir "
            f"(need one of {required_any} + inference.pdiparams); found: {found}",
            file=sys.stderr,
        )
        return 2

    samples = load_labels(Path(args.labels), Path(args.data_root))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    metrics = evaluate(model_dir, samples)
    print(
        f"measured: exact_match={metrics['exact_match']:.4f} "
        f"cer={metrics['cer']:.4f} n={metrics['n_samples']}"
    )

    baseline_path = Path(args.baseline)
    if args.write_baseline:
        record = {
            **metrics,
            "model_dir": str(model_dir),
            "labels": args.labels,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": git_commit(),
            "tolerances": {"exact_match_drop": args.tol_exact, "cer_rise": args.tol_cer},
        }
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"baseline written: {baseline_path}")
        return 0

    if not baseline_path.is_file():
        print(
            f"ERROR: baseline not found at {baseline_path}. Generate one on a "
            f"verified model with --write-baseline and commit it.",
            file=sys.stderr,
        )
        return 2
    baseline = json.loads(baseline_path.read_text())
    if "exact_match" not in baseline or "cer" not in baseline:
        print(f"ERROR: baseline {baseline_path} lacks exact_match/cer keys", file=sys.stderr)
        return 2

    exact_floor = baseline["exact_match"] - args.tol_exact
    cer_ceiling = baseline["cer"] + args.tol_cer
    print(
        f"baseline: exact_match={baseline['exact_match']:.4f} "
        f"cer={baseline['cer']:.4f} "
        f"(gate: exact>={exact_floor:.4f}, cer<={cer_ceiling:.4f})"
    )

    failures = []
    if metrics["exact_match"] < exact_floor:
        failures.append(
            f"exact_match {metrics['exact_match']:.4f} < floor {exact_floor:.4f}"
        )
    if metrics["cer"] > cer_ceiling:
        failures.append(f"cer {metrics['cer']:.4f} > ceiling {cer_ceiling:.4f}")

    if failures:
        print("GATE FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("GATE PASS: model within tolerance of the committed baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
