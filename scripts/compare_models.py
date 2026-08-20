#!/usr/bin/env python3
"""
Measure every available VIN model on the same splits with the same metrics
and log one MLflow run per model into the `model_comparison` experiment.

Purpose: side-by-side comparison in the MLflow UI (select runs -> Compare).
Every number is measured IN the run that reports it - nothing is copied
from other runs or documents (empiricism invariant). Uniform metric keys
across all runs make the UI comparison table line up:

    {split}_char_accuracy, {split}_exact_match, {split}_f1_micro,
    {split}_precision, {split}_recall, {split}_cer,
    {split}_checksum_valid_rate, {split}_n, {split}_errors, {split}_seconds

Model basis is recorded per run (params: family, semantics_basis,
postprocess, weights). From-scratch checkpoints are evaluated under
legacy_batch_axis_attention=True - the semantics they were trained and
historically measured under; under the fixed batch-first forward the same
weights are a different (garbage) function, measured at 0.1113 val char
accuracy. See SVTREncoder's docstring and LOGBOOK 2026-08-20.

Usage:
    python scripts/compare_models.py                  # all available models
    python scripts/compare_models.py --only stock     # substring filter

Runtime: ~6 minutes on CPU for the full table (142 images x 7 variants).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SPLITS = {
    "val": "finetune_data/val_labels.txt",
    "test": "finetune_data/test_labels.txt",
}


def _read_epoch(checkpoint: str) -> Optional[int]:
    """Epoch recorded in the checkpoint's provenance sidecar, if present."""
    info = Path(checkpoint).with_name(Path(checkpoint).stem + "_info.json")
    if not info.is_file():
        return None
    try:
        return int(json.loads(info.read_text()).get("epoch"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def eval_checkpoint_on_split(checkpoint: str, label_file: str,
                             postprocess: bool, legacy: bool,
                             config_path: str) -> Dict[str, float]:
    """Canonical single-image evaluation (see evaluate_checkpoint)."""
    from src.vin_ocr.tracking.model_registry import evaluate_checkpoint

    started = time.time()
    measured = evaluate_checkpoint(
        checkpoint, label_file,
        config_path=config_path,
        legacy_batch_axis_attention=legacy,
        postprocess=postprocess,
    )
    measured["errors"] = 0.0  # decode path is total: every image scores
    measured["seconds"] = round(time.time() - started, 1)
    return measured


def eval_stock_pipeline_on_split(label_file: str,
                                 postprocess: bool) -> Dict[str, float]:
    """Stock PaddleOCR engine through VINOCRPipeline, canonical metrics."""
    from src.vin_ocr.core.char_metrics import char_level_metrics
    from src.vin_ocr.core.vin_utils import validate_vin
    from src.vin_ocr.pipeline.vin_pipeline import VINOCRPipeline

    pipeline = VINOCRPipeline(
        preprocess_mode="engraved",
        enable_postprocess=postprocess,
        verbose=False,
    )
    pairs: List[Tuple[str, str]] = []
    errors = 0
    started = time.time()
    for line in Path(label_file).read_text().splitlines():
        rel_path, gt = line.split("\t")
        image_path = str(Path("finetune_data") / rel_path)
        try:
            result = pipeline.recognize(image_path)
        except Exception:
            errors += 1
            continue
        if result.get("error"):
            errors += 1
            continue
        pairs.append(((result.get("vin") or "")[:17], gt))
    seconds = round(time.time() - started, 1)

    n = len(pairs)
    metrics = char_level_metrics(pairs)
    exact = sum(1 for p, g in pairs if p == g)
    checksum_ok = sum(validate_vin(p).checksum_valid for p, _ in pairs)
    return {
        "exact_match": exact / n if n else 0.0,
        "char_accuracy": metrics.char_accuracy,
        "f1_micro": metrics.f1_micro,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "cer": metrics.cer,
        "checksum_valid_rate": checksum_ok / n if n else 0.0,
        "n": float(n),
        "errors": float(errors),
        "seconds": seconds,
    }


class ModelSpec:
    """One row of the comparison: how to evaluate it and its recorded basis."""

    def __init__(self, name: str, family: str, semantics: str,
                 evaluator: Callable[[str], Dict[str, float]],
                 params: Dict[str, Any],
                 available: Callable[[], Optional[str]]):
        self.name = name
        self.family = family
        self.semantics = semantics
        self.evaluator = evaluator
        self.params = params
        self.available = available  # returns skip reason or None


def _ckpt_spec(name: str, checkpoint: str, postprocess: bool,
               family: str = "LCNetV3-SVTR-CTC 3.23M (from-scratch, refuted route)",
               legacy: bool = True,
               config_path: str = "configs/vin_finetune_config.yml") -> ModelSpec:
    """Checkpoint row. `legacy`/`config_path` MUST match the checkpoint's
    training basis: LCNetV3-SVTR-CTC checkpoints predate the batch-axis fix
    (legacy=True, finetune config); Rosetta postdates it (legacy=False,
    rosetta config selects the architecture via its algorithm key)."""
    def available() -> Optional[str]:
        if not Path(checkpoint).is_file():
            return f"checkpoint missing: {checkpoint}"
        return None

    return ModelSpec(
        name=name,
        family=family,
        semantics=("legacy-batch-axis-attention" if legacy
                   else "batch-first (post-fix, batch-independent)"),
        evaluator=lambda label_file: eval_checkpoint_on_split(
            checkpoint, label_file, postprocess, legacy, config_path),
        params={
            "weights": checkpoint,
            "epoch": _read_epoch(checkpoint),
            "postprocess": postprocess,
            "config": config_path,
            "input": "pre-cropped plate, VINRecognitionDataset preprocessing",
        },
        available=available,
    )


def _torch_ckpt_spec(name: str, checkpoint: str, postprocess: bool,
                     config_path: str = "configs/vin_rosetta_torch_config.yml") -> ModelSpec:
    """torch/MPS-trained checkpoint row: canonical metrics via
    evaluate_torch_checkpoint (same keys, same single-image basis)."""
    def available() -> Optional[str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return "torch not installed"
        if not Path(checkpoint).is_file():
            return f"checkpoint missing: {checkpoint}"
        return None

    def evaluator(label_file: str) -> Dict[str, float]:
        from src.vin_ocr.training.finetune_torch import evaluate_torch_checkpoint
        started = time.time()
        measured = evaluate_torch_checkpoint(
            checkpoint, label_file, config_path=config_path,
            postprocess=postprocess)
        measured["errors"] = 0.0
        measured["seconds"] = round(time.time() - started, 1)
        return measured

    return ModelSpec(
        name=name,
        family="Rosetta-ResNet34-torch 21.3M (ImageNet-1k warm start, MPS-trained)",
        semantics="batch-first (torch, batch-independent)",
        evaluator=evaluator,
        params={
            "weights": checkpoint,
            "epoch": None,
            "postprocess": postprocess,
            "config": config_path,
            "input": "pre-cropped plate, VINRecognitionDataset preprocessing",
        },
        available=available,
    )


def _stock_spec(name: str, postprocess: bool) -> ModelSpec:
    def available() -> Optional[str]:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return "paddleocr not installed"
        return None

    return ModelSpec(
        name=name,
        family="PP-OCRv3-mobile det+rec (stock pretrained, production default)",
        semantics="batch-independent by engine design",
        evaluator=lambda label_file: eval_stock_pipeline_on_split(
            label_file, postprocess),
        params={
            "weights": "PP-OCRv3_mobile_det + en_PP-OCRv3_mobile_rec (official zoo)",
            "preprocess_mode": "engraved",
            "postprocess": postprocess,
            "input": "pre-cropped plate through full det+rec pipeline",
        },
        available=available,
    )


def build_specs() -> List[ModelSpec]:
    return [
        # Real model names. Stock = official PaddleOCR zoo identifiers;
        # custom family = its actual composition (LCNetV3-SVTR-CTC, 3.23M
        # params), versioned by training epoch. Stage labels live in params.
        _stock_spec("PP-OCRv3-mobile+pipeline", postprocess=True),
        _stock_spec("PP-OCRv3-mobile+pipeline-nopostproc", postprocess=False),
        _ckpt_spec("LCNetV3-SVTR-CTC-ep25",
                   "output/vin_rec_finetune/latest.pdparams", postprocess=False),
        _ckpt_spec("LCNetV3-SVTR-CTC-ep36-warmrestart",
                   "output/vin_rec_finetune_stage2/latest.pdparams", postprocess=False),
        _ckpt_spec("LCNetV3-SVTR-CTC-ep19",
                   "output/vin_rec_finetune_stage3/best_val_loss.pdparams",
                   postprocess=False),
        _ckpt_spec("LCNetV3-SVTR-CTC-ep44",
                   "output/vin_rec_finetune_stage3/latest.pdparams", postprocess=False),
        _ckpt_spec("LCNetV3-SVTR-CTC-ep44+postproc",
                   "output/vin_rec_finetune_stage3/latest.pdparams", postprocess=True),
        _ckpt_spec("Rosetta-ResNet34vd",
                   "output/vin_rosetta/latest.pdparams", postprocess=False,
                   family="Rosetta-ResNet34vd 21.3M (from-scratch)",
                   legacy=False,
                   config_path="configs/vin_rosetta_config.yml"),
        _ckpt_spec("Rosetta-ResNet34vd+postproc",
                   "output/vin_rosetta/latest.pdparams", postprocess=True,
                   family="Rosetta-ResNet34vd 21.3M (from-scratch)",
                   legacy=False,
                   config_path="configs/vin_rosetta_config.yml"),
        _torch_ckpt_spec("Rosetta-ResNet34-IN1K",
                         "output/vin_rosetta_torch_in1k/best_char_accuracy.pt",
                         postprocess=False),
        _torch_ckpt_spec("Rosetta-ResNet34-IN1K+postproc",
                         "output/vin_rosetta_torch_in1k/best_char_accuracy.pt",
                         postprocess=True),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="",
                        help="Substring filter on model names")
    parser.add_argument("--experiment", default="model_comparison")
    args = parser.parse_args()

    from src.vin_ocr.tracking import start_run

    specs = [s for s in build_specs() if args.only in s.name]
    if not specs:
        print(f"no models match filter {args.only!r}")
        return 1

    for label_file in SPLITS.values():
        if not Path(label_file).is_file():
            print(f"label file missing: {label_file} - stage the dataset first")
            return 1

    results: Dict[str, Dict[str, float]] = {}
    for spec in specs:
        reason = spec.available()
        if reason:
            print(f"SKIP {spec.name}: {reason}")
            continue
        print(f"== {spec.name} ==")
        with start_run(
            f"eval/{spec.name}",
            experiment=args.experiment,
            dataset_roots=[Path("finetune_data")],
            params={
                "model": spec.name,
                "family": spec.family,
                "semantics_basis": spec.semantics,
                **{k: str(v) for k, v in spec.params.items()},
            },
        ) as run:
            flat: Dict[str, float] = {}
            for split, label_file in SPLITS.items():
                measured = spec.evaluator(label_file)
                flat.update({f"{split}_{k}": v for k, v in measured.items()})
                print(f"  {split}: char={measured['char_accuracy']:.4f} "
                      f"exact={measured['exact_match']:.4f} "
                      f"f1={measured['f1_micro']:.4f} n={int(measured['n'])} "
                      f"errors={int(measured['errors'])} "
                      f"({measured['seconds']}s)")
            run.log_metrics(flat)
            results[spec.name] = flat

    if results:
        print("\n=== COMPARISON (sorted by test_char_accuracy) ===")
        header = f"{'model':38s} {'val_char':>8s} {'val_exact':>9s} {'test_char':>9s} {'test_exact':>10s}"
        print(header)
        print("-" * len(header))
        ranked = sorted(results.items(),
                        key=lambda kv: kv[1]["test_char_accuracy"], reverse=True)
        for name, m in ranked:
            print(f"{name:38s} {m['val_char_accuracy']:8.4f} "
                  f"{m['val_exact_match']:9.4f} {m['test_char_accuracy']:9.4f} "
                  f"{m['test_exact_match']:10.4f}")
        print("\nView: mlflow UI -> experiment "
              f"'{args.experiment}' -> select runs -> Compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
