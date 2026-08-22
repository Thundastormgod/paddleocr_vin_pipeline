#!/usr/bin/env python3
"""
Promote a trained checkpoint into the MLflow model registry.

Thin, argument-checked runner around
src.vin_ocr.tracking.model_registry.register_checkpoint_version so the
model-promote workflow (and humans) have one documented entry point.
Registration evaluates the checkpoint on the given label file and
attaches the metrics to the registered version.

Requires MLFLOW_TRACKING_URI (and DagsHub credentials where applicable)
in the environment.

Exit codes:
    0  version registered; name/version/metrics printed
    2  configuration error (missing checkpoint/labels/tracking URI)
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint path (with or without .pdparams)")
    parser.add_argument("--labels", default="finetune_data/val_labels.txt",
                        help="Evaluation label file used during registration")
    parser.add_argument("--stage-label", default="",
                        help="Free-form stage tag recorded on the version")
    parser.add_argument("--config", default="configs/vin_finetune_config.yml")
    parser.add_argument("--dict", dest="dict_path", default="configs/vin_dict.txt")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print(
            "ERROR: MLFLOW_TRACKING_URI is not set; refusing to register "
            "against an implicit local store.",
            file=sys.stderr,
        )
        return 2

    ckpt = Path(args.checkpoint)
    ckpt_params = ckpt if ckpt.suffix == ".pdparams" else ckpt.with_suffix(".pdparams")
    if not ckpt_params.is_file():
        print(f"ERROR: checkpoint not found: {ckpt_params}", file=sys.stderr)
        return 2
    if not Path(args.labels).is_file():
        print(f"ERROR: label file not found: {args.labels}", file=sys.stderr)
        return 2

    from src.vin_ocr.tracking.model_registry import register_checkpoint_version

    result = register_checkpoint_version(
        checkpoint_path=str(ckpt),
        label_file=args.labels,
        stage_label=args.stage_label,
        config_path=args.config,
        dict_path=args.dict_path,
        max_eval_samples=args.max_eval_samples,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
