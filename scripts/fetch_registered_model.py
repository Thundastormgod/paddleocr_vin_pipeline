#!/usr/bin/env python3
"""
Fetch the latest registered model version from the MLflow registry and
locate its servable Paddle inference directory.

Companion to scripts/model_eval_gate.py: the model-gate workflow calls
this first to materialize the gated artifact, then evaluates it.

Requires MLFLOW_TRACKING_URI (and, for DagsHub, MLFLOW_TRACKING_USERNAME
/ MLFLOW_TRACKING_PASSWORD) in the environment.

Prints the resolved inference directory path on the LAST line of stdout
so shell callers can capture it:  MODEL_DIR=$(... | tail -1)

Exit codes:
    0  inference dir resolved and printed
    2  configuration/registry/layout error (message on stderr)
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def fail(msg: str) -> "SystemExit":
    print(f"ERROR: {msg}", file=sys.stderr)
    return SystemExit(2)


def main() -> int:
    try:
        from src.vin_ocr.tracking.model_registry import REGISTERED_MODEL_NAME
        default_name = REGISTERED_MODEL_NAME
    except ImportError:  # minimal env without the tracking extra: documented default
        default_name = "vin-lcnetv3-svtr-ctc"

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--name", default=default_name,
                        help=f"Registered model name (default: {default_name})")
    parser.add_argument("--version", default=None,
                        help="Specific version number; default is the latest")
    parser.add_argument("--dest", default="gate-model",
                        help="Directory to download artifacts into")
    args = parser.parse_args()

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise fail(
            "MLFLOW_TRACKING_URI is not set. For DagsHub use "
            "https://dagshub.com/<owner>/<repo>.mlflow plus "
            "MLFLOW_TRACKING_USERNAME/MLFLOW_TRACKING_PASSWORD."
        )

    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()

    if args.version is not None:
        try:
            version = client.get_model_version(args.name, args.version)
        except MlflowException as e:
            raise fail(f"model {args.name!r} version {args.version} not found: {e}")
    else:
        versions = client.search_model_versions(f"name='{args.name}'")
        if not versions:
            raise fail(
                f"registry has no versions of {args.name!r}. Register one with "
                f"scripts/promote_model.py before enabling the model gate."
            )
        version = max(versions, key=lambda v: int(v.version))

    print(f"registry: {args.name} v{version.version} (run {version.run_id})")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=version.source, dst_path=str(dest)
    )

    # Locate a servable Paddle inference dir anywhere in the artifacts.
    root = Path(local)
    candidates = [
        p.parent for p in root.rglob("*")
        if p.name in ("inference.json", "inference.pdmodel")
        and (p.parent / "inference.pdiparams").is_file()
    ]
    if not candidates:
        layout = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        raise fail(
            f"downloaded artifacts of {args.name} v{version.version} contain no "
            f"servable inference dir (inference.json/.pdmodel + .pdiparams). "
            f"Files found: {layout[:50]}"
        )

    print(str(candidates[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
