"""
Experiment tracking with mandatory provenance.

Why this package exists
-----------------------
This repository has shipped numbers that could not be traced to the code that
produced them:

* ``validate_architectures.py`` emitted a 46.51% accuracy figure while
  importing only ``yaml``, ``json`` and ``pathlib`` - it loaded no model, no
  dataset and no checkpoint.
* ``zenml_vin_pipeline.run_training_experiment`` returned hardcoded metrics
  marked ``# Simulated`` without ever calling ``train()``, and downstream steps
  derived a "performance tier" and "recommendations" from them.
* The README reported exact-match accuracy as ``19/382`` when the underlying
  record showed ``sample_size_for_metrics = 20``.

Each was found by reading the code. None could have been caught by reading the
results, because a result was just a number in a JSON file with no link to
anything. Repo-wide, **zero** files captured a commit SHA alongside a metric.

This package makes that link structural. Every run opened through
:func:`start_run` carries the commit it ran against, a diff reconstructing the
working tree from that commit, the resolved dependency versions, DVC hashes and
fingerprints for the data it read, and the exact command to replay it. A number
that cannot be reproduced can at least be identified as such:
``provenance_complete`` and ``provenance_reproducible`` are recorded on every
run.

Layering
--------
Provenance capture (:mod:`git_state`, :mod:`environment`, :mod:`dataset`,
:mod:`provenance`) is pure stdlib plus PyYAML and has no tracking-backend
dependency. Only :mod:`run` and :mod:`reproduce` require MLflow. Answering
"what code and what data produced this number" is therefore never contingent
on an optional dependency being installed.

Quick start
-----------
::

    from pathlib import Path
    from src.vin_ocr.tracking import start_run

    with start_run("eval-ppocrv5", dataset_roots=[Path("finetune_data/val_images")]) as run:
        run.log_params({"provider": "paddleocr", "det_box_thresh": 0.3})
        run.log_metrics({"exact_match": 0.4186, "character_accuracy": 0.9015})

Runs land in a local SQLite store at ``<repo>/mlflow.db`` by default - no
server, no credentials, no network. View them with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db

Set ``MLFLOW_TRACKING_URI`` to use a remote store instead. Replay any run with::

    python -m src.vin_ocr.tracking.reproduce <run_id>
"""

from .dataset import (
    DataState,
    DatasetFingerprint,
    DvcPointer,
    capture_data_state,
    fingerprint_directory,
    read_dvc_pointers,
)
from .environment import (
    EnvironmentState,
    capture_environment,
    resolve_version,
)
from .errors import (
    GitCommandError,
    GitUnavailableError,
    NotAGitRepositoryError,
    ProvenanceError,
    TrackingBackendUnavailableError,
    TrackingError,
)
from .git_state import (
    GitState,
    SkippedFile,
    capture_git_state,
    find_repo_root,
)
from .provenance import (
    Provenance,
    capture_provenance,
    read_logbook,
)
from .run import (
    TrackedRun,
    reproduce_command,
    resolve_tracking_uri,
    start_run,
)

__all__ = [
    # Run API
    "start_run",
    "TrackedRun",
    "reproduce_command",
    "resolve_tracking_uri",
    # Provenance
    "Provenance",
    "capture_provenance",
    "read_logbook",
    # Git
    "GitState",
    "SkippedFile",
    "capture_git_state",
    "find_repo_root",
    # Environment
    "EnvironmentState",
    "capture_environment",
    "resolve_version",
    # Data
    "DataState",
    "DatasetFingerprint",
    "DvcPointer",
    "capture_data_state",
    "fingerprint_directory",
    "read_dvc_pointers",
    # Errors
    "TrackingError",
    "ProvenanceError",
    "GitUnavailableError",
    "NotAGitRepositoryError",
    "GitCommandError",
    "TrackingBackendUnavailableError",
]
