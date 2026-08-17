"""
The tracked-run context manager - where provenance meets the metric store.

Usage::

    from src.vin_ocr.tracking import start_run

    with start_run("finetune-ppocrv5", experiment="vin-finetune",
                   dataset_roots=[Path("finetune_data/train_images")]) as run:
        run.log_params({"learning_rate": 1e-4, "batch_size": 32})
        for epoch in range(epochs):
            run.log_metrics({"train_loss": loss, "val_accuracy": acc}, step=epoch)

Every run created this way carries, without the caller doing anything:

* the commit it ran against, the branch, and whether the tree was dirty;
* a diff that reconstructs the working tree from that commit;
* the resolved versions of paddlepaddle, paddleocr, numpy, opencv and friends;
* DVC content hashes and dataset fingerprints for the data it read;
* the LOGBOOK.md entry, as the run description;
* the exact command to replay it.

Backend
-------
MLflow, defaulting to a local SQLite store at ``<repo>/mlflow.db`` with
artifacts under ``<repo>/mlartifacts``. That requires no server, no credentials
and no network::

    mlflow ui --backend-store-uri sqlite:///mlflow.db

Setting ``MLFLOW_TRACKING_URI`` redirects to a remote store (for example a
DagsHub-hosted MLflow) without any code change.

SQLite rather than the more commonly shown ``./mlruns`` file store: MLflow 3
placed the filesystem tracking backend in maintenance mode, and it now raises
``MlflowException`` unless ``MLFLOW_ALLOW_FILE_STORE=true`` is set. A file-store
default would therefore ship broken on any current install.

Failure policy
--------------
Provenance capture failures raise by default. This package exists because
untraceable metrics shipped from this repository; a tracker that quietly
degrades to "no provenance" would reproduce that failure. ``strict=False`` (or
``VIN_TRACKING_STRICT=0``) downgrades the failure to a loud warning, but the
run is then tagged ``provenance_complete=false`` so it can never be mistaken
for a provenanced one.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from .errors import ProvenanceError, TrackingBackendUnavailableError
from .git_state import find_repo_root
from .provenance import Provenance, capture_provenance

logger = logging.getLogger(__name__)

#: Standard MLflow variable; when set it overrides the local default.
TRACKING_URI_ENV: str = "MLFLOW_TRACKING_URI"

#: Default experiment name, overridable per call or by environment.
EXPERIMENT_ENV: str = "VIN_MLFLOW_EXPERIMENT"
DEFAULT_EXPERIMENT: str = "vin-ocr"

#: Set to "0"/"false" to downgrade provenance failures to warnings.
STRICT_ENV: str = "VIN_TRACKING_STRICT"

#: Local backend store used when no tracking URI is configured.
#:
#: SQLite, not the ``./mlruns`` file store. MLflow 3 placed the filesystem
#: tracking backend in maintenance mode and now raises MlflowException unless
#: MLFLOW_ALLOW_FILE_STORE=true is set, so a file-store default would ship
#: broken on any current install. SQLite needs no server either, and supports
#: the full feature set including run search and the model registry.
LOCAL_DB_FILENAME: str = "mlflow.db"

#: Artifact root for the local backend.
#:
#: Set explicitly rather than left to MLflow's default, which resolves
#: "./mlartifacts" against the process CWD - so artifacts would land in a
#: different place depending on where training was launched from.
LOCAL_ARTIFACT_DIRNAME: str = "mlartifacts"

#: Conservative ceiling; MLflow's own limit has varied across versions.
MAX_PARAM_CHARS: int = 500

_ARTIFACT_TRACKED_DIFF = "git_info/tracked.diff"
_ARTIFACT_UNTRACKED_DIFF = "git_info/untracked.diff"
_ARTIFACT_COMMIT_MESSAGE = "git_info/commit_message.txt"
_ARTIFACT_PROVENANCE = "provenance.json"
_ARTIFACT_LOGBOOK = "logbook.md"


def _import_mlflow() -> ModuleType:
    """
    Import MLflow, converting the ImportError into an actionable message.

    Returns:
        The imported mlflow module.

    Raises:
        TrackingBackendUnavailableError: If MLflow is not installed.
    """
    try:
        import mlflow  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise TrackingBackendUnavailableError(
            "MLflow is required for experiment tracking but is not installed. "
            "Install it with:  pip install 'mlflow>=2.9'  "
            "(or  pip install -e '.[tracking]' )"
        ) from exc
    return mlflow


def resolve_tracking_uri(repo_root: Optional[Path] = None) -> str:
    """
    Determine where run metadata is stored.

    Args:
        repo_root: Repository root, used for the local default.

    Returns:
        ``MLFLOW_TRACKING_URI`` when set and non-empty, otherwise a
        ``sqlite:///`` URI pointing at ``<repo_root>/mlflow.db``.

    Note:
        Inspect the local store without a server using
        ``mlflow ui --backend-store-uri sqlite:///mlflow.db``.
    """
    configured = os.environ.get(TRACKING_URI_ENV, "").strip()
    if configured:
        return configured

    root = find_repo_root(repo_root)
    return f"sqlite:///{(root / LOCAL_DB_FILENAME).resolve()}"


def resolve_artifact_root(repo_root: Optional[Path] = None) -> str:
    """
    Determine where run artifacts (diffs, provenance records) are written.

    Args:
        repo_root: Repository root, used for the local default.

    Returns:
        A ``file:`` URI for ``<repo_root>/mlartifacts``, anchored to the
        repository rather than the process CWD.
    """
    root = find_repo_root(repo_root)
    artifact_root = (root / LOCAL_ARTIFACT_DIRNAME).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root.as_uri()


def _strict_default() -> bool:
    """Return the default strictness, honouring VIN_TRACKING_STRICT."""
    raw = os.environ.get(STRICT_ENV, "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _truncate(value: Any) -> str:
    """Coerce a parameter value to a bounded string."""
    text = str(value)
    if len(text) <= MAX_PARAM_CHARS:
        return text
    logger.warning("Truncating over-long parameter value (%d chars)", len(text))
    return text[: MAX_PARAM_CHARS - 3] + "..."


def reproduce_command(run_id: str, tracking_uri: Optional[str] = None) -> str:
    """
    Build the shell command that restores a run's code and data state.

    Args:
        run_id: MLflow run identifier.
        tracking_uri: Tracking URI to embed, when not the default.

    Returns:
        A runnable command string, stored as a run parameter so it is visible
        directly in the MLflow UI.
    """
    command = f"python -m src.vin_ocr.tracking.reproduce {run_id}"
    if tracking_uri:
        command += f" --tracking-uri {tracking_uri}"
    return command


@dataclass
class TrackedRun:
    """
    Handle to an active tracked run.

    Thin, explicit wrapper over the MLflow fluent API. It exists so callers
    never import mlflow directly, which keeps the backend swappable and keeps
    optional-dependency handling in one place.
    """

    run_id: str
    experiment_id: str
    tracking_uri: str
    provenance: Optional[Provenance]
    _mlflow: ModuleType

    @property
    def reproduce_command(self) -> str:
        """The command that replays this run's state."""
        return reproduce_command(self.run_id, self.tracking_uri)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters. Values are stringified and length-bounded."""
        if params:
            self._mlflow.log_params({k: _truncate(v) for k, v in params.items()})

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric value, optionally at a training step."""
        self._mlflow.log_metric(key, float(value), step=step)

    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        """Log several metric values at the same step."""
        if metrics:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items()}, step=step
            )

    def log_dict(self, payload: Any, artifact_path: str) -> None:
        """Log a JSON-serialisable object as an artifact."""
        self._mlflow.log_text(
            json.dumps(payload, indent=2, default=str), artifact_path
        )

    def log_text(self, text: str, artifact_path: str) -> None:
        """Log a text artifact without touching the local filesystem."""
        self._mlflow.log_text(text, artifact_path)

    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None) -> None:
        """Log an existing file from disk as an artifact."""
        self._mlflow.log_artifact(str(local_path), artifact_path)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        """Attach tags to the run."""
        for key, value in tags.items():
            self._mlflow.set_tag(key, _truncate(value))


def _log_provenance(mlflow: ModuleType, provenance: Provenance) -> None:
    """
    Attach a provenance record to the active run.

    Diffs are written with ``mlflow.log_text``, which streams straight to the
    artifact store. The alternative - a NamedTemporaryFile that must then be
    removed - leaks a file on every early return, and its random name forces
    the reader to glob for it. Fixed artifact paths let ``reproduce`` fetch by
    exact name.

    Args:
        mlflow: The imported mlflow module.
        provenance: Record to attach.
    """
    mlflow.log_params({k: _truncate(v) for k, v in provenance.to_params().items()})
    mlflow.set_tag("provenance_complete", "true")
    mlflow.set_tag("git_commit", provenance.git.commit)
    mlflow.set_tag("mlflow.source.git.commit", provenance.git.commit)

    mlflow.log_text(provenance.git.message, _ARTIFACT_COMMIT_MESSAGE)
    if provenance.git.tracked_diff:
        mlflow.log_text(provenance.git.tracked_diff, _ARTIFACT_TRACKED_DIFF)
    if provenance.git.untracked_diff:
        mlflow.log_text(provenance.git.untracked_diff, _ARTIFACT_UNTRACKED_DIFF)

    mlflow.log_text(
        json.dumps(provenance.to_dict(), indent=2, default=str), _ARTIFACT_PROVENANCE
    )

    if provenance.logbook:
        mlflow.log_text(provenance.logbook, _ARTIFACT_LOGBOOK)
        mlflow.set_tag("mlflow.note.content", provenance.logbook[:1000])

    if not provenance.is_reproducible:
        skipped = ", ".join(s.path for s in provenance.git.untracked_skipped)
        logger.warning(
            "Run is not byte-reproducible: %d untracked file(s) recorded by "
            "digest only (%s). Their contents cannot be recovered from this "
            "run; put them under DVC to make the run replayable.",
            len(provenance.git.untracked_skipped),
            skipped,
        )


def _capture_or_warn(
    repo_root: Optional[Path],
    dataset_roots: Sequence[Path],
    include_data_content: bool,
    strict: bool,
) -> Optional[Provenance]:
    """
    Capture provenance, honouring the strictness policy.

    Args:
        repo_root: Repository root, or None to discover it.
        dataset_roots: Data directories to fingerprint.
        include_data_content: Hash dataset contents as well as sizes.
        strict: Re-raise on failure when True.

    Returns:
        The record, or None when capture failed and strict is False.

    Raises:
        ProvenanceError: On capture failure when strict is True.
    """
    try:
        return capture_provenance(
            repo_root, dataset_roots, include_data_content=include_data_content
        )
    except ProvenanceError as exc:
        if strict:
            raise ProvenanceError(
                f"Could not establish provenance for this run: {exc}. "
                f"Metrics without provenance are what this tracking layer "
                f"exists to prevent. Set {STRICT_ENV}=0 to record the run "
                f"anyway - it will be tagged provenance_complete=false."
            ) from exc
        logger.warning(
            "PROVENANCE UNAVAILABLE (%s). This run's metrics cannot be traced "
            "to the code that produced them and will be tagged "
            "provenance_complete=false.",
            exc,
        )
        return None


def _ensure_experiment(
    mlflow: ModuleType,
    name: str,
    repo_root: Optional[Path],
    *,
    is_local_store: bool,
) -> str:
    """
    Select the experiment, creating it with an explicit artifact root if new.

    ``mlflow.set_experiment(name)`` would create the experiment implicitly, but
    with MLflow's default artifact location, which for a SQLite backend is
    "./mlartifacts" resolved against the process CWD. Runs launched from
    different directories would then scatter their diffs across the filesystem.
    The artifact root is therefore pinned to the repository - but only for the
    local store, since a remote tracking server owns its own artifact layout.

    Args:
        mlflow: The imported mlflow module.
        name: Experiment name.
        repo_root: Repository root for the local artifact root.
        is_local_store: False when MLFLOW_TRACKING_URI points elsewhere.

    Returns:
        The experiment ID.
    """
    from mlflow.exceptions import MlflowException  # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    client = MlflowClient()
    existing = client.get_experiment_by_name(name)

    if existing is None:
        location = resolve_artifact_root(repo_root) if is_local_store else None
        try:
            client.create_experiment(name, artifact_location=location)
        except MlflowException:
            # Another process (e.g. a parallel Optuna trial) won the race.
            # Re-reading is correct; swallowing a genuine failure is not, so
            # the lookup below raises if the experiment still does not exist.
            pass
        existing = client.get_experiment_by_name(name)
        if existing is None:  # pragma: no cover - backend failure
            raise TrackingBackendUnavailableError(
                f"Could not create or read MLflow experiment '{name}'"
            )

    mlflow.set_experiment(experiment_id=existing.experiment_id)
    return existing.experiment_id


@contextmanager
def start_run(
    run_name: str,
    *,
    experiment: Optional[str] = None,
    dataset_roots: Sequence[Path] = (),
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, Any]] = None,
    include_data_content: bool = False,
    strict: Optional[bool] = None,
    repo_root: Optional[Path] = None,
) -> Iterator[TrackedRun]:
    """
    Open a tracked run with full provenance attached.

    Provenance is captured BEFORE the run is created, so a capture failure
    under ``strict`` leaves no orphaned, half-populated run behind.

    Args:
        run_name: Human-readable name for the run.
        experiment: Experiment to log under. Falls back to
            ``VIN_MLFLOW_EXPERIMENT`` then ``DEFAULT_EXPERIMENT``.
        dataset_roots: Data directories the run reads, fingerprinted so the
            metrics can be tied to the data that produced them.
        params: Hyperparameters to log immediately.
        tags: Extra tags to attach.
        include_data_content: Hash dataset file contents as well as sizes.
        strict: Fail when provenance cannot be captured. Defaults to the
            ``VIN_TRACKING_STRICT`` environment variable, itself defaulting
            to True.
        repo_root: Repository root. Discovered from the package when omitted.

    Yields:
        A TrackedRun handle.

    Raises:
        TrackingBackendUnavailableError: If MLflow is not installed.
        ProvenanceError: If provenance capture fails and ``strict`` is True.
    """
    effective_strict = _strict_default() if strict is None else strict
    provenance = _capture_or_warn(
        repo_root, dataset_roots, include_data_content, effective_strict
    )

    mlflow = _import_mlflow()
    is_local_store = not os.environ.get(TRACKING_URI_ENV, "").strip()
    tracking_uri = resolve_tracking_uri(repo_root)
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = (
        experiment or os.environ.get(EXPERIMENT_ENV, "").strip() or DEFAULT_EXPERIMENT
    )
    _ensure_experiment(
        mlflow, experiment_name, repo_root, is_local_store=is_local_store
    )

    with mlflow.start_run(run_name=run_name) as active:
        run = TrackedRun(
            run_id=active.info.run_id,
            experiment_id=active.info.experiment_id,
            tracking_uri=tracking_uri,
            provenance=provenance,
            _mlflow=mlflow,
        )

        if provenance is not None:
            _log_provenance(mlflow, provenance)
        else:
            mlflow.set_tag("provenance_complete", "false")

        mlflow.log_param("reproduce_command", _truncate(run.reproduce_command))

        if params:
            run.log_params(params)
        if tags:
            run.set_tags(tags)

        logger.info(
            "Tracked run '%s' (%s) in experiment '%s' at %s",
            run_name,
            run.run_id,
            experiment_name,
            tracking_uri,
        )
        yield run
