"""
Replay the code and data state of a tracked run.

Given a run ID, this restores the working tree to the commit that run executed
against and re-applies the diff captured at run start, then pulls the matching
data with DVC. The point is falsifiability: any number this project reports can
be re-derived by a third party, rather than taken on trust.

Usage::

    python -m src.vin_ocr.tracking.reproduce <run_id>
    python -m src.vin_ocr.tracking.reproduce <run_id> --tracking-uri http://localhost:5000
    python -m src.vin_ocr.tracking.reproduce <run_id> --dry-run

Safety
------
``git restore --worktree`` overwrites files unconditionally, so the command
refuses to run against a dirty working tree. It never switches branch and
never rewrites history; only the working tree and the DVC cache are touched.
Use ``--dry-run`` to see exactly what would happen first.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .errors import TrackingBackendUnavailableError
from .git_state import find_repo_root
from .run import (
    _ARTIFACT_TRACKED_DIFF,
    _ARTIFACT_UNTRACKED_DIFF,
    _import_mlflow,
    resolve_tracking_uri,
)

logger = logging.getLogger(__name__)

_DIFF_ARTIFACTS: Tuple[str, ...] = (_ARTIFACT_TRACKED_DIFF, _ARTIFACT_UNTRACKED_DIFF)


class ReproduceError(RuntimeError):
    """A precondition for replaying a run was not met."""


def _require(executable: str) -> str:
    """
    Resolve an executable, raising with an actionable message if absent.

    Args:
        executable: Program name to resolve.

    Returns:
        Absolute path to the executable.

    Raises:
        ReproduceError: If it is not on PATH.
    """
    resolved = shutil.which(executable)
    if resolved is None:
        raise ReproduceError(f"'{executable}' is not installed or not on PATH")
    return resolved


def _run(args: Sequence[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess with a fixed executable list and no shell."""
    proc = subprocess.run(  # noqa: S603 - list args, resolved executable, no shell
        list(args),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and proc.returncode != 0:
        raise ReproduceError(
            f"{' '.join(args)} exited {proc.returncode}: "
            f"{proc.stderr.strip() or '<no stderr>'}"
        )
    return proc


def ensure_clean_worktree(repo_root: Path) -> None:
    """
    Refuse to proceed when the working tree has local changes.

    Args:
        repo_root: Repository to check.

    Raises:
        ReproduceError: If tracked files are modified or untracked files exist.
            Untracked files matter as much as modified ones here: restoring to
            a commit leaves them in place, and re-applying a diff that creates
            the same path then fails.
    """
    git = _require("git")
    status = _run([git, "status", "--porcelain"], repo_root)
    if status.stdout.strip():
        raise ReproduceError(
            "Working tree has local changes. Commit or stash them first:\n"
            + status.stdout.rstrip()
        )


def verify_commit_exists(repo_root: Path, commit: str) -> None:
    """
    Confirm the recorded commit is present locally.

    Args:
        repo_root: Repository to check.
        commit: Commit SHA recorded with the run.

    Raises:
        ReproduceError: If the object is missing, with a fetch hint.
    """
    git = _require("git")
    proc = _run(
        [git, "cat-file", "-e", f"{commit}^{{commit}}"], repo_root, check=False
    )
    if proc.returncode != 0:
        raise ReproduceError(
            f"Commit {commit} is not present locally. Run 'git fetch --all' "
            f"and try again."
        )


def fetch_commit(run) -> str:
    """
    Extract the commit SHA recorded with a run.

    Args:
        run: An ``mlflow.entities.Run``.

    Returns:
        The commit SHA.

    Raises:
        ReproduceError: If the run carries no commit, which means it was
            created outside this tracking layer or with provenance disabled.
    """
    commit = (
        run.data.tags.get("git_commit")
        or run.data.tags.get("mlflow.source.git.commit")
        or run.data.params.get("git_commit")
    )
    if not commit:
        raise ReproduceError(
            f"Run {run.info.run_id} records no git commit "
            f"(provenance_complete="
            f"{run.data.tags.get('provenance_complete', 'unknown')}). "
            f"It cannot be replayed."
        )
    return str(commit)


def download_diffs(run_id: str, destination: Path) -> List[Path]:
    """
    Download whichever diff artifacts the run actually recorded.

    A clean run records no diffs at all, which is a valid state rather than an
    error, so the artifact listing is consulted before downloading instead of
    catching a download failure and guessing what it meant.

    Args:
        run_id: Run to fetch from.
        destination: Local directory to download into.

    Returns:
        Local paths of the downloaded diffs, in application order.
    """
    mlflow = _import_mlflow()
    from mlflow.tracking import MlflowClient  # noqa: PLC0415 - optional dependency

    client = MlflowClient()
    available = {item.path for item in client.list_artifacts(run_id, "git_info")}

    downloaded: List[Path] = []
    for artifact in _DIFF_ARTIFACTS:
        if artifact not in available:
            continue
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact, dst_path=str(destination)
        )
        downloaded.append(Path(local))
    return downloaded


def restore_to_commit(repo_root: Path, commit: str) -> None:
    """Restore every tracked file to its state at ``commit``, keeping HEAD."""
    git = _require("git")
    _run([git, "restore", "--source", commit, "--worktree", ":/"], repo_root)


def apply_diff(repo_root: Path, diff_path: Path) -> None:
    """Apply a captured diff onto the working tree."""
    git = _require("git")
    _run([git, "apply", "--whitespace=nowarn", str(diff_path)], repo_root)


def dvc_pull(repo_root: Path) -> bool:
    """
    Pull DVC-tracked data for the restored state.

    Returns:
        True when data was pulled, False when DVC is unavailable or has
        nothing to pull. Failure is reported, never silently ignored.
    """
    dvc = shutil.which("dvc")
    if dvc is None:
        logger.warning("dvc is not on PATH; DVC-tracked data was NOT restored")
        return False
    if not (repo_root / ".dvc").is_dir():
        logger.info("No .dvc directory; nothing to pull")
        return False

    proc = _run([dvc, "pull"], repo_root, check=False)
    if proc.returncode != 0:
        logger.warning("dvc pull failed: %s", proc.stderr.strip())
        return False
    logger.info("dvc pull completed")
    return True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.vin_ocr.tracking.reproduce",
        description="Restore the code and data state of a tracked run.",
    )
    parser.add_argument("run_id", help="MLflow run ID to replay.")
    parser.add_argument(
        "--tracking-uri",
        help="Tracking URI. Defaults to MLFLOW_TRACKING_URI or the local store.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Repository root. Discovered automatically when omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be restored without modifying anything.",
    )
    parser.add_argument(
        "--skip-dvc",
        action="store_true",
        help="Do not run 'dvc pull' after restoring the code state.",
    )
    return parser.parse_args(argv)


def _report(run, commit: str, repo_root: Path) -> None:
    """Print a human-readable summary of the run about to be replayed."""
    tags = run.data.tags
    print(f"run id      : {run.info.run_id}")
    print(f"run name    : {tags.get('mlflow.runName', '<unnamed>')}")
    print(f"commit      : {commit}")
    print(f"branch      : {run.data.params.get('git_branch', '<unknown>')}")
    print(f"dirty       : {run.data.params.get('git_dirty', '<unknown>')}")
    print(f"provenance  : {tags.get('provenance_complete', '<unknown>')}")
    print(f"reproducible: {run.data.params.get('provenance_reproducible', '<unknown>')}")
    print(f"repository  : {repo_root}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point.

    Returns:
        Process exit status: 0 on success, 1 on a handled precondition failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    try:
        repo_root = find_repo_root(Path(args.repo) if args.repo else None)
        mlflow = _import_mlflow()
        mlflow.set_tracking_uri(args.tracking_uri or resolve_tracking_uri(repo_root))

        run = mlflow.get_run(args.run_id)
        commit = fetch_commit(run)
        _report(run, commit, repo_root)

        if args.dry_run:
            print("\n--dry-run: nothing was modified.")
            return 0

        ensure_clean_worktree(repo_root)
        verify_commit_exists(repo_root, commit)

        with tempfile.TemporaryDirectory(prefix="vin_reproduce_") as tmp:
            diffs = download_diffs(args.run_id, Path(tmp))
            logger.info("Restoring working tree to %s", commit)
            restore_to_commit(repo_root, commit)
            for diff in diffs:
                logger.info("Applying %s", diff.name)
                apply_diff(repo_root, diff)
            if not diffs:
                logger.info("Run recorded no diff; the tree was clean at run time")

        if not args.skip_dvc:
            dvc_pull(repo_root)

        print("\nWorking tree now reflects the run's state.")
        return 0

    except (ReproduceError, TrackingBackendUnavailableError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
