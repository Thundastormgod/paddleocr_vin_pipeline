"""
Provenance aggregation - code, environment and data captured as one record.

This module has no dependency on any tracking backend. It is pure stdlib plus
PyYAML, so provenance capture is testable, and correct, whether or not MLflow
is installed. Only ``vin_ocr.tracking.run`` needs a backend, and it consumes
the record produced here.

That separation is deliberate: the ability to answer "what code and what data
produced this number" must not be contingent on an optional dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

from .dataset import DataState, capture_data_state
from .environment import EnvironmentState, capture_environment
from .git_state import GitState, capture_git_state, find_repo_root

logger = logging.getLogger(__name__)

#: Narrative log attached to every run, read from the repository root.
LOGBOOK_FILENAME: str = "LOGBOOK.md"

#: Truncation ceiling for the logbook when attached as a run description.
MAX_LOGBOOK_CHARS: int = 20_000


def read_logbook(repo_root: Optional[Path] = None) -> Optional[str]:
    """
    Read LOGBOOK.md from the repository root.

    Resolved from the repository root rather than the process CWD. The obvious
    implementation, ``open("LOGBOOK.md")``, silently reads a different file (or
    raises) depending on where the process was launched from - and an
    unhandled FileNotFoundError would abort a training run over a logging
    concern.

    A missing logbook is a legitimate state, not an error: it returns None.
    An unreadable logbook is logged as a warning and also returns None, since
    the narrative is commentary, not the provenance record itself.

    Args:
        repo_root: Repository root. Discovered from the package when omitted.

    Returns:
        Logbook contents, truncated to MAX_LOGBOOK_CHARS, or None.
    """
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    path = root / LOGBOOK_FILENAME

    if not path.is_file():
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None

    if len(text) > MAX_LOGBOOK_CHARS:
        return text[:MAX_LOGBOOK_CHARS] + "\n\n[truncated]\n"
    return text


@dataclass(frozen=True)
class Provenance:
    """
    Everything needed to reconstruct and audit a run.

    Attributes:
        git: Source state, including a diff that reconstructs the working tree.
        environment: Interpreter, platform and resolved dependency versions.
        data: DVC hashes and dataset fingerprints.
        logbook: Narrative entry, or None when LOGBOOK.md is absent.
        captured_at: UTC timestamp, ISO 8601, of capture - a real clock read,
            not a placeholder. (A fabrication-era summary artifact previously
            carried a hardcoded timestamp that made a re-run byte-identical
            and therefore indistinguishable from a measurement; removed
            2026-08-20.)
    """

    git: GitState
    environment: EnvironmentState
    data: DataState
    logbook: Optional[str]
    captured_at: str

    @property
    def is_reproducible(self) -> bool:
        """
        True when this run can be replayed exactly.

        A dirty tree is still fully reproducible here, because the diff is
        captured. What breaks replay is an untracked file that was too large
        or too binary to embed - it is recorded by digest, but its contents
        are not recoverable from the run alone.
        """
        return not self.git.untracked_skipped

    def to_params(self) -> Dict[str, str]:
        """Return the flat parameter set logged with the run."""
        params: Dict[str, str] = {}
        params.update(self.git.to_params())
        params.update(self.environment.to_params())
        params.update(self.data.to_params())
        params["provenance_reproducible"] = str(self.is_reproducible).lower()
        return params

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the whole record."""
        return {
            "captured_at": self.captured_at,
            "is_reproducible": self.is_reproducible,
            "git": self.git.to_dict(),
            "environment": self.environment.to_dict(),
            "data": self.data.to_dict(),
            "has_logbook": self.logbook is not None,
        }


def capture_provenance(
    repo_root: Optional[Path] = None,
    dataset_roots: Sequence[Path] = (),
    *,
    include_data_content: bool = False,
) -> Provenance:
    """
    Capture the complete provenance record for a run.

    Args:
        repo_root: Repository root. Discovered from the package when omitted.
        dataset_roots: Data directories the run will read, fingerprinted so
            metrics can be tied to the data that produced them.
        include_data_content: Hash dataset file contents as well as sizes.
            O(total bytes); off by default.

    Returns:
        A populated Provenance record.

    Raises:
        GitUnavailableError: If git is not on PATH.
        NotAGitRepositoryError: If no Git working tree can be found.
        GitCommandError: If a git invocation fails unexpectedly.
    """
    root = find_repo_root(repo_root)

    return Provenance(
        git=capture_git_state(root),
        environment=capture_environment(),
        data=capture_data_state(
            root, dataset_roots, include_content=include_data_content
        ),
        logbook=read_logbook(root),
        captured_at=datetime.now(timezone.utc).isoformat(),
    )
