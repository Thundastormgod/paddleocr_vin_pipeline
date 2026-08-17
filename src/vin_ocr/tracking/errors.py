"""
Exception types for the tracking package.

Provenance capture fails LOUDLY by design. This package exists because this
codebase shipped metrics that could not be traced to the code that produced
them - a hardcoded 46.51% presented as a measurement, `# Simulated` training
results, and sample sizes extrapolated from n=20 to n=382. A tracking layer
that silently degrades to "no provenance available" recreates exactly that
condition, so every failure mode here raises rather than returning a sentinel.

The single escape hatch is explicit and visible: see
``vin_ocr.tracking.run.start_run(strict=...)``, which still tags the resulting
run ``provenance_complete=false`` so an unprovenanced run is never
indistinguishable from a provenanced one.
"""

from __future__ import annotations


class TrackingError(Exception):
    """Base class for every error raised by vin_ocr.tracking."""


class ProvenanceError(TrackingError):
    """Provenance could not be established for a run."""


class GitUnavailableError(ProvenanceError):
    """The ``git`` executable could not be found on PATH."""


class NotAGitRepositoryError(ProvenanceError):
    """The given path is not inside a Git working tree."""


class GitCommandError(ProvenanceError):
    """A git subprocess exited with an unexpected status."""

    def __init__(self, args: tuple, returncode: int, stderr: str) -> None:
        self.args_run = args
        self.returncode = returncode
        self.stderr = stderr.strip()
        super().__init__(
            f"git {' '.join(args)} exited {returncode}: {self.stderr or '<no stderr>'}"
        )


class TrackingBackendUnavailableError(TrackingError):
    """The configured tracking backend (MLflow) is not importable."""
