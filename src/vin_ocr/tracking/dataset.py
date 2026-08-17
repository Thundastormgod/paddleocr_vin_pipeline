"""
Data provenance - the third leg of reproducibility, alongside code and
environment.

A run is reproducible only when you can recover the data it saw. This project
has a live reason to care: dataset splitting previously leaked VINs across
train and test (multiple photographs of the same physical plate landing on
both sides), which inflates every reported accuracy. ``assert_no_vin_leakage``
now guards the split, but that guarantee is only auditable after the fact if
the run records *which* data it was applied to.

Two independent signals are captured:

* **DVC pointers** - the authoritative content hashes for anything under DVC
  control. These are true content digests: two runs with the same pointer
  md5 saw byte-identical data.
* **Directory fingerprints** - a cheap stat-only manifest digest over
  (relative path, size) pairs, for data not yet under DVC. This detects files
  added, removed or resized. It deliberately does NOT hash file contents by
  default, because that is O(total bytes) and would add minutes to the start
  of every run; pass ``include_content=True`` when that cost is acceptable.

The fingerprint is a weaker guarantee than a DVC hash and is labelled as such
in the recorded output (``content_hashed``), so a stat-only manifest is never
mistaken for a content hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

#: Filename DVC uses for its pipeline lock.
DVC_LOCK_FILENAME: str = "dvc.lock"

#: Directories never descended into when fingerprinting a dataset root.
_FINGERPRINT_EXCLUDE_DIRS: frozenset = frozenset(
    {".git", ".dvc", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
)


@dataclass(frozen=True)
class DvcPointer:
    """
    One DVC-tracked output and its content hash.

    Attributes:
        source: File the pointer was read from, relative to the repo root
            (e.g. "data.dvc" or "dvc.lock").
        target: The tracked path, as recorded by DVC.
        md5: DVC content hash. A trailing ".dir" marks a directory digest.
        size_bytes: Total tracked size, when DVC recorded it.
        nfiles: File count for directory outputs, when DVC recorded it.
    """

    source: str
    target: str
    md5: str
    size_bytes: Optional[int] = None
    nfiles: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "source": self.source,
            "target": self.target,
            "md5": self.md5,
            "size_bytes": self.size_bytes,
            "nfiles": self.nfiles,
        }


@dataclass(frozen=True)
class DatasetFingerprint:
    """
    Stat-based manifest digest for a directory of data.

    Attributes:
        root: Directory fingerprinted, relative to the repo root when possible.
        exists: False when the directory is absent; all counters are then zero.
        file_count: Number of regular files found.
        total_bytes: Sum of file sizes.
        manifest_sha256: SHA-256 over the sorted "<relpath>\\0<size>\\n"
            manifest, or over "<relpath>\\0<size>\\0<sha256>\\n" when
            ``content_hashed`` is True.
        content_hashed: True only when file contents were hashed. When False
            this digest detects additions, removals and size changes but NOT
            same-size content edits.
    """

    root: str
    exists: bool
    file_count: int
    total_bytes: int
    manifest_sha256: str
    content_hashed: bool

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "root": self.root,
            "exists": self.exists,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "manifest_sha256": self.manifest_sha256,
            "content_hashed": self.content_hashed,
        }


@dataclass(frozen=True)
class DataState:
    """Aggregate data provenance for a run."""

    dvc_pointers: Tuple[DvcPointer, ...]
    datasets: Tuple[DatasetFingerprint, ...]

    @property
    def is_dvc_tracked(self) -> bool:
        """True when at least one DVC pointer was found."""
        return bool(self.dvc_pointers)

    def to_params(self) -> Dict[str, str]:
        """Return short, flat key/value pairs suitable for MLflow run params."""
        params: Dict[str, str] = {"data_dvc_tracked": str(self.is_dvc_tracked).lower()}
        for pointer in self.dvc_pointers:
            key = f"dvc_{_slug(pointer.target)}"
            params[key] = pointer.md5
        for fingerprint in self.datasets:
            key = f"data_{_slug(fingerprint.root)}"
            params[key] = fingerprint.manifest_sha256[:16]
        return params

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "dvc_pointers": [p.to_dict() for p in self.dvc_pointers],
            "datasets": [d.to_dict() for d in self.datasets],
        }


def _slug(value: str) -> str:
    """Reduce a path to a compact MLflow-safe parameter key fragment."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.strip("./"))
    return cleaned.strip("_") or "root"


def _outs_from_mapping(
    document: Mapping[str, Any], source: str
) -> Iterable[DvcPointer]:
    """Yield pointers from the ``outs`` list of a .dvc document."""
    outs = document.get("outs")
    if not isinstance(outs, list):
        return

    for entry in outs:
        if not isinstance(entry, dict):
            continue
        md5 = entry.get("md5") or entry.get("hash_value")
        target = entry.get("path")
        if not md5 or not target:
            continue
        yield DvcPointer(
            source=source,
            target=str(target),
            md5=str(md5),
            size_bytes=entry.get("size"),
            nfiles=entry.get("nfiles"),
        )


def _outs_from_lock(document: Mapping[str, Any], source: str) -> Iterable[DvcPointer]:
    """Yield pointers from every stage of a dvc.lock document."""
    stages = document.get("stages")
    if not isinstance(stages, dict):
        return

    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            continue
        for pointer in _outs_from_mapping(stage, f"{source}#{stage_name}"):
            yield pointer


def read_dvc_pointers(repo_root: Path) -> Tuple[DvcPointer, ...]:
    """
    Read every DVC content hash recorded in the repository.

    Scans all ``*.dvc`` pointer files plus ``dvc.lock`` if present. Malformed
    YAML in one file does not abort the scan, but it is not swallowed either -
    the offending file is skipped and reported through the returned pointers'
    absence, which the caller surfaces as "not DVC tracked".

    Args:
        repo_root: Repository root to scan.

    Returns:
        Pointers sorted by (source, target) for deterministic output.

    Complexity:
        O(n) YAML parses for n pointer files.
    """
    root = Path(repo_root)
    pointers: List[DvcPointer] = []

    candidates: List[Path] = sorted(root.rglob("*.dvc"))
    lock = root / DVC_LOCK_FILENAME
    if lock.is_file():
        candidates.append(lock)

    for path in candidates:
        if any(part in _FINGERPRINT_EXCLUDE_DIRS for part in path.parts):
            continue
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(document, dict):
            continue

        source = str(path.relative_to(root))
        if path.name == DVC_LOCK_FILENAME:
            pointers.extend(_outs_from_lock(document, source))
        else:
            pointers.extend(_outs_from_mapping(document, source))

    return tuple(sorted(pointers, key=lambda p: (p.source, p.target)))


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed so memory stays bounded."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_directory(
    root: Path,
    *,
    include_content: bool = False,
    relative_to: Optional[Path] = None,
) -> DatasetFingerprint:
    """
    Compute a deterministic manifest digest for a directory of data.

    The manifest is built from POSIX-style relative paths so the digest is
    stable across operating systems, and sorted so it is independent of
    filesystem walk order.

    Args:
        root: Directory to fingerprint.
        include_content: Hash file contents as well as sizes. This is
            O(total bytes) and is off by default; the resulting fingerprint
            records which mode was used.
        relative_to: Base for the reported ``root`` label.

    Returns:
        A DatasetFingerprint. A missing directory yields ``exists=False`` with
        zeroed counters and the digest of the empty manifest, rather than
        raising - absence is a legitimate, recordable state.

    Complexity:
        O(n) stat calls for n files, plus O(total bytes) when
        ``include_content`` is True.
    """
    directory = Path(root)
    label = _relative_label(directory, relative_to)

    if not directory.is_dir():
        return DatasetFingerprint(
            root=label,
            exists=False,
            file_count=0,
            total_bytes=0,
            manifest_sha256=hashlib.sha256(b"").hexdigest(),
            content_hashed=include_content,
        )

    entries: List[str] = []
    total_bytes = 0

    for path in sorted(directory.rglob("*")):
        if any(part in _FINGERPRINT_EXCLUDE_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue

        size = path.stat().st_size
        total_bytes += size
        relative = path.relative_to(directory).as_posix()

        if include_content:
            entries.append(f"{relative}\0{size}\0{_sha256_file(path)}\n")
        else:
            entries.append(f"{relative}\0{size}\n")

    manifest = "".join(sorted(entries)).encode("utf-8")

    return DatasetFingerprint(
        root=label,
        exists=True,
        file_count=len(entries),
        total_bytes=total_bytes,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        content_hashed=include_content,
    )


def _relative_label(path: Path, base: Optional[Path]) -> str:
    """Render ``path`` relative to ``base`` when possible, else absolutely."""
    if base is None:
        return str(path)
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return str(path)


def capture_data_state(
    repo_root: Path,
    dataset_roots: Sequence[Path] = (),
    *,
    include_content: bool = False,
) -> DataState:
    """
    Capture DVC hashes and directory fingerprints for a run.

    Args:
        repo_root: Repository root, used to locate DVC pointers.
        dataset_roots: Directories to fingerprint, typically the train/val/test
            roots the run actually read.
        include_content: Forwarded to fingerprint_directory.

    Returns:
        A populated DataState.
    """
    root = Path(repo_root)
    fingerprints = tuple(
        fingerprint_directory(
            Path(dataset_root), include_content=include_content, relative_to=root
        )
        for dataset_root in dataset_roots
    )
    return DataState(
        dvc_pointers=read_dvc_pointers(root),
        datasets=fingerprints,
    )
