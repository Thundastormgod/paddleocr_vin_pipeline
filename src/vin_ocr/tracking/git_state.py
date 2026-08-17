"""
Git provenance capture - the code half of run reproducibility.

Records the exact source state a run executed against: the commit, the branch,
and a diff that reconstructs the working tree from that commit, including
untracked files.

Differences from the usual approach (and why)
---------------------------------------------
The common idiom for capturing untracked files is ``git add -N .`` followed by
``git diff HEAD``. This module deliberately does NOT do that. ``git add -N``
mutates the caller's index as a side effect of *logging*, in the middle of a
training run that may last hours. Instead, untracked files are diffed
individually with ``git diff --no-index -- /dev/null <path>``, which is
read-only. This has been verified to leave ``git status --porcelain`` byte
identical before and after capture, and the resulting diffs round-trip
exactly through ``git apply``.

Bounded capture
---------------
Untracked files are embedded only when they are small text files. Large or
binary untracked files are recorded by path, size and SHA-256 in
``untracked_skipped`` instead of being embedded. This keeps artifact size
bounded (Power of 10 rule 3) and avoids encoding corruption, while still
telling you precisely what was NOT captured - large binaries are DVC's job,
not the diff's.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import (
    GitCommandError,
    GitUnavailableError,
    NotAGitRepositoryError,
)

#: Untracked files larger than this are recorded by hash rather than embedded.
DEFAULT_MAX_UNTRACKED_BYTES: int = 256 * 1024

#: Bytes inspected when deciding whether an untracked file is binary.
_BINARY_SNIFF_BYTES: int = 8192


@dataclass(frozen=True)
class SkippedFile:
    """An untracked file recorded by digest instead of by content."""

    path: str
    size_bytes: int
    sha256: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GitState:
    """
    Immutable snapshot of a Git working tree at run start.

    Attributes:
        commit: Full 40-character commit SHA the run executed against.
        branch: Branch name, or "HEAD" when detached.
        is_dirty: True when the working tree differs from ``commit`` in any
            way, including the presence of untracked (non-ignored) files.
        subject: First line of the commit message.
        message: Full commit message.
        author: Commit author as "Name <email>".
        committed_at: Commit timestamp, ISO 8601 with offset.
        remote_url: URL of the ``origin`` remote, or None when absent.
        tracked_diff: ``git diff HEAD --binary`` output; "" when clean.
        untracked_diff: Synthesised diff creating each embedded untracked file.
        untracked_skipped: Untracked files recorded by digest only.
    """

    commit: str
    branch: str
    is_dirty: bool
    subject: str
    message: str
    author: str
    committed_at: str
    remote_url: Optional[str]
    tracked_diff: str
    untracked_diff: str
    untracked_skipped: Tuple[SkippedFile, ...] = field(default=())

    @property
    def short_commit(self) -> str:
        """Abbreviated commit SHA."""
        return self.commit[:12]

    @property
    def diff(self) -> str:
        """Combined diff reconstructing the working tree from ``commit``."""
        parts = [p for p in (self.tracked_diff, self.untracked_diff) if p]
        return "".join(parts)

    def to_params(self) -> Dict[str, str]:
        """
        Return short, flat key/value pairs suitable for MLflow run params.

        The diff itself is deliberately excluded - it is logged as an artifact.
        """
        return {
            "git_commit": self.commit,
            "git_branch": self.branch,
            "git_dirty": str(self.is_dirty).lower(),
            "git_remote": self.remote_url or "",
        }

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation, excluding diff text."""
        return {
            "commit": self.commit,
            "branch": self.branch,
            "is_dirty": self.is_dirty,
            "subject": self.subject,
            "author": self.author,
            "committed_at": self.committed_at,
            "remote_url": self.remote_url,
            "has_tracked_diff": bool(self.tracked_diff),
            "has_untracked_diff": bool(self.untracked_diff),
            "untracked_skipped": [s.to_dict() for s in self.untracked_skipped],
        }


def _git_binary() -> str:
    """
    Resolve the git executable.

    Returns:
        Absolute path to ``git``.

    Raises:
        GitUnavailableError: If git is not on PATH.
    """
    git = shutil.which("git")
    if git is None:
        raise GitUnavailableError(
            "git executable not found on PATH; provenance cannot be captured"
        )
    return git


def _run_git(
    args: Sequence[str],
    cwd: Path,
    *,
    allow_codes: Tuple[int, ...] = (0,),
) -> str:
    """
    Run a git command and return stdout.

    Args:
        args: Arguments after the executable name.
        cwd: Directory to run in.
        allow_codes: Exit statuses treated as success. ``git diff --no-index``
            legitimately exits 1 when the inputs differ, which is always the
            case here.

    Returns:
        Captured stdout.

    Raises:
        GitCommandError: If the exit status is not in ``allow_codes``.
        GitUnavailableError: If git is not on PATH.
    """
    proc = subprocess.run(  # noqa: S603 - fixed executable, list args, no shell
        [_git_binary(), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode not in allow_codes:
        raise GitCommandError(tuple(args), proc.returncode, proc.stderr)
    return proc.stdout


def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Locate the root of the Git working tree containing ``start``.

    Args:
        start: Directory or file to search from. Defaults to this file's
            location, which resolves the repository even when the process CWD
            is elsewhere.

    Returns:
        Absolute path to the working-tree root.

    Raises:
        NotAGitRepositoryError: If ``start`` is not inside a working tree.
        GitUnavailableError: If git is not on PATH.
    """
    origin = Path(start).resolve() if start is not None else Path(__file__).resolve()
    directory = origin if origin.is_dir() else origin.parent

    try:
        output = _run_git(["rev-parse", "--show-toplevel"], directory)
    except GitCommandError as exc:
        raise NotAGitRepositoryError(f"{directory} is not inside a Git repository") from exc

    root = output.strip()
    if not root:
        raise NotAGitRepositoryError(f"{directory} is not inside a Git repository")
    return Path(root).resolve()


def _is_binary(path: Path) -> bool:
    """Return True if the file contains a NUL byte in its opening bytes."""
    try:
        with open(path, "rb") as handle:
            return b"\x00" in handle.read(_BINARY_SNIFF_BYTES)
    except OSError:
        # Unreadable (broken symlink, permissions) - treat as non-embeddable.
        return True


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed so memory stays bounded."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _list_untracked(repo_root: Path) -> List[str]:
    """
    List untracked, non-ignored files relative to the repository root.

    ``--exclude-standard`` honours .gitignore, so credentials, checkpoints and
    datasets excluded there are never captured into a diff artifact.
    """
    output = _run_git(
        ["ls-files", "--others", "--exclude-standard", "-z"], repo_root
    )
    return [entry for entry in output.split("\0") if entry]


def _capture_untracked(
    repo_root: Path, max_bytes: int
) -> Tuple[str, Tuple[SkippedFile, ...]]:
    """
    Build a diff creating each small untracked text file.

    Args:
        repo_root: Working-tree root.
        max_bytes: Files larger than this are recorded by digest only.

    Returns:
        The combined diff text and the tuple of skipped files.
    """
    chunks: List[str] = []
    skipped: List[SkippedFile] = []

    for relative in _list_untracked(repo_root):
        absolute = repo_root / relative
        if not absolute.is_file():
            continue  # directory entry or dangling symlink

        size = absolute.stat().st_size
        if size > max_bytes:
            reason = f"larger than {max_bytes} bytes"
        elif _is_binary(absolute):
            reason = "binary content"
        else:
            # Exit status 1 means "inputs differ", which is always true here.
            chunks.append(
                _run_git(
                    ["diff", "--no-index", "--", "/dev/null", relative],
                    repo_root,
                    allow_codes=(0, 1),
                )
            )
            continue

        skipped.append(
            SkippedFile(
                path=relative,
                size_bytes=size,
                sha256=_sha256(absolute),
                reason=reason,
            )
        )

    return "".join(chunks), tuple(skipped)


def capture_git_state(
    repo_root: Optional[Path] = None,
    *,
    max_untracked_bytes: int = DEFAULT_MAX_UNTRACKED_BYTES,
) -> GitState:
    """
    Capture the full Git provenance of the current working tree.

    This is read-only: it does not stage, stash, checkout or otherwise alter
    the repository. Verified by asserting ``git status --porcelain`` is
    unchanged across the call (see tests).

    Args:
        repo_root: Working-tree root. Discovered from this file when omitted.
        max_untracked_bytes: Size ceiling for embedding an untracked file.

    Returns:
        A populated GitState.

    Raises:
        GitUnavailableError: If git is not on PATH.
        NotAGitRepositoryError: If no working tree can be found.
        GitCommandError: If any git invocation fails unexpectedly.

    Complexity:
        O(n) git invocations for n untracked files, plus one streamed read per
        file that is hashed rather than embedded.
    """
    root = find_repo_root(repo_root)

    commit = _run_git(["rev-parse", "HEAD"], root).strip()
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root).strip()
    subject = _run_git(["log", "-1", "--pretty=%s"], root).strip()
    message = _run_git(["log", "-1", "--pretty=%B"], root).rstrip("\n")
    author = _run_git(["log", "-1", "--pretty=%an <%ae>"], root).strip()
    committed_at = _run_git(["log", "-1", "--pretty=%cI"], root).strip()

    # A missing 'origin' remote is normal, not an error.
    remote = _run_git(
        ["config", "--get", "remote.origin.url"], root, allow_codes=(0, 1)
    ).strip()

    status = _run_git(["status", "--porcelain"], root)
    tracked_diff = _run_git(["diff", "HEAD", "--binary"], root)
    untracked_diff, skipped = _capture_untracked(root, max_untracked_bytes)

    return GitState(
        commit=commit,
        branch=branch,
        is_dirty=bool(status.strip()),
        subject=subject,
        message=message,
        author=author,
        committed_at=committed_at,
        remote_url=remote or None,
        tracked_diff=tracked_diff,
        untracked_diff=untracked_diff,
        untracked_skipped=skipped,
    )
