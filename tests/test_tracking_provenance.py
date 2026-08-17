"""
Tests for provenance capture.

These cover the layer that answers "what code and what data produced this
number". They deliberately depend on nothing but the standard library, PyYAML
and git, so provenance stays verifiable in the CI environment, which installs
neither MLflow nor PaddlePaddle.

The two load-bearing tests are:

* :meth:`TestCaptureIsReadOnly.test_capture_does_not_touch_the_index` - the
  usual way to capture untracked files is ``git add -N .``, which mutates the
  caller's index as a side effect of logging, mid-training-run. This asserts
  that does not happen.
* :meth:`TestReproduceRoundTrip.test_restore_then_apply_reconstructs_the_tree` -
  provenance that cannot actually be replayed is decoration. This performs the
  full restore-and-apply cycle and compares file digests.
"""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Sequence

import pytest
import yaml

from src.vin_ocr.tracking import (
    capture_environment,
    capture_git_state,
    capture_provenance,
    fingerprint_directory,
    find_repo_root,
    read_dvc_pointers,
    read_logbook,
    reproduce_command,
    resolve_tracking_uri,
    resolve_version,
)
from src.vin_ocr.tracking.run import resolve_artifact_root
from src.vin_ocr.tracking.errors import (
    GitUnavailableError,
    NotAGitRepositoryError,
)
from src.vin_ocr.tracking.git_state import DEFAULT_MAX_UNTRACKED_BYTES

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is required for provenance tests"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` and return stdout, failing loudly on error."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _write(path: Path, content: str) -> None:
    """Write text, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _tree_digests(root: Path) -> Dict[str, str]:
    """Map every non-.git file to the SHA-256 of its contents."""
    digests: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts or not path.is_file():
            continue
        digests[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return digests


def _status(repo: Path) -> str:
    """Return the porcelain status, the canonical 'is anything different'."""
    return _git(repo, "status", "--porcelain")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A committed git repository with a clean working tree."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "commit.gpgsign", "false")

    _write(root / "module.py", "VALUE = 1\n")
    _write(root / "pkg" / "inner.py", "INNER = 'a'\n")
    _write(root / ".gitignore", "*.log\nsecrets/\n")

    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "initial commit")
    return root


@pytest.fixture()
def dirty_repo(repo: Path) -> Path:
    """The same repository with tracked edits, new files and ignored files."""
    _write(repo / "module.py", "VALUE = 2\n")            # modified tracked
    _write(repo / "added.py", "NEW = True\n")            # untracked
    _write(repo / "pkg" / "nested_new.py", "N = 0\n")    # untracked, nested
    _write(repo / "debug.log", "ignored by gitignore\n")  # ignored
    _write(repo / "secrets" / "token.txt", "SHOULD-NEVER-BE-CAPTURED\n")
    return repo


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


class TestCaptureIsReadOnly:
    """
    Capture must not mutate the repository.

    The widely-copied implementation of this idea (including the MLOps recipe
    this package is modelled on) runs ``git add -N .`` so that untracked files
    appear in ``git diff HEAD``. That stages intent-to-add entries into the
    user's index as a side effect of *logging*, during a run that may last
    hours. This package diffs untracked files individually against /dev/null
    instead, which touches nothing.
    """

    def test_capture_does_not_touch_the_index(self, dirty_repo: Path):
        before = _status(dirty_repo)
        capture_git_state(dirty_repo)
        assert _status(dirty_repo) == before

    def test_capture_does_not_change_head(self, dirty_repo: Path):
        before = _git(dirty_repo, "rev-parse", "HEAD")
        capture_git_state(dirty_repo)
        assert _git(dirty_repo, "rev-parse", "HEAD") == before

    def test_capture_does_not_modify_any_file(self, dirty_repo: Path):
        before = _tree_digests(dirty_repo)
        capture_git_state(dirty_repo)
        assert _tree_digests(dirty_repo) == before


# ---------------------------------------------------------------------------
# What gets captured
# ---------------------------------------------------------------------------


class TestGitStateContent:
    """The captured record must describe the tree accurately."""

    def test_clean_tree_is_not_dirty_and_has_no_diff(self, repo: Path):
        state = capture_git_state(repo)
        assert state.is_dirty is False
        assert state.tracked_diff == ""
        assert state.untracked_diff == ""
        assert state.diff == ""

    def test_records_commit_branch_and_subject(self, repo: Path):
        state = capture_git_state(repo)
        assert state.commit == _git(repo, "rev-parse", "HEAD").strip()
        assert len(state.commit) == 40
        assert state.short_commit == state.commit[:12]
        assert state.branch == "main"
        assert state.subject == "initial commit"

    def test_untracked_files_make_the_tree_dirty(self, repo: Path):
        _write(repo / "brand_new.py", "x = 1\n")
        state = capture_git_state(repo)
        assert state.is_dirty is True, (
            "an untracked file changes what the code does and must count as dirty"
        )

    def test_captures_modified_and_untracked_separately(self, dirty_repo: Path):
        state = capture_git_state(dirty_repo)
        assert "module.py" in state.tracked_diff
        assert "added.py" in state.untracked_diff
        assert "pkg/nested_new.py" in state.untracked_diff

    def test_gitignored_files_are_never_captured(self, dirty_repo: Path):
        """
        .gitignore is this repository's credential guard. A provenance layer
        that embedded ignored files into a diff artifact would exfiltrate the
        very secrets that guard exists to keep out of git.
        """
        state = capture_git_state(dirty_repo)
        combined = state.diff
        assert "SHOULD-NEVER-BE-CAPTURED" not in combined
        assert "debug.log" not in combined
        assert not any("secrets/" in s.path for s in state.untracked_skipped)

    def test_missing_origin_remote_is_not_an_error(self, repo: Path):
        state = capture_git_state(repo)
        assert state.remote_url is None

    def test_remote_url_is_recorded_when_present(self, repo: Path):
        _git(repo, "remote", "add", "origin", "https://example.invalid/x.git")
        assert capture_git_state(repo).remote_url == "https://example.invalid/x.git"


class TestBoundedUntrackedCapture:
    """
    Embedding is bounded, and what is skipped is stated explicitly.

    Silence about a skipped file would be the worst outcome: the run would
    look reproducible while missing part of its input.
    """

    def test_large_untracked_file_is_hashed_not_embedded(self, repo: Path):
        payload = "x" * 2048
        _write(repo / "big.txt", payload)

        state = capture_git_state(repo, max_untracked_bytes=1024)

        assert "big.txt" not in state.untracked_diff
        assert len(state.untracked_skipped) == 1
        skipped = state.untracked_skipped[0]
        assert skipped.path == "big.txt"
        assert skipped.size_bytes == len(payload)
        assert skipped.sha256 == hashlib.sha256(payload.encode()).hexdigest()
        assert "larger than" in skipped.reason

    def test_binary_untracked_file_is_hashed_not_embedded(self, repo: Path):
        (repo / "blob.bin").write_bytes(b"\x00\x01\x02binary\x00content")
        state = capture_git_state(repo)
        assert [s.path for s in state.untracked_skipped] == ["blob.bin"]
        assert state.untracked_skipped[0].reason == "binary content"

    def test_small_text_file_is_embedded(self, repo: Path):
        _write(repo / "small.txt", "just a line\n")
        state = capture_git_state(repo)
        assert state.untracked_skipped == ()
        assert "just a line" in state.untracked_diff

    def test_default_ceiling_is_bounded(self):
        assert 0 < DEFAULT_MAX_UNTRACKED_BYTES <= 1024 * 1024


# ---------------------------------------------------------------------------
# The claim that matters: it can actually be replayed
# ---------------------------------------------------------------------------


class TestReproduceRoundTrip:
    """
    Provenance that cannot be replayed is decoration.

    This performs the exact sequence `reproduce.py` performs - restore the
    working tree to the recorded commit, then apply the captured diffs - and
    compares file-content digests before and after.
    """

    @staticmethod
    def _replay(repo: Path, state, diff_dir: Path) -> None:
        """Restore to the recorded commit and re-apply the captured diffs."""
        _git(repo, "restore", "--source", state.commit, "--worktree", ":/")
        for name in _git(repo, "ls-files", "--others", "--exclude-standard").split():
            (repo / name).unlink()

        for index, chunk in enumerate((state.tracked_diff, state.untracked_diff)):
            if not chunk:
                continue
            patch = diff_dir / f"{index}.diff"
            patch.write_text(chunk, encoding="utf-8")
            _git(repo, "apply", "--whitespace=nowarn", str(patch))

    def test_restore_then_apply_reconstructs_the_tree(
        self, dirty_repo: Path, tmp_path: Path
    ):
        state = capture_git_state(dirty_repo)
        expected = _tree_digests(dirty_repo)

        self._replay(dirty_repo, state, tmp_path)

        actual = _tree_digests(dirty_repo)

        # Gitignored files are deliberately outside provenance: they are never
        # captured, and replay does not delete them either. Compare only the
        # set provenance actually claims to reconstruct.
        ignored = {"debug.log", "secrets/token.txt"}
        strip = lambda tree: {k: v for k, v in tree.items() if k not in ignored}

        assert strip(expected) == strip(actual)
        assert set(strip(actual)) == {
            ".gitignore",
            "module.py",
            "pkg/inner.py",
            "added.py",
            "pkg/nested_new.py",
        }

    def test_replay_is_a_no_op_for_a_clean_tree(self, repo: Path, tmp_path: Path):
        state = capture_git_state(repo)
        expected = _tree_digests(repo)
        self._replay(repo, state, tmp_path)
        assert _tree_digests(repo) == expected


# ---------------------------------------------------------------------------
# Repository discovery
# ---------------------------------------------------------------------------


class TestRepoDiscovery:
    """Root discovery must be explicit about failure."""

    def test_finds_root_from_a_subdirectory(self, repo: Path):
        nested = repo / "pkg"
        assert find_repo_root(nested).resolve() == repo.resolve()

    def test_outside_a_repository_raises(self, tmp_path: Path):
        outside = tmp_path / "not_a_repo"
        outside.mkdir()
        with pytest.raises(NotAGitRepositoryError):
            find_repo_root(outside)

    def test_missing_git_binary_raises(self, repo: Path, monkeypatch):
        monkeypatch.setattr(
            "src.vin_ocr.tracking.git_state.shutil.which", lambda _name: None
        )
        with pytest.raises(GitUnavailableError):
            capture_git_state(repo)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TestEnvironmentCapture:
    """Dependency versions are read from metadata, never by importing."""

    def test_records_the_running_interpreter(self):
        import platform

        state = capture_environment()
        assert state.python_version == platform.python_version()
        assert state.platform_machine == platform.machine()

    def test_absent_distributions_are_omitted_not_recorded_as_none(self):
        state = capture_environment(["definitely-not-a-real-distribution-xyz"])
        assert state.packages == {}

    def test_present_distribution_is_recorded(self):
        state = capture_environment(["pytest"])
        assert state.packages["pytest"] == resolve_version("pytest")

    def test_unknown_distribution_resolves_to_none(self):
        assert resolve_version("definitely-not-a-real-distribution-xyz") is None

    def test_params_are_namespaced(self):
        params = capture_environment(["pytest"]).to_params()
        assert params["pkg_pytest"]
        assert "python_version" in params
        assert "platform" in params


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class TestDatasetFingerprint:
    """Fingerprints must be deterministic and honest about their strength."""

    @staticmethod
    def _dataset(root: Path, files: Sequence[tuple]) -> Path:
        for name, content in files:
            _write(root / name, content)
        return root

    def test_missing_directory_is_recorded_not_raised(self, tmp_path: Path):
        result = fingerprint_directory(tmp_path / "absent")
        assert result.exists is False
        assert result.file_count == 0
        assert result.total_bytes == 0

    def test_is_deterministic(self, tmp_path: Path):
        root = self._dataset(tmp_path / "d", [("a.txt", "a"), ("b/c.txt", "cc")])
        assert (
            fingerprint_directory(root).manifest_sha256
            == fingerprint_directory(root).manifest_sha256
        )

    def test_counts_files_and_bytes(self, tmp_path: Path):
        root = self._dataset(tmp_path / "d", [("a.txt", "ab"), ("b/c.txt", "cde")])
        result = fingerprint_directory(root)
        assert result.file_count == 2
        assert result.total_bytes == 5

    def test_detects_an_added_file(self, tmp_path: Path):
        root = self._dataset(tmp_path / "d", [("a.txt", "a")])
        before = fingerprint_directory(root).manifest_sha256
        _write(root / "b.txt", "b")
        assert fingerprint_directory(root).manifest_sha256 != before

    def test_detects_a_size_change(self, tmp_path: Path):
        root = self._dataset(tmp_path / "d", [("a.txt", "a")])
        before = fingerprint_directory(root).manifest_sha256
        _write(root / "a.txt", "aaaa")
        assert fingerprint_directory(root).manifest_sha256 != before

    def test_stat_only_mode_cannot_see_same_size_edits(self, tmp_path: Path):
        """
        Pins the documented limitation rather than pretending it away. The
        recorded ``content_hashed`` flag is what stops a stat-only manifest
        being read as a content hash.
        """
        root = self._dataset(tmp_path / "d", [("a.txt", "aaaa")])
        before = fingerprint_directory(root)
        _write(root / "a.txt", "bbbb")
        after = fingerprint_directory(root)

        assert after.manifest_sha256 == before.manifest_sha256
        assert before.content_hashed is False

    def test_content_mode_does_see_same_size_edits(self, tmp_path: Path):
        root = self._dataset(tmp_path / "d", [("a.txt", "aaaa")])
        before = fingerprint_directory(root, include_content=True)
        _write(root / "a.txt", "bbbb")
        after = fingerprint_directory(root, include_content=True)

        assert after.manifest_sha256 != before.manifest_sha256
        assert before.content_hashed is True


class TestDvcPointers:
    """DVC hashes are the authoritative data digests when available."""

    def test_no_dvc_files_yields_nothing(self, tmp_path: Path):
        assert read_dvc_pointers(tmp_path) == ()

    def test_reads_a_dvc_pointer_file(self, tmp_path: Path):
        (tmp_path / "data.dvc").write_text(
            yaml.safe_dump(
                {
                    "outs": [
                        {
                            "md5": "abc123.dir",
                            "size": 4096,
                            "nfiles": 43,
                            "path": "data",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        pointers = read_dvc_pointers(tmp_path)
        assert len(pointers) == 1
        assert pointers[0].target == "data"
        assert pointers[0].md5 == "abc123.dir"
        assert pointers[0].nfiles == 43

    def test_reads_stages_from_dvc_lock(self, tmp_path: Path):
        (tmp_path / "dvc.lock").write_text(
            yaml.safe_dump(
                {
                    "schema": "2.0",
                    "stages": {
                        "prepare": {
                            "cmd": "python prepare.py",
                            "outs": [{"path": "splits", "md5": "deadbeef"}],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        pointers = read_dvc_pointers(tmp_path)
        assert len(pointers) == 1
        assert pointers[0].target == "splits"
        assert pointers[0].source.startswith("dvc.lock#prepare")

    def test_malformed_yaml_does_not_abort_the_scan(self, tmp_path: Path):
        (tmp_path / "broken.dvc").write_text("outs: [oh no: :\n", encoding="utf-8")
        (tmp_path / "good.dvc").write_text(
            yaml.safe_dump({"outs": [{"md5": "aa", "path": "kept"}]}), encoding="utf-8"
        )
        assert [p.target for p in read_dvc_pointers(tmp_path)] == ["kept"]


# ---------------------------------------------------------------------------
# Logbook
# ---------------------------------------------------------------------------


class TestLogbook:
    """
    The logbook resolves from the repository root, not the process CWD.

    The obvious ``open("LOGBOOK.md")`` reads a different file, or none,
    depending on where the process was launched from - and raises
    FileNotFoundError into the middle of a training run.
    """

    def test_absent_logbook_returns_none(self, repo: Path):
        assert read_logbook(repo) is None

    def test_reads_from_the_repo_root_regardless_of_cwd(
        self, repo: Path, tmp_path: Path, monkeypatch
    ):
        _write(repo / "LOGBOOK.md", "# Entry\n\nSwitched to cosine schedule.\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        assert "cosine schedule" in read_logbook(repo)

    def test_oversized_logbook_is_truncated(self, repo: Path):
        _write(repo / "LOGBOOK.md", "y" * 50_000)
        text = read_logbook(repo)
        assert text.endswith("[truncated]\n")
        assert len(text) < 50_000


# ---------------------------------------------------------------------------
# Aggregate record
# ---------------------------------------------------------------------------


class TestProvenanceRecord:
    """The aggregate must serialise cleanly and report its own limits."""

    def test_captures_all_three_legs(self, dirty_repo: Path):
        record = capture_provenance(dirty_repo, dataset_roots=[dirty_repo / "pkg"])
        assert record.git.commit
        assert record.environment.python_version
        assert len(record.data.datasets) == 1
        assert record.captured_at.endswith("+00:00")

    def test_is_reproducible_when_everything_was_embedded(self, dirty_repo: Path):
        assert capture_provenance(dirty_repo).is_reproducible is True

    def test_is_not_reproducible_when_a_file_was_skipped(self, repo: Path):
        (repo / "blob.bin").write_bytes(b"\x00binary\x00")
        record = capture_provenance(repo)
        assert record.is_reproducible is False
        assert record.to_params()["provenance_reproducible"] == "false"

    def test_params_are_flat_strings(self, dirty_repo: Path):
        params = capture_provenance(dirty_repo).to_params()
        assert all(isinstance(k, str) for k in params)
        assert all(isinstance(v, str) for v in params.values())
        assert params["git_commit"]
        assert params["git_dirty"] == "true"

    def test_to_dict_is_json_serialisable_and_omits_diff_text(self, dirty_repo: Path):
        import json

        payload = capture_provenance(dirty_repo).to_dict()
        encoded = json.dumps(payload)
        assert "has_tracked_diff" in payload["git"]
        assert "tracked_diff" not in payload["git"], "diff belongs in an artifact"
        assert len(encoded) < 20_000


# ---------------------------------------------------------------------------
# Backend configuration (no MLflow import required)
# ---------------------------------------------------------------------------


class TestTrackingUriResolution:
    """Local-first by default; a single env var switches to a remote store."""

    def test_defaults_to_a_local_sqlite_store(self, repo: Path, monkeypatch):
        """
        Not the './mlruns' file store: MLflow 3 put the filesystem tracking
        backend in maintenance mode and raises unless MLFLOW_ALLOW_FILE_STORE
        is set, so a file-store default ships broken on current installs.
        """
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        uri = resolve_tracking_uri(repo)
        assert uri.startswith("sqlite:///")
        assert uri.endswith("/mlflow.db")

    def test_artifact_root_is_anchored_to_the_repo_not_the_cwd(
        self, repo: Path, tmp_path: Path, monkeypatch
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        root = resolve_artifact_root(repo)
        assert root.startswith("file://")
        assert root.endswith("/mlartifacts")
        assert str(repo.resolve()) in root

    def test_environment_variable_takes_precedence(self, repo: Path, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.invalid/mlflow")
        assert resolve_tracking_uri(repo) == "https://dagshub.invalid/mlflow"

    def test_blank_environment_variable_falls_back_to_local(
        self, repo: Path, monkeypatch
    ):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "   ")
        assert resolve_tracking_uri(repo).startswith("sqlite:///")

    def test_reproduce_command_names_the_run(self):
        command = reproduce_command("abc123")
        assert "src.vin_ocr.tracking.reproduce" in command
        assert command.endswith("abc123")

    def test_reproduce_command_carries_a_remote_uri(self):
        command = reproduce_command("abc123", "https://dagshub.invalid/mlflow")
        assert "--tracking-uri https://dagshub.invalid/mlflow" in command


@pytest.mark.parametrize("entry", ["mlflow.db", "mlartifacts/"])
def test_local_store_is_gitignored(entry: str):
    """
    The default store lives inside the working tree; committing run records
    would bloat the repository and re-create the 'results checked into git
    with no writer left to explain them' situation the audit found in results/.
    """
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    ignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert entry in ignore_text, f"add '{entry}' to .gitignore"
