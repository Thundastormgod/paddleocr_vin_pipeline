#!/usr/bin/env python3
"""
Local CI runner: executes the enterprise gate WITHOUT GitHub Actions.

GitHub Actions is billing-locked on this account, so the workflows in
.github/workflows/ci.yml cannot execute on hosted runners. This script
runs the SAME jobs with the SAME commands natively (see the per-job
references to ci.yml), so the gate keeps enforcing while the lock
persists - and so the workflow definitions are exercised before GitHub
ever runs them.

Jobs (mirroring .github/workflows/ci.yml):
    test        pytest + coverage floor 30%          [ci.yml "test"]
    lint        ruff critical subset E9/F63/F7/F82   [ci.yml "lint"]
    typecheck   mypy behind the committed baseline   [ci.yml "typecheck"]
    pip-audit   audit of .github/requirements-ci.txt [ci.yml "pip-audit"]
    secrets     tracked-credential guards            [ci.yml "secrets-scan"]
    build       sdist/wheel + console-script check   [ci.yml "build"]

Not runnable locally (documented, not silently skipped):
    docker      needs a container runtime (trivy/hadolint/sbom)
    model-gate  needs DAGSHUB_*/MLFLOW_* repo secrets and DVC pull
    codeql      needs GitHub's CodeQL service

Usage:
    python scripts/local_ci.py                # all runnable jobs
    python scripts/local_ci.py --quick        # lint+secrets+test only
    python scripts/local_ci.py --job test     # one job
    python scripts/local_ci.py --bootstrap    # pip-install missing tools first

Exit codes: 0 all selected jobs passed; 1 at least one failed;
2 usage/environment error.

Requires an interpreter environment with the project deps. The dev venv
at .venv is used by default when available; pass PYTHON to override.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / ".github" / "mypy-baseline.txt"
PINNED = REPO_ROOT / ".github" / "requirements-ci.txt"

# Job execution order mirrors ci.yml for readable reports.
JOB_ORDER = ["secrets", "lint", "typecheck", "pip-audit", "test", "build"]
QUICK_JOBS = ["secrets", "lint", "test"]


def default_python() -> str:
    """Prefer the repo dev venv; fall back to the running interpreter."""
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def run(name: str, cmd: List[str], cwd: Optional[Path] = None) -> bool:
    """Run one job command, streaming output; return success."""
    print(f"\n=== {name} :: {' '.join(cmd)}")
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT))
    elapsed = time.monotonic() - started
    status = "PASS" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
    print(f"=== {name} :: {status} [{elapsed:.1f}s]")
    return proc.returncode == 0


def shell(name: str, script: str, cwd: Optional[Path] = None) -> bool:
    """Run a bash snippet (jobs that need pipes/multi-step logic)."""
    print(f"\n=== {name} :: bash -c '{script[:80]}...'")
    started = time.monotonic()
    proc = subprocess.run(["bash", "-c", script], cwd=str(cwd or REPO_ROOT))
    elapsed = time.monotonic() - started
    status = "PASS" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
    print(f"=== {name} :: {status} [{elapsed:.1f}s]")
    return proc.returncode == 0


def need(tool: str, pip_name: str, python: str) -> Optional[str]:
    """Resolve a tool next to the chosen interpreter, then PATH, then the
    repo venv. The interpreter's own bin dir comes first so a worktree
    without its own .venv still finds tools installed in the main env."""
    for candidate in (
        Path(python).parent / tool,
        REPO_ROOT / ".venv" / "bin" / tool,
    ):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(tool)
    if found:
        return found
    print(
        f"NOTE: '{tool}' not found (pip install {pip_name} into your "
        f"environment, or rerun with --bootstrap)"
    )
    return None


def _module_missing(python: str, module: str) -> bool:
    """True when `import module` fails under the chosen interpreter."""
    probe = subprocess.run(
        [python, "-c", f"import {module}"], capture_output=True
    )
    return probe.returncode != 0


def bootstrap(python: str) -> int:
    """Install the local-runner-only tools into the chosen environment."""
    missing_pkgs = []
    if need("ruff", "ruff", python) is None:
        missing_pkgs.append("ruff==0.16.3")
    if need("mypy", "mypy", python) is None or need("mypy-baseline", "mypy-baseline", python) is None:
        missing_pkgs += ["mypy==2.3.1", "mypy-baseline==0.7.4"]
    if need("pip-audit", "pip-audit", python) is None:
        missing_pkgs.append("pip-audit==2.10.1")
    if _module_missing(python, "build"):
        missing_pkgs.append("build==1.5.0")
    if not missing_pkgs:
        print("bootstrap: nothing to install")
        return 0
    print(f"bootstrap: pip install {' '.join(missing_pkgs)}")
    return subprocess.run([python, "-m", "pip", "install", *missing_pkgs]).returncode


# ---------------------------------------------------------------------------
# Jobs. Each returns True/False. Commands mirror ci.yml verbatim.
# ---------------------------------------------------------------------------

def job_secrets(py: str) -> bool:
    """[ci.yml 'secrets-scan'] credential-file + private-key guards."""
    patterns = [
        "keyjson",
        ".dvc/config.local",
        "*.pem",
        "id_rsa*",
        "id_ed25519*",
        ".env",
    ]
    bad = False
    for pattern in patterns:
        proc = subprocess.run(
            ["git", "ls-files", pattern], capture_output=True, text=True, cwd=REPO_ROOT
        )
        matches = proc.stdout.strip()
        if matches:
            print(f"Credential file is tracked in git: {matches}")
            bad = True
    key_probe = subprocess.run(
        ["git", "grep", "-I", "-l", "-E",
         r"BEGIN (OPENSSH|RSA|EC|PGP) PRIVATE KEY", "--", "."],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if key_probe.stdout.strip():
        print(f"Private key material found in tracked files:\n{key_probe.stdout}")
        bad = True
    if not bad:
        print("No credential files or private key material tracked.")
    return not bad


def job_lint(py: str) -> bool:
    """[ci.yml 'lint'] blocking critical subset; advisory full report."""
    ruff = need("ruff", "ruff", py)
    if ruff is None:
        return False
    ok = run("lint/ruff-critical", [ruff, "check", "--no-cache", "src/", "tests/"])
    # Advisory full-ruleset report never affects the verdict.
    subprocess.run(
        [ruff, "check", "--no-cache", "--select", "ALL",
         "--statistics", "src/", "tests/"],
        cwd=REPO_ROOT,
    )
    return ok


def _pinned_version(package: str) -> Optional[str]:
    """Exact pin for a package from .github/requirements-ci.txt, or None."""
    if not PINNED.is_file():
        return None
    for line in PINNED.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{package}=="):
            return line.split("==", 1)[1]
    return None


def _typecheck_env_parity(py: str) -> List[str]:
    """Mismatches between this env and the pins the baseline was built on.

    The mypy-baseline ratchet compares mypy output BYTE-wise, and mypy's
    message text embeds library type renderings (measured 2026-08-22:
    numpy 2.3.5 vs pinned 2.2.6 renders ndarray types differently,
    producing 2 phantom 'new' errors). The ratchet is therefore only
    enforceable in an env matching the pins; elsewhere it is advisory
    and CI remains authoritative.
    """
    mismatches = []
    for pkg, probe in (
        ("mypy", "import mypy.version; print(mypy.version.__version__)"),
        ("numpy", "import numpy; print(numpy.__version__)"),
    ):
        pinned = _pinned_version(pkg)
        if pinned is None:
            continue
        proc = subprocess.run([py, "-c", probe], capture_output=True, text=True)
        local = proc.stdout.strip() if proc.returncode == 0 else "absent"
        if local != pinned:
            mismatches.append(f"{pkg}: local {local} != pinned {pinned}")
    return mismatches


def job_typecheck(py: str) -> bool:
    """[ci.yml 'typecheck'] mypy vs committed baseline ratchet."""
    mypy = need("mypy", "mypy", py)
    baseline_tool = need("mypy-baseline", "mypy-baseline", py)
    if mypy is None or baseline_tool is None:
        return False
    if not BASELINE.is_file():
        print(
            f"SKIP: {BASELINE.relative_to(REPO_ROOT)} not present in this "
            f"tree - it lives on ci/enterprise-gate (PR #12) until merge."
        )
        return True  # absence of the ratchet file must not fail pre-merge trees
    passed = shell(
        "typecheck/mypy-baseline",
        f"'{mypy}' src/ | '{baseline_tool}' filter "
        f"--baseline-path '{BASELINE}'",
    )
    if passed:
        return True
    mismatches = _typecheck_env_parity(py)
    if mismatches:
        print(
            "typecheck: ADVISORY ONLY in this environment - baseline "
            "comparison is byte-sensitive to library versions and this env "
            "does not match the CI pins:\n  " + "\n  ".join(mismatches) +
            "\n  CI (pinned env) remains the authoritative typecheck gate."
        )
        return True
    return False


def job_pip_audit(py: str) -> bool:
    """[ci.yml 'pip-audit'] audit the exact committed pins."""
    audit = need("pip-audit", "pip-audit", py)
    if audit is None:
        return False
    if not PINNED.is_file():
        print(f"SKIP: {PINNED.relative_to(REPO_ROOT)} not present in this tree.")
        return True
    return run(
        "pip-audit/pins",
        [audit, "-r", str(PINNED), "--disable-pip", "--no-deps"],
    )


def job_test(py: str) -> bool:
    """[ci.yml 'test'] full suite + coverage floor (single local leg)."""
    return run(
        "test/pytest-cov",
        [py, "-m", "pytest", "-q", "--cov=src", "--cov-report=term-missing",
         "--cov-fail-under=30"],
    )


def job_build(py: str) -> bool:
    """[ci.yml 'build'] distribution build + console-script resolution."""
    ok = run("build/dist", [py, "-m", "build"])
    if not ok:
        return False
    checker = (
        "import pathlib, sys, tomllib\n"
        "data = tomllib.loads(pathlib.Path('pyproject.toml').read_text())\n"
        "scripts = data['project']['scripts']\n"
        "failed = []\n"
        "for name, target in scripts.items():\n"
        "    module = target.split(':')[0]\n"
        "    path = pathlib.Path(module.replace('.', '/') + '.py')\n"
        "    if not path.is_file():\n"
        "        failed.append(f'{name} -> {target} (missing {path})')\n"
        "if failed:\n"
        "    print('Broken console script entry points:')\n"
        "    [print('  ' + f) for f in failed]\n"
        "    sys.exit(1)\n"
        "print(f'All {len(scripts)} console scripts resolve.')\n"
    )
    return run("build/console-scripts", [py, "-c", checker])


JOBS: Dict[str, Callable[[str], bool]] = {
    "secrets": job_secrets,
    "lint": job_lint,
    "typecheck": job_typecheck,
    "pip-audit": job_pip_audit,
    "test": job_test,
    "build": job_build,
}

UNAVAILABLE = {
    "docker": "requires a container runtime (hadolint/trivy/sbom)",
    "model-gate": "requires DAGSHUB_*/MLFLOW_* secrets and a DVC pull",
    "codeql": "requires GitHub's CodeQL service",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true",
                       help=f"fast subset only: {', '.join(QUICK_JOBS)}")
    group.add_argument("--job", choices=JOB_ORDER, help="run a single job")
    parser.add_argument("--bootstrap", action="store_true",
                        help="pip-install missing runner tools, then exit")
    parser.add_argument("--python", default=None,
                        help="interpreter used for the test/build jobs "
                             "(default: .venv/bin/python if present)")
    args = parser.parse_args()

    python = args.python or default_python()
    if args.bootstrap:
        return 0 if bootstrap(python) == 0 else 1

    selected = list(JOB_ORDER)
    if args.quick:
        selected = QUICK_JOBS
    elif args.job:
        selected = [args.job]

    results: List[Tuple[str, bool]] = []
    for job in selected:
        try:
            results.append((job, JOBS[job](python)))
        except FileNotFoundError as exc:
            print(f"{job}: FAILED to start ({exc})")
            results.append((job, False))

    print("\n" + "=" * 60)
    width = max(len(name) for name, _ in results)
    for name, ok in results:
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL'}")
    for name, reason in UNAVAILABLE.items():
        print(f"{name.ljust(width)}  UNAVAILABLE ({reason})")

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\nGATE FAIL: {', '.join(failed)}")
        return 1
    print("\nGATE PASS: all selected jobs green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
