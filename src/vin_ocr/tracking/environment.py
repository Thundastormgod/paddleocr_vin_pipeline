"""
Environment provenance - the "what was installed" half of reproducibility.

A commit SHA pins the source but not the interpreter or the dependency
versions the source ran against. This project has four competing dependency
manifests (``requirements.txt`` pins ``paddleocr==3.3.3``, ``pyproject.toml``
floors it at ``>=2.9.0``, ``environment.yml`` selects the GPU build of
paddlepaddle, ``requirements_optuna.txt`` disagrees on optuna) and no lockfile,
so two checkouts of the same commit can legitimately run different code.
Recording the resolved versions makes that visible per run instead of leaving
it to be discovered later.

Versions are read from installed package metadata via ``importlib.metadata``,
never by importing the package. Importing paddle costs seconds and can abort
the process when a GPU build meets a machine without CUDA; reading metadata
does neither.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Dict, Mapping, Optional, Sequence, Tuple

#: Distributions whose versions materially change model behaviour or metrics.
#: Names are distribution names as they appear on PyPI, not import names -
#: 'opencv-python' imports as 'cv2', 'paddlepaddle' imports as 'paddle'.
TRACKED_DISTRIBUTIONS: Tuple[str, ...] = (
    "paddlepaddle",
    "paddlepaddle-gpu",
    "paddleocr",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "Pillow",
    "onnx",
    "onnxruntime",
    "optuna",
    "mlflow",
    "dvc",
    "zenml",
    "rapidfuzz",
)


@dataclass(frozen=True)
class EnvironmentState:
    """
    Immutable snapshot of the execution environment.

    Attributes:
        python_version: Interpreter version, e.g. "3.12.13".
        python_implementation: e.g. "CPython".
        platform_system: e.g. "Darwin", "Linux".
        platform_release: Kernel or OS release string.
        platform_machine: Processor architecture, e.g. "arm64", "x86_64".
        packages: Distribution name -> resolved version, for installed
            members of TRACKED_DISTRIBUTIONS only. Absent distributions are
            omitted rather than recorded as None, so "missing" and "present"
            are never confused.
    """

    python_version: str
    python_implementation: str
    platform_system: str
    platform_release: str
    platform_machine: str
    packages: Mapping[str, str]

    def to_params(self) -> Dict[str, str]:
        """
        Return short, flat key/value pairs suitable for MLflow run params.

        Package versions are namespaced under ``pkg_`` so they cannot collide
        with metric or hyperparameter names.
        """
        params: Dict[str, str] = {
            "python_version": self.python_version,
            "platform": f"{self.platform_system}-{self.platform_machine}",
        }
        for name, version in sorted(self.packages.items()):
            params[f"pkg_{name.replace('-', '_')}"] = version
        return params

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "platform_machine": self.platform_machine,
            "packages": dict(sorted(self.packages.items())),
        }


def resolve_version(distribution: str) -> Optional[str]:
    """
    Return the installed version of a distribution, or None if absent.

    Args:
        distribution: PyPI distribution name (not the import name).

    Returns:
        Version string, or None when the distribution is not installed.
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def capture_environment(
    distributions: Sequence[str] = TRACKED_DISTRIBUTIONS,
) -> EnvironmentState:
    """
    Capture interpreter, platform and dependency versions.

    Args:
        distributions: Distribution names to resolve. Defaults to
            TRACKED_DISTRIBUTIONS.

    Returns:
        A populated EnvironmentState.

    Complexity:
        O(n) metadata lookups for n distributions; no imports are performed.
    """
    packages: Dict[str, str] = {}
    for name in distributions:
        version = resolve_version(name)
        if version is not None:
            packages[name] = version

    return EnvironmentState(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        packages=packages,
    )


def python_executable() -> str:
    """Return the running interpreter path, for reproduce commands."""
    return sys.executable
