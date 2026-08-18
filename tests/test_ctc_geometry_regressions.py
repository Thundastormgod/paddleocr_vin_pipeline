"""
Regression tests for CTC geometry (C4) and non-finite loss guarding.

C4, confirmed by arithmetic at HEAD before the fix: the SVTR_LCNet builder
in train_from_scratch.py used isotropic stride 2 in its stem and all four
stages - a 2^5 = 32x downsample in BOTH axes. At the configured input width
of 320 that leaves T = 10 output timesteps for a 17-character VIN target,
so no CTC alignment lattice exists, the loss is inf/nan from the first
batch, and `epoch_loss += loss.item()` averaged the poison silently. The
correct pattern - stride (2, 1) so height compresses while width is
preserved - already existed 250 lines later in the same file (PPHGNet
blocks) and had simply not been propagated.

These tests are structural (AST over the builder source): paddle is not
installed in this environment, so the network cannot be constructed. The
geometry, however, is fully determined by the stride literals, and the
guard is fully determined by the accumulation sites.
"""

import ast
import math
from pathlib import Path

import pytest

from src.vin_ocr.training.metrics import require_finite_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_PATH = REPO_ROOT / "src" / "vin_ocr" / "training" / "train_from_scratch.py"
FINETUNE_PATH = REPO_ROOT / "src" / "vin_ocr" / "training" / "finetune_paddleocr.py"

#: The training input width every scratch config in this repo uses.
INPUT_WIDTH = 320
#: A VIN is 17 characters; CTC needs at least one timestep per character.
VIN_LENGTH = 17


def _width_stride(node: ast.expr) -> int:
    """
    Width component of a stride argument: int -> both axes, tuple -> (h, w).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) == 2:
        w = node.elts[1]
        if isinstance(w, ast.Constant) and isinstance(w.value, int):
            return w.value
    raise AssertionError(f"unrecognised stride literal at line {node.lineno}")


def _svtr_lcnet_class() -> ast.ClassDef:
    tree = ast.parse(SCRATCH_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SVTRLCNet":
            return node
    raise AssertionError("SVTRLCNet class not found in train_from_scratch.py")


def _backbone_width_downsample(cls: ast.ClassDef) -> int:
    """
    Cumulative width downsample of the SVTRLCNet backbone, from the AST:
    the stem Conv2D's ``stride=`` keyword times the stride argument of
    every ``self._make_stage(...)`` call.
    """
    product = 1
    found_stem = False
    found_stages = 0

    for node in ast.walk(cls):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Stem: nn.Conv2D(..., stride=2, ...) - the first conv in __init__
        if (
            isinstance(func, ast.Attribute) and func.attr == "Conv2D"
            and not found_stem
        ):
            for kw in node.keywords:
                if kw.arg == "stride":
                    product *= _width_stride(kw.value)
                    found_stem = True
        # Stages: self._make_stage(in, out, stride)
        if isinstance(func, ast.Attribute) and func.attr == "_make_stage":
            assert len(node.args) == 3, "unexpected _make_stage signature"
            product *= _width_stride(node.args[2])
            found_stages += 1

    assert found_stem, "stem Conv2D stride not found"
    assert found_stages == 4, f"expected 4 stages, found {found_stages}"
    return product


class TestSVTRLCNetCTCGeometry:
    """WAS BROKEN: 32x width downsample -> T=10 < 17 -> empty CTC lattice."""

    def test_width_downsample_leaves_enough_timesteps(self):
        cls = _svtr_lcnet_class()
        downsample = _backbone_width_downsample(cls)
        timesteps = INPUT_WIDTH // downsample
        assert timesteps >= VIN_LENGTH, (
            f"SVTR_LCNet width downsample {downsample}x leaves "
            f"{timesteps} CTC timesteps for a {VIN_LENGTH}-char target - "
            f"no alignment lattice exists and the loss is inf from batch 0"
        )

    def test_late_stages_preserve_width(self):
        """The exact fix: stages 2-4 stride (2, 1), matching PPHGNet."""
        cls = _svtr_lcnet_class()
        stage_strides = [
            node.args[2]
            for node in ast.walk(cls)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_make_stage"
        ]
        late = stage_strides[1:]
        assert len(late) == 3
        for stride_node in late:
            assert _width_stride(stride_node) == 1, (
                f"stage at line {stride_node.lineno} downsamples width"
            )


class TestNonFiniteLossGuards:
    """A NaN/inf loss must halt training, not poison the epoch average."""

    def test_guard_raises_on_nan_and_inf(self):
        with pytest.raises(RuntimeError) as excinfo:
            require_finite_loss(float("nan"), context="unit test")
        assert "unit test" in str(excinfo.value)
        with pytest.raises(RuntimeError):
            require_finite_loss(float("inf"), context="unit test")
        with pytest.raises(RuntimeError):
            require_finite_loss(-math.inf, context="unit test")

    def test_guard_passes_finite_values_through(self):
        assert require_finite_loss(0.0, context="t") == 0.0
        assert require_finite_loss(2.5, context="t") == 2.5

    @pytest.mark.parametrize("path", [SCRATCH_PATH, FINETUNE_PATH])
    def test_every_loss_accumulation_is_guarded(self, path):
        """
        AST invariant: every ``x += <expr involving loss.item()>`` in the
        trainers must route through require_finite_loss. This is how the
        C4 failure stayed invisible - inf entered the average silently.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.AugAssign):
                continue
            calls = [n for n in ast.walk(node.value) if isinstance(n, ast.Call)]
            touches_loss_item = any(
                isinstance(c.func, ast.Attribute) and c.func.attr == "item"
                for c in calls
            )
            if not touches_loss_item:
                continue
            guarded = any(
                isinstance(c.func, ast.Name)
                and c.func.id == "require_finite_loss"
                for c in calls
            )
            if not guarded:
                offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == [], (
            f"unguarded loss accumulation (NaN poisons the average "
            f"silently): {offenders}"
        )
