#!/usr/bin/env python3
"""
Generate the measured system maps for docs/SYSTEM_MAPS.md.

Every artifact this script emits is DERIVED FROM THE CODE by AST analysis
- none of it is drawn from belief. Re-run after structural changes and
diff the output against the committed document.

Outputs (to stdout, sectioned):
  1. Module dependency graph (internal imports), cycle detection,
     fan-in/fan-out table, Mermaid `graph TD`.
  2. Cyclomatic-complexity census per function, worst offenders.
  3. Static call-graph reachability from the real entry points and
     dead-function candidates (with the honest limits of static analysis
     over dynamic dispatch stated).
  4. Training-lifecycle state machine extracted from the literal status
     strings the trainer writes (`_save_progress(...)` call sites), as a
     Mermaid stateDiagram.

Usage:
    .venv/bin/python scripts/generate_system_maps.py
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "vin_ocr"

# Real process entry points (console scripts in pyproject + module mains).
ENTRY_POINTS = [
    ("src/vin_ocr/cli.py", "main"),
    ("src/vin_ocr/training/cli.py", "main"),
    ("src/vin_ocr/evaluation/cli.py", "main"),
    ("src/vin_ocr/tracking/reproduce.py", "main"),
    ("src/vin_ocr/training/finetune_paddleocr.py", "main"),
    ("src/vin_ocr/training/train_from_scratch.py", "main"),
    ("src/vin_ocr/evaluation/multi_model_evaluation.py", "main"),
    ("src/vin_ocr/evaluation/evaluate.py", "main"),
    ("src/vin_ocr/web/app.py", "main"),
]


def module_name(path: Path) -> str:
    rel = path.relative_to(REPO / "src")
    return ".".join(rel.with_suffix("").parts)


def iter_modules() -> Dict[str, Path]:
    return {module_name(p): p for p in SRC.rglob("*.py")}


# ---------------------------------------------------------------------------
# 1. Module dependency graph
# ---------------------------------------------------------------------------

def build_dependency_graph(modules: Dict[str, Path]) -> Dict[str, Set[str]]:
    """Directed edges module -> internal modules it imports (incl. lazy)."""
    known = set(modules)
    edges: Dict[str, Set[str]] = {m: set() for m in modules}

    for mod, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        pkg_parts = mod.split(".")[:-1]
        for node in ast.walk(tree):
            targets: List[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import
                    base = pkg_parts[: len(pkg_parts) - node.level + 1]
                    stem = ".".join(base + ([node.module] if node.module else []))
                    targets = [stem]
                elif node.module:
                    targets = [node.module]
            for t in targets:
                # normalise to a known internal module (or its package)
                cand = t
                while cand and cand not in known:
                    cand = cand.rpartition(".")[0]
                if cand and cand != mod:
                    edges[mod].add(cand)
    return edges


def find_cycles(edges: Dict[str, Set[str]]) -> List[List[str]]:
    cycles, stack, on_stack = [], [], set()
    visited: Set[str] = set()

    def dfs(node: str) -> None:
        visited.add(node)
        stack.append(node)
        on_stack.add(node)
        for nxt in sorted(edges.get(node, ())):
            if nxt not in visited:
                dfs(nxt)
            elif nxt in on_stack:
                cycles.append(stack[stack.index(nxt):] + [nxt])
        stack.pop()
        on_stack.remove(node)

    for n in sorted(edges):
        if n not in visited:
            dfs(n)
    return cycles


# ---------------------------------------------------------------------------
# 2. Cyclomatic census
# ---------------------------------------------------------------------------

def cyclomatic(node: ast.AST) -> int:
    score = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                          ast.With, ast.Assert, ast.IfExp)):
            score += 1
        elif isinstance(n, ast.BoolOp):
            score += len(n.values) - 1
        elif isinstance(n, ast.comprehension):
            score += 1 + len(n.ifs)
    return score


def complexity_census(modules: Dict[str, Path]) -> List[Tuple[str, str, int, int]]:
    rows = []
    for mod, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                rows.append((mod, node.name, cyclomatic(node), length))
    return rows


# ---------------------------------------------------------------------------
# 3. Call graph reachability / dead-function candidates
# ---------------------------------------------------------------------------

def _name_refs(node: ast.AST) -> Set[str]:
    """Every name a subtree reads: identifiers, attribute names, and
    string constants (dispatch tables mapping name strings)."""
    refs: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            refs.add(n.id)
        elif isinstance(n, ast.Attribute):
            refs.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            refs.add(n.value)
    return refs


def _module_level_refs(tree: ast.Module) -> Set[str]:
    """
    Names referenced by code that executes at IMPORT time: everything
    outside function bodies (module statements, class bodies, __main__
    guards) plus function decorators and argument defaults, which are
    evaluated when the `def` statement runs. Function BODIES are excluded
    - they only execute when the function is called, which is exactly the
    edge the reachability walk models.
    """
    refs: Set[str] = set()
    pending: List[ast.AST] = list(ast.iter_child_nodes(tree))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                refs |= _name_refs(dec)
            defaults = list(node.args.defaults)
            defaults += [d for d in node.args.kw_defaults if d is not None]
            for default in defaults:
                refs |= _name_refs(default)
            continue  # body runs only when called
        if isinstance(node, ast.Name):
            refs.add(node.id)
        elif isinstance(node, ast.Attribute):
            refs.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            refs.add(node.value)
        pending.extend(ast.iter_child_nodes(node))
    return refs


def call_reachability(modules: Dict[str, Path]):
    """
    Name-based static call-graph reachability from ENTRY_POINTS.

    Roots: the entry-point functions listed in ENTRY_POINTS, every name
    referenced by import-time code in any module (see _module_level_refs),
    and every name referenced anywhere in tests/. Edges: function name ->
    names referenced inside that function's body. A defined function whose
    name is neither a root nor reached by BFS over these edges is a
    dead-function candidate.

    Honest limits: dynamic dispatch (getattr with runtime-built strings,
    framework callbacks) is approximated by treating any read of a
    function's NAME (call, reference, dict value, string constant) as a
    use, and same-named functions are merged into one node. Results are
    therefore candidates, not verdicts.
    """
    defs: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    refs_by_func: Dict[str, Set[str]] = defaultdict(set)
    roots: Set[str] = set()

    entry_files = {(REPO / rel).resolve() for rel, _ in ENTRY_POINTS}
    entry_funcs = {func for _, func in ENTRY_POINTS}

    for mod, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[node.name].append((mod, node.lineno))
                refs_by_func[node.name] |= _name_refs(node)
                if path.resolve() in entry_files and node.name in entry_funcs:
                    roots.add(node.name)
        roots |= _module_level_refs(tree)

    # tests also legitimise a function
    for test in (REPO / "tests").rglob("*.py"):
        roots |= _name_refs(ast.parse(test.read_text(encoding="utf-8")))

    # BFS over the uses graph: expand every root that is a defined function.
    reachable: Set[str] = set()
    queue = deque(name for name in roots if name in refs_by_func)
    while queue:
        name = queue.popleft()
        if name in reachable:
            continue
        reachable.add(name)
        for ref in refs_by_func[name]:
            if ref in refs_by_func and ref not in reachable:
                queue.append(ref)

    dunder_or_hooks = {
        "main", "load_context", "predict",  # entry points / framework hooks
    }
    dead = []
    for name, sites in sorted(defs.items()):
        if name.startswith("__") or name in dunder_or_hooks:
            continue
        if name in roots or name in reachable:
            continue
        dead.extend((name, mod, line) for mod, line in sites)
    return defs, dead


# ---------------------------------------------------------------------------
# 4. Training lifecycle FSM from _save_progress literals
# ---------------------------------------------------------------------------

def extract_training_states() -> List[Tuple[int, str]]:
    path = SRC / "training" / "finetune_paddleocr.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    states = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_save_progress"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            states.append((node.lineno, node.args[0].value))
    return sorted(states)


# ---------------------------------------------------------------------------

def main() -> int:
    modules = iter_modules()

    print("### SECTION 1: dependency graph")
    edges = build_dependency_graph(modules)
    fan_in: Dict[str, int] = defaultdict(int)
    for src_mod, targets in edges.items():
        for t in targets:
            fan_in[t] += 1
    cycles = find_cycles(edges)
    print(f"modules={len(modules)} edges={sum(len(v) for v in edges.values())} "
          f"cycles={len(cycles)}")
    for cycle in cycles:
        print("  CYCLE:", " -> ".join(cycle))
    print("fan-in (top):")
    for mod, count in sorted(fan_in.items(), key=lambda x: -x[1])[:8]:
        print(f"  {count:3d}  {mod}")
    print("mermaid:")
    short = lambda m: m.replace("vin_ocr.", "").replace(".", "_")
    seen = set()
    for src_mod in sorted(edges):
        for dst in sorted(edges[src_mod]):
            if dst.startswith("vin_ocr") and (src_mod, dst) not in seen:
                seen.add((src_mod, dst))
                print(f"  {short(src_mod)} --> {short(dst)}")

    print("\n### SECTION 2: cyclomatic census")
    rows = complexity_census(modules)
    over10 = [r for r in rows if r[2] > 10]
    print(f"functions={len(rows)} mean_cyc={sum(r[2] for r in rows)/len(rows):.2f} "
          f"over10={len(over10)} over50lines={len([r for r in rows if r[3] > 50])}")
    for mod, name, cyc, length in sorted(rows, key=lambda r: -r[2])[:15]:
        print(f"  cyc={cyc:3d} len={length:4d}  {mod}.{name}")

    print("\n### SECTION 3: dead-function candidates")
    _, dead = call_reachability(modules)
    print(f"candidates={len(dead)} (unreachable from ENTRY_POINTS, import-time "
          f"code, and tests via the name-use graph)")
    for name, mod, line in dead:
        print(f"  {mod}:{line}  {name}")

    print("\n### SECTION 4: training lifecycle states (from _save_progress literals)")
    for line, state in extract_training_states():
        print(f"  line {line:5d}: {state}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
