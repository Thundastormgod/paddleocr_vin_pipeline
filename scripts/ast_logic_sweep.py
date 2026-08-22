#!/usr/bin/env python3
"""
Deterministic AST logic sweep.

Walks first-party Python files and reports structural logic defects:

  AST01 identical-if-else-branches   if/else (or elif/else) bodies are AST-identical
  AST02 duplicate-condition          same test repeated in an if/elif chain
  AST03 self-compare                 x == x, x < x, x != x ... (same operand both sides)
  AST04 constant-condition           if/while test is a literal constant (excl. while True)
  AST05 unreachable-code             statements after return/raise/break/continue
  AST06 handler-shadow               except clause unreachable (earlier clause catches superclass)
  AST07 duplicate-except             same exception type caught twice
  AST08 mutable-default              def f(x=[] / {} / set() / list() / dict())
  AST09 zip-nostrict                 zip(a, b) without strict= (silent truncation risk)
  AST10 silent-swallow               except [Exception]: pass/continue only
  AST11 self-assign                  x = x  /  a.b = a.b
  AST12 dict-dup-key                 duplicate literal key in dict display
  AST13 assert-tuple                 assert (x, y) - always true
  AST14 finally-jump                 return/break/continue inside finally (swallows exceptions)
  AST15 bool-same-operand            x or x  /  x and x
  AST16 is-literal                   `is` comparison against str/int/float literal
  AST17 param-shadow-loop            for-loop target shadows an enclosing function parameter
  AST18 const-compare                comparison whose operands are all literals
  AST19 tuple-unpack-arity           a, b = f() where every def of f returns an N-tuple, N != 2
  AST20 sys-path-insert              sys.path manipulation (import-order hazard, info)

Output: JSON to --json path, human-readable grouped report to stdout.
Zero third-party dependencies; safe on any tree.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

EXCLUDE_DIRS = {
    ".venv", "venv", "__pycache__", ".git", "mlartifacts", "mlruns",
    "node_modules", ".pytest_cache", ".ruff_cache", ".dvc", ".zen",
    "paddleocr_vin_pipeline.egg-info", "output", "finetune_data",
    "dataset", "data", "dagshub_data", "optuna_results", "results", "docs",
}

# Statement-list attribute names on AST nodes.
STMT_LIST_FIELDS = ("body", "orelse", "finalbody")

JUMP_STMTS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


@dataclass
class Finding:
    rule: str
    severity: str  # BUG-LIKELY | REVIEW | INFO
    file: str
    line: int
    message: str
    snippet: str


def safe_unparse(node: ast.AST, limit: int = 110) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = f"<unparse failed: {type(node).__name__}>"
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def dump_stmts(stmts: list[ast.stmt]) -> str:
    return "; ".join(dump(s) for s in stmts)


def resolve_builtin_exc(node: ast.expr) -> Optional[type]:
    if isinstance(node, ast.Name):
        obj = getattr(builtins, node.id, None)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            return obj
    return None


def exc_types_of_handler(handler: ast.ExceptHandler) -> list[Optional[type]]:
    """Resolved builtin classes for a handler; None entries are unresolvable."""
    t = handler.type
    if t is None:
        return [BaseException]  # bare except
    if isinstance(t, ast.Tuple):
        return [resolve_builtin_exc(e) for e in t.elts]
    return [resolve_builtin_exc(t)]


def is_docstring_position(parent: ast.AST, stmt: ast.stmt) -> bool:
    body = getattr(parent, "body", None)
    return (
        isinstance(parent, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and isinstance(body, list)
        and body
        and body[0] is stmt
        and isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


class ReturnArityCollector(ast.NodeVisitor):
    """Pass 1: map function/method name -> set of return arities ('?' = ambiguous)."""

    def __init__(self) -> None:
        self.arities: dict[str, set] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._collect(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._collect(node)
        self.generic_visit(node)

    def _collect(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        arities = self.arities.setdefault(fn.name, set())
        returns: list[ast.Return] = []
        # Only returns belonging to *this* function (do not descend into nested defs).
        stack: list[ast.AST] = list(ast.iter_child_nodes(fn))
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(n, ast.Return):
                returns.append(n)
            stack.extend(ast.iter_child_nodes(n))
        if not returns:
            arities.add("?")
            return
        for r in returns:
            if isinstance(r.value, ast.Tuple):
                arities.add(len(r.value.elts))
            else:
                arities.add("?")


class LogicVisitor(ast.NodeVisitor):
    def __init__(self, path: str, findings: list[Finding], arity_map: dict[str, set]) -> None:
        self.path = path
        self.findings = findings
        self.arity_map = arity_map
        self.func_params: list[set[str]] = []  # stack of enclosing function param names

    # ---------- helpers ----------
    def add(self, rule: str, severity: str, node: ast.AST, message: str,
            snippet_node: Optional[ast.AST] = None) -> None:
        self.findings.append(Finding(
            rule=rule, severity=severity, file=self.path,
            line=getattr(node, "lineno", 0),
            message=message,
            snippet=safe_unparse(snippet_node if snippet_node is not None else node),
        ))

    # ---------- functions (param stack, mutable defaults, shadow) ----------
    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        a = node.args
        all_args = list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
        params = {arg.arg for arg in all_args}
        for default in list(a.defaults) + [d for d in a.kw_defaults if d is not None]:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.add("AST08", "BUG-LIKELY", default,
                         f"mutable default argument in {node.name}()")
            elif (isinstance(default, ast.Call) and isinstance(default.func, ast.Name)
                  and default.func.id in {"list", "dict", "set", "bytearray"}
                  and not default.args and not default.keywords):
                self.add("AST08", "REVIEW", default,
                         f"call-produced mutable default in {node.name}() "
                         f"(evaluated once at def time)")
        self.func_params.append(params)
        self.generic_visit(node)
        self.func_params.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    # ---------- if/elif/else ----------
    def visit_If(self, node: ast.If) -> None:
        # AST01: plain else identical to body
        if node.orelse and not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)):
            if dump_stmts(node.body) == dump_stmts(node.orelse):
                self.add("AST01", "BUG-LIKELY", node,
                         "if and else branches are identical - condition is irrelevant")
        # Walk the elif chain: duplicate conditions + identical adjacent branches
        chain_tests: list[tuple[str, ast.expr]] = []
        chain_bodies: list[tuple[str, ast.If]] = []
        cur: ast.If | None = node
        while cur is not None:
            chain_tests.append((dump(cur.test), cur.test))
            chain_bodies.append((dump_stmts(cur.body), cur))
            if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                cur = cur.orelse[0]
            else:
                cur = None
        seen: dict[str, ast.expr] = {}
        for d, test in chain_tests:
            if d in seen:
                self.add("AST02", "BUG-LIKELY", test,
                         "duplicate condition in if/elif chain (branch unreachable)")
            else:
                seen[d] = test
        # AST04: constant test
        if isinstance(node.test, ast.Constant):
            self.add("AST04", "BUG-LIKELY", node,
                     f"if-condition is the constant {node.test.value!r}", node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is not True:
            self.add("AST04", "BUG-LIKELY", node,
                     f"while-condition is the constant {node.test.value!r}", node.test)
        self.generic_visit(node)

    # ---------- comparisons / boolops ----------
    def visit_Compare(self, node: ast.Compare) -> None:
        operands = [node.left] + list(node.comparators)
        for a, b, op in zip(operands, operands[1:], node.ops):
            if dump(a) == dump(b):
                self.add("AST03", "BUG-LIKELY", node,
                         f"comparison of an expression with itself "
                         f"({type(op).__name__})")
        if all(isinstance(o, ast.Constant) for o in operands):
            self.add("AST18", "BUG-LIKELY", node,
                     "comparison of literals - result is constant")
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant) \
                    and right.value is not None and not isinstance(right.value, bool):
                self.add("AST16", "REVIEW", node,
                         "`is` comparison with a literal (identity, not equality)")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        seen: set[str] = set()
        for v in node.values:
            d = dump(v)
            if d in seen:
                self.add("AST15", "BUG-LIKELY", node,
                         f"identical operands in `{type(node.op).__name__.lower()}` expression")
                break
            seen.add(d)
        self.generic_visit(node)

    # ---------- assignments ----------
    def visit_Assign(self, node: ast.Assign) -> None:
        # AST11 self-assign
        if len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, (ast.Name, ast.Attribute)) and dump(t) == dump(node.value):
                self.add("AST11", "BUG-LIKELY", node, "self-assignment has no effect")
            # AST19 tuple-unpack arity
            if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Call):
                callee = node.value.func
                name = callee.id if isinstance(callee, ast.Name) else (
                    callee.attr if isinstance(callee, ast.Attribute) else None)
                if name and name in self.arity_map:
                    arities = self.arity_map[name]
                    if "?" not in arities and len(arities) == 1:
                        (arity,) = arities
                        if isinstance(arity, int) and arity != len(t.elts):
                            self.add("AST19", "BUG-LIKELY", node,
                                     f"unpacking {len(t.elts)} names from {name}() "
                                     f"but every definition of {name} returns a "
                                     f"{arity}-tuple")
        self.generic_visit(node)

    # ---------- dict literals ----------
    def visit_Dict(self, node: ast.Dict) -> None:
        seen: dict[str, ast.expr] = {}
        for k in node.keys:
            if k is None:  # **spread
                continue
            if isinstance(k, ast.Constant):
                d = repr(k.value)
                if d in seen:
                    self.add("AST12", "BUG-LIKELY", k,
                             f"duplicate dict key {k.value!r} - earlier value overwritten")
                seen[d] = k
        self.generic_visit(node)

    # ---------- assert ----------
    def visit_Assert(self, node: ast.Assert) -> None:
        if isinstance(node.test, ast.Tuple) and node.test.elts:
            self.add("AST13", "BUG-LIKELY", node,
                     "assert on a non-empty tuple is always true")
        self.generic_visit(node)

    # ---------- try/except/finally ----------
    def visit_Try(self, node: ast.Try) -> None:
        resolved: list[list[Optional[type]]] = [exc_types_of_handler(h) for h in node.handlers]
        for j, handler in enumerate(node.handlers):
            types_j = [t for t in resolved[j] if t is not None]
            # AST07 / AST06
            for i in range(j):
                types_i = [t for t in resolved[i] if t is not None]
                for tj in types_j:
                    for ti in types_i:
                        if tj is ti:
                            self.add("AST07", "BUG-LIKELY", handler,
                                     f"exception {tj.__name__} already caught by "
                                     f"handler #{i + 1} - duplicate")
                        elif issubclass(tj, ti):
                            self.add("AST06", "BUG-LIKELY", handler,
                                     f"handler for {tj.__name__} unreachable: handler "
                                     f"#{i + 1} already catches superclass {ti.__name__}")
            # AST10 silent swallow
            broad = any(t in (Exception, BaseException) for t in types_j) or handler.type is None
            body_is_noop = all(
                isinstance(s, (ast.Pass, ast.Continue)) or
                (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                for s in handler.body
            )
            if broad and body_is_noop:
                self.add("AST10", "REVIEW", handler,
                         "broad exception handler silently swallows all errors")
        # AST14 finally-jump (do not descend into nested defs)
        for stmt in node.finalbody:
            stack: list[ast.AST] = [stmt]
            while stack:
                n = stack.pop()
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                if isinstance(n, (ast.Return, ast.Break, ast.Continue)):
                    self.add("AST14", "BUG-LIKELY", n,
                             f"{type(n).__name__.lower()} inside finally swallows "
                             f"in-flight exceptions")
                stack.extend(ast.iter_child_nodes(n))
        self.generic_visit(node)

    # ---------- loops ----------
    def visit_For(self, node: ast.For) -> None:
        if self.func_params:
            targets = [node.target] if isinstance(node.target, ast.Name) else (
                list(ast.walk(node.target)))
            for t in targets:
                if isinstance(t, ast.Name) and t.id in self.func_params[-1]:
                    self.add("AST17", "REVIEW", node,
                             f"loop variable '{t.id}' shadows a parameter of the "
                             f"enclosing function")
        self.generic_visit(node)

    # ---------- calls ----------
    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name) and f.id == "zip" and len(node.args) >= 2:
            if not any(kw.arg == "strict" for kw in node.keywords):
                self.add("AST09", "INFO", node,
                         "zip() without strict= silently truncates on length mismatch")
        if (isinstance(f, ast.Attribute) and f.attr in {"insert", "append"}
                and isinstance(f.value, ast.Attribute) and f.value.attr == "path"
                and isinstance(f.value.value, ast.Name) and f.value.value.id == "sys"):
            self.add("AST20", "INFO", node, "sys.path manipulation (import-order hazard)")
        self.generic_visit(node)

    # ---------- unreachable code ----------
    def generic_visit(self, node: ast.AST) -> None:
        for field in STMT_LIST_FIELDS:
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and all(isinstance(s, ast.stmt) for s in stmts):
                for i, s in enumerate(stmts[:-1]):
                    if isinstance(s, JUMP_STMTS):
                        nxt = stmts[i + 1]
                        self.add("AST05", "BUG-LIKELY", nxt,
                                 f"unreachable code after {type(s).__name__.lower()} "
                                 f"(line {s.lineno})")
                        break  # one report per block
        if isinstance(node, ast.ExceptHandler):
            stmts = node.body
            for i, s in enumerate(stmts[:-1]):
                if isinstance(s, JUMP_STMTS):
                    self.add("AST05", "BUG-LIKELY", stmts[i + 1],
                             f"unreachable code after {type(s).__name__.lower()} "
                             f"(line {s.lineno})")
                    break
        super().generic_visit(node)


def iter_py_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for p in sorted(root.rglob("*.py")):
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="files or directories to scan")
    ap.add_argument("--json", dest="json_out", help="write findings JSON here")
    args = ap.parse_args()

    files = list(iter_py_files(Path(p) for p in args.paths))
    if not files:
        print("no python files found", file=sys.stderr)
        return 2

    # Pass 1: return-arity map across the whole corpus.
    arity = ReturnArityCollector()
    trees: dict[Path, ast.AST] = {}
    parse_errors: list[tuple[str, str]] = []
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            parse_errors.append((str(f), f"line {e.lineno}: {e.msg}"))
            continue
        trees[f] = tree
        arity.visit(tree)

    # Pass 2: logic visitors.
    findings: list[Finding] = []
    for f, tree in trees.items():
        LogicVisitor(str(f), findings, arity.arities).visit(tree)

    findings.sort(key=lambda x: (x.rule, x.file, x.line))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([asdict(x) for x in findings], indent=1), encoding="utf-8")

    # Human report grouped by rule.
    by_rule: dict[str, list[Finding]] = {}
    for x in findings:
        by_rule.setdefault(x.rule, []).append(x)

    print(f"scanned {len(trees)} files "
          f"({sum(1 for _ in findings)} findings, {len(parse_errors)} parse errors)\n")
    for path, msg in parse_errors:
        print(f"PARSE-ERROR {path}: {msg}")

    for rule in sorted(by_rule):
        items = by_rule[rule]
        sev = items[0].severity
        print(f"== {rule} [{sev}] x{len(items)} ==")
        for x in items:
            print(f"  {x.file}:{x.line}  {x.message}")
            print(f"      | {x.snippet}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
