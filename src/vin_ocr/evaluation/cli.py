#!/usr/bin/env python3
"""
VIN Evaluation CLI - `vin-evaluate`
===================================

Dispatcher over the evaluation entry points in this package. Each subcommand
forwards all remaining arguments to the underlying module's own argparse
parser, so `vin-evaluate single --help` shows the real options.

Usage:
    vin-evaluate single [args...]   Evaluate one model, with train/val/test splits
    vin-evaluate multi  [args...]   Compare multiple OCR models on the same dataset
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import List, Optional


# subcommand -> (module path, human description)
_COMMANDS = {
    "single": (
        "src.vin_ocr.evaluation.evaluate",
        "Evaluate a single model (exact match, F1, CER, NED, per-position)",
    ),
    "multi": (
        "src.vin_ocr.evaluation.multi_model_evaluation",
        "Compare multiple OCR models side by side on one dataset",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vin-evaluate",
        description="VIN OCR evaluation commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Arguments after the subcommand are passed straight through, e.g.\n"
            "  vin-evaluate single --data-dir ./data --split test --output results.json\n"
            "  vin-evaluate multi --max-images 100\n"
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, description) in _COMMANDS.items():
        # add_help=False so `vin-evaluate single --help` reaches the target module
        sub.add_parser(name, help=description, add_help=False)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the `vin-evaluate` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if argv[0] == "--version":
        parser.parse_args(argv)
        return 0

    command, rest = argv[0], argv[1:]
    if command not in _COMMANDS:
        parser.print_help(sys.stderr)
        print(f"\nerror: unknown command {command!r}", file=sys.stderr)
        return 2

    module_path, _ = _COMMANDS[command]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        print(f"error: could not import {module_path}: {exc}", file=sys.stderr)
        return 1

    target_main = getattr(module, "main", None)
    if target_main is None:
        print(f"error: {module_path} defines no main()", file=sys.stderr)
        return 1

    saved_argv = sys.argv
    sys.argv = [f"vin-evaluate {command}", *rest]
    try:
        result = target_main()
    finally:
        sys.argv = saved_argv

    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
