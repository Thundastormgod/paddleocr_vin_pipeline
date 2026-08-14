#!/usr/bin/env python3
"""
VIN Training CLI - `vin-train`
==============================

Dispatcher over the training entry points in this package. Each subcommand
forwards all remaining arguments to the underlying module's own argparse
parser, so `vin-train finetune --help` shows the real options.

Usage:
    vin-train finetune  [args...]   Fine-tune PaddleOCR recognition model
    vin-train scratch   [args...]   Train a recognition model from scratch
    vin-train deepseek  [args...]   Fine-tune DeepSeek-OCR (LoRA/QLoRA)
    vin-train tune      [args...]   Optuna hyperparameter search
    vin-train export    [args...]   Export a trained Paddle model to ONNX
    vin-train export-deepseek [args...]  Export DeepSeek model to ONNX
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import List, Optional


# subcommand -> (module path, human description)
_COMMANDS = {
    "finetune": (
        "src.vin_ocr.training.finetune_paddleocr",
        "Fine-tune the PaddleOCR recognition model on VIN data",
    ),
    "scratch": (
        "src.vin_ocr.training.train_from_scratch",
        "Train a VIN recognition model from randomly initialised weights",
    ),
    "deepseek": (
        "src.vin_ocr.training.finetune_deepseek",
        "Fine-tune the DeepSeek-OCR vision-language model (LoRA/QLoRA)",
    ),
    "tune": (
        "src.vin_ocr.training.hyperparameter_tuning.optuna_tuning",
        "Run an Optuna hyperparameter search",
    ),
    "export": (
        "src.vin_ocr.training.export_onnx",
        "Export a trained Paddle model to ONNX",
    ),
    "export-deepseek": (
        "src.vin_ocr.training.export_deepseek_onnx",
        "Export a fine-tuned DeepSeek model to ONNX",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vin-train",
        description="VIN OCR training commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Arguments after the subcommand are passed straight through, e.g.\n"
            "  vin-train finetune --config configs/vin_finetune_config.yml --epochs 30\n"
            "  vin-train finetune --help\n"
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, description) in _COMMANDS.items():
        # add_help=False so `vin-train finetune --help` reaches the target module
        sub.add_parser(name, help=description, add_help=False)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the `vin-train` console script."""
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
        print(
            f"error: could not import {module_path}: {exc}\n"
            f"Install the training extras with:  pip install -e '.[training]'",
            file=sys.stderr,
        )
        return 1

    target_main = getattr(module, "main", None)
    if target_main is None:
        print(f"error: {module_path} defines no main()", file=sys.stderr)
        return 1

    # Rewrite argv so the target module's argparse sees a sensible prog name
    saved_argv = sys.argv
    sys.argv = [f"vin-train {command}", *rest]
    try:
        result = target_main()
    finally:
        sys.argv = saved_argv

    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
