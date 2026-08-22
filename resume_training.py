#!/usr/bin/env python3
"""
Resume Training Script for VIN OCR

Composes and launches a fine-tuning run that RESUMES from a checkpoint,
verifying BEFORE launch that the resume state (epoch/global_step) actually
exists next to the weights.

Previously broken in two ways (audit H6):
  - it passed `best_accuracy.pdparams` WITH the extension, the trainer
    derived `best_accuracy.pdparams_info.json` (never written), and the
    epoch counter silently reset to 0 while this script printed
    "RESUME ENFORCED";
  - when --checkpoint didn't exist after a BEST fallback had already been
    appended, it ran `cmd.remove(args.checkpoint)` on a path that was never
    in cmd -> ValueError.
Each branch now composes the command completely (nothing is ever removed),
the checkpoint is passed in the extensionless form, and a missing resume
state is a loud sys.exit instead of a silent fresh start.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

#: The trainer's best-by-exact-match checkpoint, extensionless: the form
#: finetune_paddleocr's --resume expects (it derives .pdparams / .pdopt /
#: _info.json itself).
BEST_CHECKPOINT = "output/vin_rec_finetune/best_accuracy"


def _strip_pdparams(path: str) -> str:
    """Normalise a checkpoint argument to the extensionless form."""
    if path.endswith(".pdparams"):
        return path[:-len(".pdparams")]
    return path


def _checkpoint_exists(base: str) -> bool:
    """A checkpoint 'exists' when its weights file exists."""
    return (project_root / (base + ".pdparams")).exists()


def _select_checkpoint(args) -> Optional[str]:
    """
    Choose the checkpoint to resume from (extensionless, relative to the
    project root), or None for a fresh start. Precedence:

    1. --force-resume: the BEST checkpoint, falling back to --checkpoint.
    2. --checkpoint: the given one, falling back to the BEST checkpoint.
    3. Neither: the BEST checkpoint when present.
    """
    requested = _strip_pdparams(args.checkpoint) if args.checkpoint else None

    if args.force_resume:
        if _checkpoint_exists(BEST_CHECKPOINT):
            print(f"🎯 FORCE RESUME: using best checkpoint: {BEST_CHECKPOINT}")
            return BEST_CHECKPOINT
        print(f"❌ Best checkpoint not found: {BEST_CHECKPOINT}.pdparams")
        if requested and _checkpoint_exists(requested):
            print(f"💡 Falling back to specified checkpoint: {requested}")
            return requested
        print("💡 No usable checkpoint found")
        return None

    if requested:
        if _checkpoint_exists(requested):
            print(f"✅ Using specified checkpoint: {requested}")
            return requested
        print(f"❌ Specified checkpoint not found: {requested}.pdparams")
        if _checkpoint_exists(BEST_CHECKPOINT):
            print(f"💡 Falling back to best checkpoint: {BEST_CHECKPOINT}")
            return BEST_CHECKPOINT
        print("💡 No usable checkpoint found")
        return None

    if _checkpoint_exists(BEST_CHECKPOINT):
        print(f"✅ Auto-detected best checkpoint: {BEST_CHECKPOINT}")
        return BEST_CHECKPOINT
    print("❌ No best checkpoint found")
    return None


def _verify_resume_state(checkpoint: str) -> dict:
    """
    Verify the resume state the trainer will restore, BEFORE launching.

    The trainer's load_checkpoint derives `<checkpoint>_info.json` and
    restores epoch/global_step/best-metric baselines from it (and reports
    what it restored in its return value). If that file is missing here, the
    run would load weights but silently restart at epoch 0 while this script
    claims a resume - so its absence is fatal.

    Returns:
        The parsed resume info (epoch, global_step, best_accuracy, ...).
    """
    info_path = project_root / (checkpoint + "_info.json")
    if not info_path.exists():
        print(f"❌ RESUME STATE MISSING: {info_path}")
        print("   The trainer would load the weights but reset epoch/")
        print("   global_step/best-metric baselines to fresh-run values.")
        print("   For a weights-only warm start, call the trainer directly:")
        print(f"   {sys.executable} src/vin_ocr/training/finetune_paddleocr.py "
              f"--resume {checkpoint}")
        sys.exit(1)
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"❌ RESUME STATE UNREADABLE: {info_path}: {exc}")
        sys.exit(1)
    if "epoch" not in info or "global_step" not in info:
        print(f"❌ RESUME STATE INCOMPLETE: {info_path} lacks epoch/global_step")
        sys.exit(1)
    print(f"🔄 RESUME VERIFIED: epoch={info['epoch']} "
          f"global_step={info['global_step']} "
          f"best_accuracy={info.get('best_accuracy', 0.0):.4f}")
    return info


def main():
    """Main resume training function."""

    print("🚀 Resuming VIN OCR Training")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser(description="Resume VIN OCR Training")
    parser.add_argument("--config", default="configs/vin_finetune_config.yml",
                        help="Configuration file path (relative to project root)")
    parser.add_argument("--checkpoint",
                        help="Checkpoint path, with or without .pdparams "
                             "(relative to project root)")
    parser.add_argument("--force-resume", action="store_true",
                        help="Resume from the best checkpoint, falling back "
                             "to --checkpoint if the best one is absent")

    args = parser.parse_args()

    print("📋 Configuration:")
    print(f"   Config: {args.config}")
    print(f"   Force resume: {args.force_resume}")
    print("   All other settings from config file")

    # Config must exist before anything else runs.
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print(f"💡 Current project root: {project_root}")
        configs_dir = project_root / "configs"
        if configs_dir.exists():
            print("💡 Available config files:")
            for candidate in sorted(configs_dir.glob("*.yml")):
                print(f"   - configs/{candidate.name}")
        return 1

    checkpoint_to_use = _select_checkpoint(args)

    cmd = [
        sys.executable, "src/vin_ocr/training/finetune_paddleocr.py",
        "--config", args.config,
    ]
    if checkpoint_to_use:
        _verify_resume_state(checkpoint_to_use)
        cmd.extend(["--resume", checkpoint_to_use])
    else:
        print("⚠️  NO CHECKPOINT: starting fresh training")

    print("\n🎯 Final Command:")
    print(f"   {' '.join(cmd)}")
    print(f"🎯 Resume Status: {'✅ Will resume' if checkpoint_to_use else '⚠️ Fresh start'}")
    print("\n🏃 Starting training...")

    # cwd=project_root is the single working-directory decision; the old
    # os.chdir(training_dir) directly above it was dead and contradictory.
    result = subprocess.run(cmd, cwd=project_root, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Training failed with exit code: {result.returncode}")
        return result.returncode

    print("\n🎉 Training completed successfully!")

    # Show results - extract output path from config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    output_dir = config.get('Global', {}).get('save_model_dir', 'output/vin_rec_finetune')
    metrics_file = project_root / f"{output_dir}/training_metrics.json"

    if metrics_file.exists():
        print(f"📊 Results saved to: {metrics_file}")

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        exact_match = metrics.get('evaluation_metrics', {}).get('image_level', {}).get('exact_match_accuracy', 0)
        char_accuracy = metrics.get('evaluation_metrics', {}).get('character_level', {}).get('character_accuracy', 0)

        print("\n📈 Final Results:")
        print(f"   Exact Match Accuracy: {exact_match:.2%}")
        print(f"   Character Accuracy: {char_accuracy:.2%}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
