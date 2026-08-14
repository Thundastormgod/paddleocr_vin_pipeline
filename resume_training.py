#!/usr/bin/env python3
"""
Resume Training Script for VIN OCR

This script resumes training with the fixed configuration
and proper CTC/CrossEntropyLoss setup.
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main resume training function."""
    
    print("🚀 Resuming VIN OCR Training with Fixed Configuration")
    print("=" * 60)
    
    # Only essential arguments - no config overrides
    import argparse
    parser = argparse.ArgumentParser(description="Resume VIN OCR Training")
    parser.add_argument("--config", default="configs/vin_finetune_config.yml", 
                       help="Configuration file path (relative to project root)")
    parser.add_argument("--checkpoint", 
                       help="Specific checkpoint file path (relative to project root)")
    parser.add_argument("--force-resume", action="store_true",
                       help="Force resume from best checkpoint even if not found")
    
    args = parser.parse_args()
    
    print(f"📋 Configuration:")
    print(f"   Config: {args.config}")
    print(f"   Force resume: {args.force_resume}")
    print(f"   All other settings from config file")
    
    # Build command - only config and resume, no CLI overrides
    cmd = [
        "python", "src/vin_ocr/training/finetune_paddleocr.py",
        "--config", args.config
    ]
    
    # ENFORCED: Always try to use best checkpoint first
    best_checkpoint = project_root / "output/vin_rec_finetune/best_accuracy.pdparams"
    checkpoint_to_use = None
    
    # Priority 1: Force resume from best checkpoint
    if args.force_resume:
        if best_checkpoint.exists():
            checkpoint_to_use = str(best_checkpoint.relative_to(project_root))
            print(f"🎯 FORCE RESUME: Using best checkpoint: {checkpoint_to_use}")
            print(f"📊 This checkpoint achieved 44.19% exact match accuracy")
        else:
            print(f"❌ Best checkpoint not found: {best_checkpoint}")
            if args.checkpoint:
                print("💡 Falling back to specified checkpoint...")
            else:
                print("💡 No fallback checkpoint available")
    
    # Priority 2: Use specified checkpoint
    elif args.checkpoint:
        checkpoint_file = project_root / args.checkpoint
        if checkpoint_file.exists():
            checkpoint_to_use = args.checkpoint
            print(f"✅ Using specified checkpoint: {checkpoint_to_use}")
        else:
            print(f"❌ Specified checkpoint not found: {checkpoint_file}")
            print("💡 Falling back to best checkpoint...")
            if best_checkpoint.exists():
                checkpoint_to_use = str(best_checkpoint.relative_to(project_root))
                print(f"🎯 Using best checkpoint: {checkpoint_to_use}")
    
    # Priority 3: Auto-detect best checkpoint
    else:
        if best_checkpoint.exists():
            checkpoint_to_use = str(best_checkpoint.relative_to(project_root))
            print(f"✅ Auto-detected best checkpoint: {checkpoint_to_use}")
            print(f"📊 This checkpoint achieved 44.19% exact match accuracy")
        else:
            print("❌ No best checkpoint found")
            print("💡 Starting fresh training...")
    
    # ENFORCEMENT: Add resume flag if we have a checkpoint
    if checkpoint_to_use:
        cmd.extend(["--resume", checkpoint_to_use])
        print(f"🔄 RESUME ENFORCED: Will resume from {checkpoint_to_use}")
    else:
        print("⚠️  NO CHECKPOINT: Starting fresh training")
    
    # ENFORCEMENT: Validate checkpoint exists before proceeding
    if checkpoint_to_use:
        checkpoint_path = project_root / checkpoint_to_use
        if not checkpoint_path.exists():
            print(f"❌ CRITICAL: Checkpoint validation failed: {checkpoint_path}")
            print("💡 This should not happen - checkpoint was detected but not found!")
            print("🔄 Falling back to fresh training...")
            # Remove resume from command to prevent errors
            if "--resume" in cmd:
                resume_index = cmd.index("--resume")
                cmd.pop(resume_index)  # Remove --resume
                cmd.pop(resume_index)  # Remove checkpoint path
            checkpoint_to_use = None
        else:
            print(f"✅ Checkpoint validation passed: {checkpoint_path}")
            print(f"📁 File size: {checkpoint_path.stat().st_size:,} bytes")
    
    print(f"\n🎯 Final Command:")
    print(f"   {' '.join(cmd)}")
    print(f"🎯 Resume Status: {'✅ Will resume' if checkpoint_to_use else '⚠️ Fresh start'}")
    
    # Check if config file exists
    config_file = project_root / args.config
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        print(f"💡 Current project root: {project_root}")
        print(f"💡 Looking for: {args.config}")
        print(f"💡 Available config files:")
        configs_dir = project_root / "configs"
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.yml"):
                print(f"   - configs/{config_file.name}")
        return 1
    
    # Check if checkpoint exists
    if args.checkpoint:
        checkpoint_file = project_root / args.checkpoint
        if checkpoint_file.exists():
            print(f"✅ Checkpoint found: {checkpoint_file}")
        else:
            print(f"❌ Checkpoint not found: {checkpoint_file}")
            print("   💡 Suggestion: Make sure the path is relative to project root")
            print(f"💡 Current project root: {project_root}")
            print(f"💡 Looking for: {args.checkpoint}")
            print("   Starting fresh training instead...")
            # Remove resume arguments
            cmd.remove("--resume")
            cmd.remove(args.checkpoint)
    
    print(f"\n🏃 Starting training...")
    
    # Change to training directory and run
    training_dir = project_root / "src/vin_ocr/training"
    os.chdir(training_dir)
    
    # Run training
    import subprocess
    result = subprocess.run(cmd, cwd=project_root, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n🎉 Training completed successfully!")
        
        # Show results - extract output path from config
        config_file = project_root / args.config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        output_dir = config.get('Global', {}).get('save_model_dir', 'output/vin_rec_finetune')
        metrics_file = project_root / f"{output_dir}/training_metrics.json"
        
        if metrics_file.exists():
            print(f"📊 Results saved to: {metrics_file}")
            
            # Quick summary
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            exact_match = metrics.get('evaluation_metrics', {}).get('image_level', {}).get('exact_match_accuracy', 0)
            char_accuracy = metrics.get('evaluation_metrics', {}).get('character_level', {}).get('character_accuracy', 0)
            
            print(f"\n📈 Final Results:")
            print(f"   Exact Match Accuracy: {exact_match:.2%}")
            print(f"   Character Accuracy: {char_accuracy:.2%}")
            
    else:
        print(f"\n❌ Training failed with exit code: {result.returncode}")
        return result.returncode
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
