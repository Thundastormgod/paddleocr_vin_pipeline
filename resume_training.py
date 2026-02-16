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
    
    # Configuration - make these configurable via command line args
    import argparse
    parser = argparse.ArgumentParser(description="Resume VIN OCR Training")
    parser.add_argument("--config", default="configs/vin_finetune_config.yml", 
                       help="Configuration file path (relative to project root)")
    parser.add_argument("--checkpoint", default="src/vin_ocr/training/output/vin_rec_finetune/best_accuracy.pdparams",
                       help="Checkpoint file path (relative to project root)")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Total epochs (including previous)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0015,
                       help="Learning rate")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                       help="Device to use (cpu/gpu)")
    
    args = parser.parse_args()
    
    print(f"📋 Configuration:")
    print(f"   Config: {args.config}")
    print(f"   Resume from: {args.checkpoint}")
    print(f"   Total epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Device: {args.device.upper()}")
    
    # Build command
    cmd = [
        "python", "src/vin_ocr/training/finetune_paddleocr.py",
        "--config", args.config,
        "--resume", args.checkpoint,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        f"--{args.device}"
    ]
    
    print(f"\n🎯 Command to run:")
    print(f"   {' '.join(cmd)}")
    
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
    checkpoint_file = project_root / args.checkpoint
    if checkpoint_file.exists():
        print(f"✅ Checkpoint found: {checkpoint_file}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_file}")
        print("   💡 Suggestion: Make sure the path is relative to project root")
        print(f"💡 Current project root: {project_root}")
        print(f"💡 Looking for: {args.checkpoint}")
        print("   Starting fresh training instead...")
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
