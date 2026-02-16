#!/usr/bin/env python3
"""
DagsHub Streaming Usage Examples for VIN OCR

This script demonstrates how to use DagsHub streaming with VIN OCR training.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def example_basic_streaming():
    """Example: Basic streaming usage."""
    print("🚀 Example 1: Basic Streaming Training")
    print("-" * 50)
    
    cmd = """python src/vin_ocr/training/finetune_paddleocr.py \\
    --config configs/vin_finetune_config.yml \\
    --stream \\
    --dagshub-user YOUR_USERNAME \\
    --dagshub-token YOUR_TOKEN \\
    --cpu \\
    --epochs 5 \\
    --batch-size 16"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("✅ Enable DagsHub streaming")
    print("✅ Use streaming config automatically")
    print("✅ Train without downloading data locally")
    print("✅ Save models to local output directory")

def example_resume_streaming():
    """Example: Resume training with streaming."""
    print("\n🔄 Example 2: Resume Training with Streaming")
    print("-" * 50)
    
    cmd = """python src/vin_ocr/training/finetune_paddleocr.py \\
    --config configs/vin_finetune_config.yml \\
    --stream \\
    --dagshub-user YOUR_USERNAME \\
    --dagshub-token YOUR_TOKEN \\
    --resume output/vin_rec_finetune/latest \\
    --epochs 15 \\
    --batch-size 16"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("✅ Resume from existing checkpoint")
    print("✅ Use streaming for data access")
    print("✅ Continue training to 15 epochs total")

def example_gpu_streaming():
    """Example: GPU training with streaming."""
    print("\n⚡ Example 3: GPU Training with Streaming")
    print("-" * 50)
    
    cmd = """python src/vin_ocr/training/finetune_paddleocr.py \\
    --config configs/vin_finetune_config.yml \\
    --stream \\
    --dagshub-user YOUR_USERNAME \\
    --dagshub-token YOUR_TOKEN \\
    --epochs 20 \\
    --batch-size 32 \\
    --lr 0.002"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("✅ Use GPU for faster training")
    print("✅ Stream data directly")
    print("✅ Use larger batch size")
    print("✅ Higher learning rate")

def example_custom_config():
    """Example: Custom config with streaming."""
    print("\n⚙️ Example 4: Custom Config with Streaming")
    print("-" * 50)
    
    cmd = """python src/vin_ocr/training/finetune_paddleocr.py \\
    --config configs/custom_vin_config.yml \\
    --stream \\
    --dagshub-user YOUR_USERNAME \\
    --dagshub-token YOUR_TOKEN \\
    --architecture PP-OCRv5 \\
    --epochs 10 \\
    --output output/custom_vin_model"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("✅ Use custom configuration")
    print("✅ Stream with PP-OCRv5 architecture")
    print("✅ Save to custom output directory")

def example_environment_variables():
    """Example: Using environment variables."""
    print("\n🔐 Example 5: Environment Variables")
    print("-" * 50)
    
    print("Set environment variables:")
    print("export DAGSHUB_USERNAME='your_username'")
    print("export DAGSHUB_TOKEN='your_access_token'")
    print()
    
    cmd = """python src/vin_ocr/training/finetune_paddleocr.py \\
    --config configs/vin_finetune_config.yml \\
    --stream \\
    --epochs 10"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("✅ Use env vars for authentication")
    print("✅ No need to pass credentials in command")

def example_data_management():
    """Example: Data management commands."""
    print("\n📊 Example 6: Data Management")
    print("-" * 50)
    
    print("Track new data:")
    print("dvc add finetune_data/new_images")
    print()
    print("Push data to DagsHub:")
    print("dvc push")
    print()
    print("Pull data locally:")
    print("dvc pull")
    print()
    print("Check data status:")
    print("dvc status")

def example_troubleshooting():
    """Example: Troubleshooting commands."""
    print("\n🔧 Example 7: Troubleshooting")
    print("-" * 50)
    
    print("Test DagsHub setup:")
    print("python test_dagshub_setup.py")
    print()
    print("Check DVC remotes:")
    print("dvc remote list")
    print()
    print("Check streaming status:")
    print("dvc dagshub info")
    print()
    print("Debug streaming:")
    export_cmd = "export DAGSHUB_DEBUG=true"
    print(f"{export_cmd}")
    print("python src/vin_ocr/training/finetune_paddleocr.py --stream ...")

def main():
    """Show all examples."""
    print("🎯 DagsHub Streaming Usage Examples")
    print("=" * 60)
    
    example_basic_streaming()
    example_resume_streaming()
    example_gpu_streaming()
    example_custom_config()
    example_environment_variables()
    example_data_management()
    example_troubleshooting()
    
    print("\n📋 Quick Start Guide")
    print("=" * 30)
    print("1. Test setup: python test_dagshub_setup.py")
    print("2. Basic training: See Example 1")
    print("3. Check results: output/vin_rec_finetune/training_metrics.json")
    print("4. Resume training: See Example 2")
    
    print("\n🔗 Important Notes")
    print("=" * 20)
    print("• Streaming requires internet connection")
    print("• Models are saved locally (not streamed)")
    print("• Data is cached automatically")
    print("• Use --cpu flag if no GPU available")
    print("• Authentication via token or env vars")

if __name__ == '__main__':
    main()
