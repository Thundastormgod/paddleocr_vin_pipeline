#!/usr/bin/env python3
"""
Test to understand the training vs final evaluation discrepancy
"""

import sys
import paddle
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_training_vs_final():
    """Test the difference between training validation and final evaluation."""
    
    print("🔍 Testing Training vs Final Evaluation")
    print("=" * 50)
    
    # Load config
    with open(project_root / "configs/vin_finetune_config.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    from src.vin_ocr.training.finetune_paddleocr import VINFineTuner
    trainer = VINFineTuner(config)
    
    # Check if we should load best model
    best_model_path = project_root / "output/vin_rec_finetune/best_accuracy.pdparams"
    latest_model_path = project_root / "output/vin_rec_finetune/latest.pdparams"
    
    print(f"📁 Model files:")
    print(f"   Best model: {best_model_path.exists()}")
    print(f"   Latest model: {latest_model_path.exists()}")
    
    # Test 1: Current model state (what training loop sees)
    print(f"\n🔍 Test 1: Current model state (training loop view)")
    val_loss_1, val_acc_1 = trainer.validate()
    print(f"   Val Loss: {val_loss_1:.4f}")
    print(f"   Val Acc: {val_acc_1:.4f}")
    
    # Test 2: Load best model (what final evaluation sees)
    if best_model_path.exists():
        print(f"\n🔍 Test 2: Best saved model (final evaluation view)")
        trainer.model.set_state_dict(paddle.load(str(best_model_path)))
        val_loss_2, val_acc_2 = trainer.validate()
        print(f"   Val Loss: {val_loss_2:.4f}")
        print(f"   Val Acc: {val_acc_2:.4f}")
        
        # Compare
        print(f"\n📊 Comparison:")
        print(f"   Training loop: {val_acc_1:.4f}")
        print(f"   Final eval:    {val_acc_2:.4f}")
        print(f"   Difference:   {abs(val_acc_2 - val_acc_1):.4f}")
        
        if val_acc_2 > val_acc_1:
            print(f"   ✅ Final evaluation uses better model (best saved)")
        else:
            print(f"   ❌ Models should be the same")
    
    # Test 3: Load latest model
    if latest_model_path.exists():
        print(f"\n🔍 Test 3: Latest saved model")
        trainer.model.set_state_dict(paddle.load(str(latest_model_path)))
        val_loss_3, val_acc_3 = trainer.validate()
        print(f"   Val Loss: {val_loss_3:.4f}")
        print(f"   Val Acc: {val_acc_3:.4f}")

if __name__ == '__main__':
    test_training_vs_final()
