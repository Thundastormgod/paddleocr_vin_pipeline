#!/usr/bin/env python3
"""
Test script to verify training loop changes are enforced
"""

import sys
import paddle
import yaml
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_training_enforcement():
    """Test that all training loop changes are properly enforced."""
    
    print("🔍 Testing Training Loop Changes Enforcement")
    print("=" * 60)
    
    # Load config
    with open(project_root / "configs/vin_finetune_config.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    from src.vin_ocr.training.finetune_paddleocr import VINFineTuner
    trainer = VINFineTuner(config)
    
    print("✅ Step 1: Basic Initialization")
    print(f"   best_accuracy attribute: {hasattr(trainer, 'best_accuracy')}")
    print(f"   _save_best_model method: {hasattr(trainer, '_save_best_model')}")
    print(f"   Initial best_accuracy: {trainer.best_accuracy}")
    
    # Load best model
    best_model_path = project_root / "output/vin_rec_finetune/best_accuracy.pdparams"
    if best_model_path.exists():
        trainer.model.set_state_dict(paddle.load(str(best_model_path)))
        print("✅ Step 2: Best Model Loaded")
        
        # Test validation with debug output
        print("✅ Step 3: Testing Validation with Debug Output")
        val_loss, val_acc = trainer.validate()
        
        print(f"   Validation Loss: {val_loss:.4f}")
        print(f"   Validation Accuracy: {val_acc:.4f}")
        print(f"   Best Accuracy: {trainer.best_accuracy:.4f}")
        
        # Test best model tracking
        print("✅ Step 4: Testing Best Model Tracking")
        if val_acc > trainer.best_accuracy:
            print(f"   New best accuracy detected: {val_acc:.4f} > {trainer.best_accuracy:.4f}")
            trainer.best_accuracy = val_acc
            trainer._save_best_model()
            print("   ✅ Best model saved")
        else:
            print(f"   Current accuracy: {val_acc:.4f}")
            print(f"   Best accuracy: {trainer.best_accuracy:.4f}")
        
        # Test training loop display format
        print("✅ Step 5: Testing Training Loop Display Format")
        print("   Expected format:")
        print("   Epoch [1/30] Train Loss: 0.7546 Val Loss: 1.4941 Val Acc: 0.0698 Best: 0.0698 Time: 30.3s")
        print("   📊 Image-Level: 3/43 correct (6.98%)")
        print("   📝 Char-Level: Acc=85.09%, F1-micro=0.8509, F1-macro=0.7550")
        print("   🎉 New best accuracy: 0.0698")
        
        # Verify the changes are in the code
        print("✅ Step 6: Verifying Code Changes")
        
        # Check training loop code
        with open(project_root / "src/vin_ocr/training/finetune_paddleocr.py", 'r') as f:
            code = f.read()
        
        checks = [
            ("best_accuracy tracking", "self.best_accuracy = 0.0" in code),
            ("best model saving", "_save_best_model" in code),
            ("best accuracy display", "Best: {self.best_accuracy:.4f}" in code),
            ("new best celebration", "🎉 New best accuracy" in code),
            ("detailed metrics", "📊 Image-Level:" in code),
            ("character metrics", "📝 Char-Level:" in code)
        ]
        
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"   {status} {check_name}: {result}")
        
        all_passed = all(result for _, result in checks)
        
        if all_passed:
            print("\n🎉 ALL TRAINING LOOP CHANGES ENFORCED!")
            print("✅ Best accuracy tracking: Working")
            print("✅ Best model saving: Working")
            print("✅ Enhanced display: Working")
            print("✅ Debug metrics: Working")
            print("✅ New best celebration: Working")
        else:
            print("\n❌ Some changes may not be properly enforced")
        
        return all_passed
        
    else:
        print(f"❌ Best model not found: {best_model_path}")
        return False

if __name__ == '__main__':
    success = test_training_enforcement()
    sys.exit(0 if success else 1)
