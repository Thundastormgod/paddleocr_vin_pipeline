#!/usr/bin/env python3
"""
Find the best epoch with highest accuracy from saved checkpoints
"""

import sys
import paddle
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def find_best_epoch():
    """Test all saved epochs to find the best performing one."""
    
    print("🔍 Finding Best Epoch Performance")
    print("=" * 50)
    
    # Load config
    with open(project_root / "configs/vin_finetune_config.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    from src.vin_ocr.training.finetune_paddleocr import VINFineTuner
    trainer = VINFineTuner(config)
    
    # Find all epoch checkpoints
    checkpoint_dir = project_root / "output/vin_rec_finetune"
    epoch_files = list(checkpoint_dir.glob("epoch_*.pdparams"))
    epoch_files.sort(key=lambda x: int(x.stem.split('_')[1]))
    
    print(f"📁 Found {len(epoch_files)} epoch checkpoints")
    
    best_epoch = None
    best_accuracy = 0.0
    best_correct = 0
    results = []
    
    for epoch_file in epoch_files:
        epoch_num = int(epoch_file.stem.split('_')[1])
        
        try:
            # Load model weights
            trainer.model.set_state_dict(paddle.load(str(epoch_file)))
            
            # Validate
            val_loss, val_acc = trainer.validate()
            
            # Get detailed metrics
            metrics = trainer.get_last_validation_metrics()
            correct = metrics.get('correct_images', 0)
            total = metrics.get('total_images', 0)
            char_acc = metrics.get('char_accuracy', 0)
            
            result = {
                'epoch': epoch_num,
                'accuracy': val_acc,
                'correct': correct,
                'total': total,
                'char_accuracy': char_acc,
                'loss': val_loss
            }
            
            results.append(result)
            
            print(f"Epoch {epoch_num:2d}: {correct:2d}/{total} correct ({val_acc:.4f}) | Char: {char_acc:.4f}")
            
            # Track best
            if correct > best_correct:
                best_correct = correct
                best_accuracy = val_acc
                best_epoch = epoch_num
                
        except Exception as e:
            print(f"Epoch {epoch_num:2d}: ERROR - {e}")
    
    # Summary
    print(f"\n🎯 BEST PERFORMANCE:")
    print(f"   Epoch: {best_epoch}")
    print(f"   Correct: {best_correct}/43")
    print(f"   Accuracy: {best_accuracy:.4f}")
    
    # Find epochs with high performance
    print(f"\n📊 HIGH PERFORMANCE EPOCHS:")
    high_perf = [r for r in results if r['correct'] >= 15]
    high_perf.sort(key=lambda x: x['correct'], reverse=True)
    
    for i, r in enumerate(high_perf[:5], 1):
        print(f"   {i}. Epoch {r['epoch']:2d}: {r['correct']}/43 correct ({r['accuracy']:.4f})")
    
    # Check if we found the 22 correct predictions
    if best_correct >= 22:
        print(f"\n🎉 FOUND EPOCH WITH 22+ CORRECT PREDICTIONS!")
        print(f"   Use epoch {best_epoch} for best performance")
        
        # Create command to use this epoch
        print(f"\n🚀 To use this best epoch:")
        print(f"   cp output/vin_rec_finetune/epoch_{best_epoch}.pdparams output/vin_rec_finetune/best_accuracy.pdparams")
        print(f"   python resume_training.py")
        
    elif best_correct >= 15:
        print(f"\n✅ Found good performance epoch: {best_epoch} with {best_correct} correct")
        
    else:
        print(f"\n⚠️ Best performance is only {best_correct} correct predictions")
    
    return best_epoch, best_correct

if __name__ == '__main__':
    best_epoch, best_correct = find_best_epoch()
    print(f"\n🏆 Best epoch: {best_epoch} with {best_correct} correct predictions")
