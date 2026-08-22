#!/usr/bin/env python3
"""
Debug script to check training vs final evaluation discrepancy
"""

import sys
import paddle
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_validation():
    """Debug the validation process."""
    
    print("🔍 Debugging Training Validation vs Final Evaluation")
    print("=" * 60)
    
    # Load the trained model
    model_path = project_root / "output/vin_rec_finetune/best_accuracy.pdparams"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Load config
    import yaml
    with open(project_root / "configs/vin_finetune_config.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    from src.vin_ocr.training.finetune_paddleocr import VINFineTuner
    trainer = VINFineTuner(config)
    
    # Load model weights
    trainer.model.set_state_dict(paddle.load(str(model_path)))
    trainer.model.eval()
    
    print(f"📊 Model loaded successfully")
    # Runtime GPU availability, not just the compile flag: a CUDA build on a
    # GPU-less machine must report CPU (audit L39).
    gpu_available = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    print(f"   Device: {'GPU' if gpu_available else 'CPU'}")
    print(f"   Classes: {len(trainer.char_to_idx)}")
    
    # Test on a few validation samples
    val_samples = trainer.val_loader.dataset.samples[:5]
    print(f"\n🔍 Testing on {len(val_samples)} validation samples:")
    
    all_preds = []
    all_targets = []
    
    for i, (img_path, target) in enumerate(val_samples):
        print(f"\nSample {i+1}:")
        print(f"  Target: {target}")
        
        # Load and preprocess image
        from PIL import Image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Preprocess. _preprocess_image returns (tensor, valid_width): the CHW
        # array plus the unpadded content width used for CTC input lengths
        # (audit M31 - the tuple was previously indexed like an array).
        image, _valid_width = trainer.val_loader.dataset._preprocess_image(image)
        
        # Add batch dimension
        image_tensor = paddle.to_tensor(image[np.newaxis, ...])
        
        # Predict
        with paddle.no_grad():
            logits = trainer.model(image_tensor)
            pred = trainer._decode_predictions(logits)[0]
        
        print(f"  Prediction: {pred}")
        print(f"  Match: {'✅' if pred == target else '❌'}")
        
        all_preds.append(pred)
        all_targets.append(target)
    
    # Calculate accuracy
    correct = sum(1 for p, t in zip(all_preds, all_targets, strict=True) if p == t)
    accuracy = correct / len(all_targets)
    
    print(f"\n📊 Results:")
    print(f"   Correct: {correct}/{len(all_targets)}")
    print(f"   Accuracy: {accuracy:.2%}")
    
    # Compare with final evaluation
    import json
    with open(project_root / "output/vin_rec_finetune/training_metrics.json", 'r') as f:
        final_metrics = json.load(f)
    
    final_exact_match = final_metrics['evaluation_metrics']['image_level']['exact_match_accuracy']
    final_char_acc = final_metrics['evaluation_metrics']['character_level']['character_accuracy']
    
    print(f"\n🎯 Comparison:")
    print(f"   This debug: {accuracy:.2%}")
    print(f"   Final eval: {final_exact_match:.2%}")
    print(f"   Char accuracy: {final_char_acc:.2%}")
    
    # "Results match" requires actual agreement within tolerance - any
    # nonzero accuracy is NOT a match (audit M31).
    if abs(accuracy - final_exact_match) < 1e-6:
        print(f"\n✅ Results match - no issue detected")
    else:
        print(f"\n❌ MISMATCH: debug accuracy {accuracy:.2%} vs final evaluation {final_exact_match:.2%}")
        print(f"   The training validation and the final evaluation disagree - investigate!")

if __name__ == '__main__':
    debug_validation()
