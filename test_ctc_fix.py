#!/usr/bin/env python3
"""
Simple test to verify CTC training fixes
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

def test_encode_label():
    """Test the _encode_label method fix."""
    
    from src.vin_ocr.training.finetune_paddleocr import VINFineTuner
    
    # Create minimal config
    config = {
        'Global': {
            'max_text_length': 17,
            'character_dict_path': './configs/vin_dict.txt'
        },
        'Architecture': {
            'Head': {
                'name': 'MultiHead'
            }
        }
    }
    
    # Initialize trainer
    trainer = VINFineTuner(config)
    
    # Test encoding
    test_label = "1HGTMN456789012345"
    print(f"Testing label encoding: {test_label}")
    
    try:
        encoded = trainer._encode_label(test_label)
        print(f"✅ Encoding successful: {encoded}")
        print(f"   Shape: {encoded.shape}")
        print(f"   Dtype: {encoded.dtype}")
        print(f"   Values: {encoded[:10]}...")
        
        # Test CTC requirements
        assert encoded.dtype == np.int32, f"❌ Wrong dtype: {encoded.dtype}, expected int32"
        assert len(encoded) == 17, f"❌ Wrong length: {len(encoded)}, expected 17"
        
        print("✅ All CTC encoding tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Encoding failed: {e}")
        return False

if __name__ == '__main__':
    success = test_encode_label()
    if success:
        print("\n🎉 CTC encoding fix verified!")
    else:
        print("\n❌ CTC encoding fix failed!")
    
    print("\n📋 Ready for training with CTC architecture!")
