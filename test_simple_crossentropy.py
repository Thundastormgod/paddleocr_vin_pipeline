#!/usr/bin/env python3
"""
Simple CrossEntropyLoss test
"""

import sys
import numpy as np
import paddle
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

def test_simple_crossentropy():
    """Simple test for CrossEntropyLoss."""
    
    print("🔍 Simple CrossEntropyLoss Test")
    print("=" * 40)
    
    try:
        # Test CrossEntropyLoss
        criterion = paddle.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        print(f"✅ CrossEntropyLoss created: {type(criterion)}")
        
        # Test with 2D labels
        batch_size = 2
        seq_len = 17
        num_classes = 34
        
        # Create dummy data
        logits = paddle.randn(batch_size, seq_len, num_classes)
        labels = paddle.randint(0, num_classes, [batch_size, seq_len])  # 2D labels
        
        print(f"✅ Test data created:")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Labels type: {type(labels)}")
        
        # Test loss calculation
        loss = criterion(logits, labels)
        print(f"✅ Loss calculation successful: {loss.item():.4f}")
        
        print("\n🎉 Simple CrossEntropyLoss test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    success = test_simple_crossentropy()
    if success:
        print("\n🚀 CrossEntropyLoss is working!")
        print("Ready to train with CrossEntropyLoss!")
    else:
        print("\n❌ CrossEntropyLoss test failed!")
        sys.exit(1)
