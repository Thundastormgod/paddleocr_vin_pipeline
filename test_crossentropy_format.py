#!/usr/bin/env python3
"""
Test CrossEntropyLoss with proper label format
"""

import sys
import numpy as np
import paddle
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

def test_crossentropy_format():
    """Test CrossEntropyLoss with proper label format."""
    
    print("🔍 Testing CrossEntropyLoss Label Format")
    print("=" * 50)
    
    try:
        # Test CrossEntropyLoss
        criterion = paddle.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        print(f"✅ CrossEntropyLoss created: {type(criterion)}")
        
        # Test with proper label format for CrossEntropyLoss
        # CrossEntropyLoss expects: [B, C] where B=batch_size, C=seq_len
        batch_size = 2
        seq_len = 17
        num_classes = 34
        
        # Create dummy data in CrossEntropyLoss format
        logits = paddle.randn(batch_size, seq_len, num_classes)  # [B, C] = [2, 17]
        labels = paddle.randint(0, num_classes, [batch_size, seq_len])  # [B] = [2]
        
        print(f"✅ Test data created:")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Expected format: [B, C] = [{batch_size}, {seq_len}]")
        
        # Test loss calculation
        loss = criterion(logits, labels)
        print(f"✅ Loss calculation successful: {loss.item():.4f}")
        
        print("\n🎉 CrossEntropyLoss format test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    success = test_crossentropy_format()
    if success:
        print("\n🚀 CrossEntropyLoss format is working!")
        print("Ready to use CrossEntropyLoss!")
    else:
        print("\n❌ CrossEntropyLoss format test failed!")
        sys.exit(1)
