#!/usr/bin/env python3
"""
Minimal test for CrossEntropyLoss configuration
"""

import sys
import numpy as np
import paddle
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

def test_minimal_crossentropy():
    """Test minimal CrossEntropyLoss configuration."""
    
    print("🔍 Testing Minimal CrossEntropyLoss Configuration")
    print("=" * 50)
    
    try:
        # Test basic imports
        import paddle.nn as nn
        import paddle.optimizer as optim
        
        print("✅ Imports successful")
        
        # Test basic CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        print(f"✅ CrossEntropyLoss created: {type(criterion)}")
        
        # Test basic optimizer
        optimizer = optim.Adam(
            parameters=paddle.nn.Linear(100, 34),  # Dummy parameters
            learning_rate=0.002
        )
        print(f"✅ Optimizer created: {type(optimizer)}")
        
        # Test basic data
        batch_size = 2
        seq_len = 17
        num_classes = 34
        
        # Create dummy data
        logits = paddle.randn(batch_size, seq_len, num_classes)
        labels = paddle.randint(0, num_classes, [batch_size, seq_len], dtype='int64')
        
        print(f"✅ Dummy data created: logits {logits.shape}, labels {labels.shape}")
        
        # Test loss calculation
        loss = criterion(logits, labels)
        print(f"✅ Loss calculation successful: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print(f"✅ Backward pass successful")
        
        print("\n🎉 Minimal CrossEntropyLoss test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    success = test_minimal_crossentropy()
    if success:
        print("\n🚀 CrossEntropyLoss configuration is working!")
        print("Ready to train with CrossEntropyLoss!")
    else:
        print("\n❌ CrossEntropyLoss test failed!")
        sys.exit(1)
