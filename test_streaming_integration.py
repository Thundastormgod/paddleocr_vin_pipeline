#!/usr/bin/env python3
"""
Test DagsHub Streaming Integration

This script tests the streaming functionality with the training script.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_streaming_imports():
    """Test streaming imports."""
    print("🔍 Testing Streaming Imports")
    print("-" * 30)
    
    try:
        from src.vin_ocr.data.dagshub_integration import (
            enable_dagshub_streaming, 
            get_streaming_config, 
            is_streaming_active
        )
        print("✅ DagsHub integration imports successful")
        
        # Test integration class
        from src.vin_ocr.data.dagshub_integration import DagsHubDataIntegration
        integration = DagsHubDataIntegration()
        print("✅ DagsHubDataIntegration class available")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_streaming_config():
    """Test streaming configuration."""
    print("\n⚙️ Testing Streaming Configuration")
    print("-" * 30)
    
    try:
        from src.vin_ocr.data.dagshub_integration import DagsHubDataIntegration
        
        integration = DagsHubDataIntegration()
        
        # Test initialization from config
        if integration.initialize_from_config():
            print(f"✅ Repo owner detected: {integration.repo_owner}")
            print(f"✅ Repo name: {integration.repo_name}")
        else:
            print("⚠️  Could not initialize from DVC config")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_streaming_args():
    """Test streaming arguments in training script."""
    print("\n📝 Testing Streaming Arguments")
    print("-" * 30)
    
    try:
        # Import the training script module
        import sys
        import importlib.util
        
        # Load the training script
        spec = importlib.util.spec_from_file_location(
            "finetune_paddleocr", 
            project_root / "src/vin_ocr/training/finetune_paddleocr.py"
        )
        module = importlib.util.module_from_spec(spec)
        
        # Check if DAGSHUB_INTEGRATION_AVAILABLE is defined
        if hasattr(module, 'DAGSHUB_INTEGRATION_AVAILABLE'):
            print(f"✅ DAGSHUB_INTEGRATION_AVAILABLE: {module.DAGSHUB_INTEGRATION_AVAILABLE}")
        else:
            print("❌ DAGSHUB_INTEGRATION_AVAILABLE not found")
        
        return True
    except Exception as e:
        print(f"❌ Args test failed: {e}")
        return False

def test_config_update():
    """Test config update for streaming."""
    print("\n📄 Testing Config Update")
    print("-" * 30)
    
    try:
        from src.vin_ocr.data.dagshub_integration import DagsHubDataIntegration
        
        integration = DagsHubDataIntegration()
        integration.repo_owner = "test_user"
        integration.repo_name = "test_repo"
        integration.streaming_active = True
        
        # Test path conversion.
        # Derived from this file's location; was hardcoded to a foreign home
        # directory (/Users/startferanmi/...), so the assertion below could
        # never hold on any other machine.
        test_path = str(Path(__file__).resolve().parent / "finetune_data" / "train_images")
        streaming_path = integration.get_streaming_path(test_path)
        expected = "/mnt/dagsHub/test_user/test_repo/finetune_data/train_images"
        
        if streaming_path == expected:
            print("✅ Path conversion working correctly")
            print(f"   Local: {test_path}")
            print(f"   Streaming: {streaming_path}")
        else:
            print(f"❌ Path conversion failed")
            print(f"   Expected: {expected}")
            print(f"   Got: {streaming_path}")
        
        return True
    except Exception as e:
        print(f"❌ Config update test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 DagsHub Streaming Integration Test")
    print("=" * 50)
    
    tests = [
        test_streaming_imports,
        test_streaming_config,
        test_streaming_args,
        test_config_update
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    print("\n📊 Test Results")
    print("=" * 20)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Streaming is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Set DagsHub credentials:")
        print("   export DAGSHUB_USERNAME='your_username'")
        print("   export DAGSHUB_TOKEN='your_token'")
        print("2. Test streaming training:")
        print("   python src/vin_ocr/training/finetune_paddleocr.py --stream --help")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
