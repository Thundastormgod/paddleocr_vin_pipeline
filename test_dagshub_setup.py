#!/usr/bin/env python3
"""
Test DagsHub Streaming Configuration

This script tests the current DagsHub setup and provides guidance.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_dvc_setup():
    """Test DVC configuration."""
    print("🔧 Testing DVC Setup")
    print("-" * 30)
    
    # Check .dvc directory
    dvc_dir = project_root / ".dvc"
    if dvc_dir.exists():
        print("✅ .dvc directory exists")
        
        # Check config
        config_file = dvc_dir / "config"
        if config_file.exists():
            print("✅ DVC config exists")
            with open(config_file, 'r') as f:
                config_content = f.read()
                print(f"   Content: {config_content.strip()}")
        else:
            print("❌ DVC config not found")
    else:
        print("❌ .dvc directory not found")
        print("   Run: dvc init")
    
    # Check .dvcignore
    dvcignore_file = project_root / ".dvcignore"
    if dvcignore_file.exists():
        print("✅ .dvcignore exists")
    else:
        print("⚠️  .dvcignore not found")

def test_dagshub_packages():
    """Test required packages."""
    print("\n📦 Testing Required Packages")
    print("-" * 30)
    
    packages = {
        'dvc': 'DVC',
        'dagshub': 'DagsHub',
        'paddlepaddle': 'PaddlePaddle'
    }
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} not installed")
            print(f"   Install with: pip install {package}")

def test_data_directories():
    """Test data directory structure."""
    print("\n📁 Testing Data Directories")
    print("-" * 30)
    
    data_dirs = [
        "finetune_data/train_images",
        "finetune_data/val_images", 
        "finetune_data/train_labels.txt",
        "finetune_data/val_labels.txt",
        "dagshub_data",
        "data"
    ]
    
    for data_path in data_dirs:
        full_path = project_root / data_path
        if full_path.exists():
            print(f"✅ {data_path}")
        else:
            print(f"❌ {data_path}")

def test_current_dvc_config():
    """Test current DVC remote configuration."""
    print("\n🌐 Testing DVC Remote Configuration")
    print("-" * 30)
    
    try:
        import subprocess
        result = subprocess.run(
            ["dvc", "remote", "list"], 
            cwd=project_root,
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print("✅ DVC remotes configured:")
            print(result.stdout)
        else:
            print("❌ No DVC remotes configured")
            print("   Run setup script to configure DagsHub remote")
            
    except FileNotFoundError:
        print("❌ DVC not installed or not in PATH")

def test_streaming_imports():
    """Test streaming module imports."""
    print("\n📡 Testing Streaming Imports")
    print("-" * 30)
    
    try:
        from src.vin_ocr.data.setup_dagshub_streaming import DagsHubDataStreamer, DAGSHUB_AVAILABLE
        print("✅ Streaming module imports successfully")
        
        if DAGSHUB_AVAILABLE:
            print("✅ DagsHub package available")
        else:
            print("❌ DagsHub package not available")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")

def main():
    """Run all tests."""
    print("🧪 DagsHub Streaming Configuration Test")
    print("=" * 50)
    
    test_dvc_setup()
    test_dagshub_packages()
    test_data_directories()
    test_current_dvc_config()
    test_streaming_imports()
    
    print("\n🎯 Next Steps")
    print("-" * 30)
    
    # Check if setup is needed
    dvc_dir = project_root / ".dvc"
    if not dvc_dir.exists():
        print("1. Run setup script:")
        print("   python setup_dagshub_dvc.py --repo-owner USER --repo-name REPO --username USER --token TOKEN")
    else:
        print("1. Test streaming:")
        print("   python train_vin_streaming.py --stream --help")
    
    print("2. Track data:")
    print("   dvc add finetune_data/train_images")
    print("   dvc push")
    
    print("3. Train with streaming:")
    print("   python train_vin_streaming.py --stream --repo-owner USER --repo-name REPO --dagshub-user USER --dagshub-token TOKEN")

if __name__ == '__main__':
    main()
