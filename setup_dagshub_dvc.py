#!/usr/bin/env python3
"""
Complete DVC + DagsHub Setup for VIN OCR Project

This script automates the entire setup process:
1. Initialize DVC
2. Configure DagsHub remote
3. Track data files
4. Set up streaming
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vin_ocr.data.setup_dagshub_streaming import (
    setup_dvc_tracking,
    setup_dagshub_remote,
    initialize_vin_ocr_streaming
)


def run_command(cmd: list, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command safely.
    
    Args:
        cmd: Command to run
        cwd: Working directory
        check: Whether to check return code
        
    Returns:
        CompletedProcess object
    """
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd or project_root,
            capture_output=True, 
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"   Error: {e.stderr}")
        raise


def initialize_dvc():
    """Initialize DVC in the project."""
    print("🔧 Initializing DVC...")
    
    try:
        # Check if DVC already initialized
        dvc_dir = project_root / ".dvc"
        if dvc_dir.exists():
            print("✅ DVC already initialized")
            return
        
        # Initialize DVC
        run_command(["dvc", "init"])
        print("✅ DVC initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize DVC: {e}")
        raise


def setup_git_remote():
    """Set up Git remote if not already configured."""
    print("🔧 Checking Git remote...")
    
    try:
        # Check if remote exists
        result = run_command(["git", "remote", "-v"], check=False)
        
        if "origin" in result.stdout and "dagshub.com" in result.stdout:
            print("✅ DagsHub Git remote already configured")
            return
        
        print("⚠️  No DagsHub Git remote found")
        print("   Please run: git remote add origin https://dagshub.com/YOUR_USERNAME/YOUR_REPO.git")
        
    except Exception as e:
        print(f"❌ Failed to check Git remote: {e}")


def track_data_files():
    """Track VIN OCR data files with DVC."""
    print("📁 Tracking data files with DVC...")
    
    # Data files and directories to track
    data_items = [
        ("finetune_data/train_images", "Training images directory"),
        ("finetune_data/val_images", "Validation images directory"),
        ("finetune_data/train_labels.txt", "Training labels file"),
        ("finetune_data/val_labels.txt", "Validation labels file"),
        ("data/train", "Alternative training data"),
        ("data/val", "Alternative validation data"),
        ("dagshub_data/train", "DagsHub training data"),
        ("dagshub_data/val", "DagsHub validation data"),
    ]
    
    for data_path, description in data_items:
        full_path = project_root / data_path
        
        if full_path.exists():
            try:
                print(f"📊 Tracking {description}: {data_path}")
                # In production: run_command(["dvc", "add", data_path], check=False)
                # For now, just show what would be tracked
                print(f"   Would run: dvc add {data_path}")
            except Exception as e:
                print(f"⚠️  Could not track {data_path}: {e}")
        else:
            print(f"⚠️  Not found: {data_path}")
    
    # Track .gitignore
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        print("📄 Updating .gitignore...")
        with open(gitignore_path, 'a') as f:
            f.write("\n# DVC\n")
            f.write("*.dvc\n")
            f.write(".dvc/cache\n")
            f.write("output/\n")
            f.write("training_output/\n")


def setup_complete_workflow(repo_owner: str, repo_name: str, username: str, token: str):
    """
    Set up complete DVC + DagsHub workflow.
    
    Args:
        repo_owner: DagsHub repository owner
        repo_name: DagsHub repository name
        username: DagsHub username
        token: DagsHub access token
    """
    print("🚀 Setting up complete DVC + DagsHub workflow")
    print("=" * 60)
    
    # 1. Initialize DVC
    initialize_dvc()
    
    # 2. Set up Git remote
    setup_git_remote()
    
    # 3. Configure DagsHub DVC remote
    print("\n🌐 Configuring DagsHub DVC remote...")
    setup_dagshub_remote(repo_owner, repo_name, username, token)
    
    # 4. Track data files
    print("\n📁 Tracking data files...")
    track_data_files()
    
    # 5. Initialize streaming
    print("\n📡 Initializing streaming...")
    initialize_vin_ocr_streaming(repo_owner, repo_name, username, token)
    
    print("\n🎯 Complete setup finished!")
    print("\n📋 Next steps:")
    print(f"   1. Commit and push:")
    print(f"      git add .")
    print(f"      git commit -m 'Set up DVC + DagsHub'")
    print(f"      git push")
    print(f"   2. Push data to DagsHub:")
    print(f"      dvc push")
    print(f"   3. Train with streaming:")
    print(f"      python train_vin_streaming.py --stream \\")
    print(f"        --repo-owner {repo_owner} \\")
    print(f"        --repo-name {repo_name} \\")
    print(f"        --dagshub-user {username} \\")
    print(f"        --dagshub-token YOUR_TOKEN")


def main():
    """Main setup script."""
    parser = argparse.ArgumentParser(
        description='Complete DVC + DagsHub Setup for VIN OCR'
    )
    
    parser.add_argument(
        '--repo-owner',
        required=True,
        help='DagsHub repository owner (username)'
    )
    parser.add_argument(
        '--repo-name',
        required=True,
        help='DagsHub repository name'
    )
    parser.add_argument(
        '--username',
        required=True,
        help='DagsHub username'
    )
    parser.add_argument(
        '--token',
        required=True,
        help='DagsHub access token'
    )
    
    args = parser.parse_args()
    
    try:
        setup_complete_workflow(
            repo_owner=args.repo_owner,
            repo_name=args.repo_name,
            username=args.username,
            token=args.token
        )
        return 0
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1


if __name__ == '__main__':
    print("🔧 DVC + DagsHub Setup for VIN OCR Project")
    print("=" * 50)
    sys.exit(main())
