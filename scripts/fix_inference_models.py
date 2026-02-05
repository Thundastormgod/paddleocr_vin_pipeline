#!/usr/bin/env python3
"""
Fix Inference Models for PaddleOCR v5
=====================================

This script fixes existing inference model directories that are missing
the required `inference.yml` config file for PaddleOCR v5 API.

PaddleOCR v5 requires:
- inference.pdmodel (or inference.pdiparams)
- inference.yml (config file)
- vin_dict.txt (character dictionary)

Usage:
    python scripts/fix_inference_models.py
    python scripts/fix_inference_models.py --dir output/my_model/inference
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# VIN characters (excludes I, O, Q as per ISO 3779)
VIN_CHARSET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"


def create_inference_yml(inference_dir: Path, num_classes: int = 34) -> bool:
    """
    Create inference.yml config file for PaddleOCR v5.
    
    Args:
        inference_dir: Path to inference directory
        num_classes: Number of output classes (33 VIN chars + 1 blank)
        
    Returns:
        True if created successfully
    """
    config = {
        'Global': {
            'model_name': 'VIN_Recognition_Model',
            'model_type': 'rec',
            'algorithm': 'SVTR_LCNet',
            'Transform': None,
            'infer_img': './doc/imgs_words/en/word_1.png',
        },
        'Architecture': {
            'model_type': 'rec',
            'algorithm': 'SVTR_LCNet',
            'in_channels': 3,
            'Backbone': {
                'name': 'PPLCNetV3',
                'scale': 0.95,
            },
            'Neck': {
                'name': 'SequenceEncoder',
                'encoder_type': 'reshape',
            },
            'Head': {
                'name': 'CTCHead',
                'out_channels': num_classes,
            },
        },
        'PostProcess': {
            'name': 'CTCLabelDecode',
            'character_dict_path': './vin_dict.txt',
            'use_space_char': False,
        },
    }
    
    config_path = inference_dir / 'inference.yml'
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"  ❌ Failed to create config: {e}")
        return False


def create_vin_dict(inference_dir: Path) -> bool:
    """Create VIN character dictionary if missing."""
    dict_path = inference_dir / 'vin_dict.txt'
    if dict_path.exists():
        return True
    
    try:
        with open(dict_path, 'w') as f:
            for char in VIN_CHARSET:
                f.write(f"{char}\n")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create dict: {e}")
        return False


def fix_inference_directory(inference_dir: Path) -> dict:
    """
    Fix a single inference directory.
    
    Returns:
        Dict with status information
    """
    result = {
        'path': str(inference_dir),
        'has_pdmodel': False,
        'has_pdiparams': False,
        'has_yml': False,
        'has_dict': False,
        'fixed_yml': False,
        'fixed_dict': False,
        'usable': False,
    }
    
    # Check existing files
    result['has_pdmodel'] = (inference_dir / 'inference.pdmodel').exists()
    result['has_pdiparams'] = (inference_dir / 'inference.pdiparams').exists()
    result['has_yml'] = (inference_dir / 'inference.yml').exists()
    result['has_dict'] = (inference_dir / 'vin_dict.txt').exists()
    
    # Fix missing yml
    if not result['has_yml']:
        if create_inference_yml(inference_dir):
            result['fixed_yml'] = True
            result['has_yml'] = True
    
    # Fix missing dict
    if not result['has_dict']:
        if create_vin_dict(inference_dir):
            result['fixed_dict'] = True
            result['has_dict'] = True
    
    # Check if usable
    result['usable'] = result['has_pdiparams'] and result['has_yml'] and result['has_dict']
    
    return result


def find_inference_directories(base_dir: Path) -> list:
    """Find all inference directories in output."""
    inference_dirs = []
    
    for item in base_dir.iterdir():
        if item.is_dir():
            inference_subdir = item / 'inference'
            if inference_subdir.exists() and inference_subdir.is_dir():
                inference_dirs.append(inference_subdir)
    
    return inference_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Fix inference model directories for PaddleOCR v5'
    )
    parser.add_argument(
        '--dir', '-d',
        default=None,
        help='Specific inference directory to fix'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='output',
        help='Base output directory to scan (default: output)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fix Inference Models for PaddleOCR v5")
    print("=" * 60)
    
    # Find directories to fix
    if args.dir:
        inference_dirs = [Path(args.dir)]
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"❌ Output directory not found: {output_dir}")
            sys.exit(1)
        inference_dirs = find_inference_directories(output_dir)
    
    if not inference_dirs:
        print("No inference directories found.")
        sys.exit(0)
    
    print(f"\nFound {len(inference_dirs)} inference directories:\n")
    
    # Process each directory
    fixed_count = 0
    usable_count = 0
    
    for inference_dir in inference_dirs:
        model_name = inference_dir.parent.name
        print(f"📁 {model_name}/inference/")
        
        result = fix_inference_directory(inference_dir)
        
        # Print status
        status_icons = {
            'has_pdmodel': '✅' if result['has_pdmodel'] else '⚠️ ',
            'has_pdiparams': '✅' if result['has_pdiparams'] else '❌',
            'has_yml': '✅' if result['has_yml'] else '❌',
            'has_dict': '✅' if result['has_dict'] else '❌',
        }
        
        print(f"   {status_icons['has_pdmodel']} inference.pdmodel {'(optional)' if not result['has_pdmodel'] else ''}")
        print(f"   {status_icons['has_pdiparams']} inference.pdiparams")
        print(f"   {status_icons['has_yml']} inference.yml {'(created)' if result['fixed_yml'] else ''}")
        print(f"   {status_icons['has_dict']} vin_dict.txt {'(created)' if result['fixed_dict'] else ''}")
        
        if result['fixed_yml'] or result['fixed_dict']:
            fixed_count += 1
        
        if result['usable']:
            usable_count += 1
            print(f"   ✅ USABLE with PaddleOCR v5 API")
        else:
            print(f"   ❌ NOT USABLE - missing required files")
        
        print()
    
    # Summary
    print("=" * 60)
    print(f"Summary: {usable_count}/{len(inference_dirs)} models are usable")
    if fixed_count > 0:
        print(f"Fixed: {fixed_count} directories")
    print("=" * 60)
    
    if usable_count < len(inference_dirs):
        print("\n⚠️  Some models cannot be used with PaddleOCR v5 API.")
        print("   They may work with the ONNX inference module or need to be")
        print("   re-exported using paddle.jit.save() to create .pdmodel files.")


if __name__ == '__main__':
    main()
