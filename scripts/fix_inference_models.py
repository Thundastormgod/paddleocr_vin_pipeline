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

# Global.model_name stamped into every yml written by this repository -
# both by this script (below) and by the trainer's _create_inference_config
# (src/vin_ocr/training/finetune_paddleocr.py). It marks a repo-trained
# CUSTOM-architecture checkpoint.
REPO_CUSTOM_MODEL_NAME = 'VIN_Recognition_Model'


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
            'model_name': REPO_CUSTOM_MODEL_NAME,
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
            # Resolved against the yml's own directory, NEVER left as a
            # CWD-relative './vin_dict.txt': consumers resolve relative
            # paths against their process CWD, so the old value only
            # worked when the consumer happened to run inside this dir.
            'character_dict_path': str((inference_dir / 'vin_dict.txt').resolve()),
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


def checkpoint_is_stock_architecture(inference_dir: Path,
                                     yml_preexisted: bool) -> bool:
    """
    True only when the checkpoint plausibly IS a stock PaddleOCR zoo model
    that the v5 API can construct and load.
    
    Every training pipeline in this repository exports CUSTOM
    architectures (finetune_paddleocr.py: LCNetV3-SVTR-CTC,
    HGNetV2-SVTR-CTC, Rosetta-ResNet34vd - the stock API's model zoo
    cannot construct their parameter structure; the stock names in the
    yml's Architecture section are boilerplate, not the checkpoint's real
    graph). Both the trainer and this script stamp
    Global.model_name = 'VIN_Recognition_Model' into the ymls they write,
    so:
    
    - a yml this script just created  -> repo provenance -> custom;
    - a pre-existing yml with the repo stamp -> repo provenance -> custom;
    - only a pre-existing yml WITHOUT the stamp (shipped by official
      tooling alongside the model) counts as stock.
    """
    if not yml_preexisted:
        return False
    config_path = inference_dir / 'inference.yml'
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        print(f"  ⚠️  Could not read {config_path}: {e}")
        return False
    if not isinstance(config, dict):
        return False
    global_section = config.get('Global') or {}
    if not isinstance(global_section, dict):
        return False
    return global_section.get('model_name') != REPO_CUSTOM_MODEL_NAME


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
        'has_json': False,
        'has_pdiparams': False,
        'has_yml': False,
        'has_dict': False,
        'fixed_yml': False,
        'fixed_dict': False,
        'stock_architecture': False,
        'usable': False,
    }
    
    # Check existing files (.json is the PIR-era static graph program)
    result['has_pdmodel'] = (inference_dir / 'inference.pdmodel').exists()
    result['has_json'] = (inference_dir / 'inference.json').exists()
    result['has_pdiparams'] = (inference_dir / 'inference.pdiparams').exists()
    result['has_yml'] = (inference_dir / 'inference.yml').exists()
    result['has_dict'] = (inference_dir / 'vin_dict.txt').exists()
    
    yml_preexisted = result['has_yml']
    
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
    
    # "USABLE with PaddleOCR v5 API" requires a checkpoint the stock API
    # can actually construct (stock architecture) AND load (a static
    # graph program plus params plus config plus dict). Repo-trained
    # custom-architecture checkpoints must never get this verdict: the
    # stock zoo cannot rebuild their parameter structure, whatever the
    # boilerplate yml claims.
    result['stock_architecture'] = checkpoint_is_stock_architecture(
        inference_dir, yml_preexisted)
    result['usable'] = (
        result['stock_architecture']
        and (result['has_pdmodel'] or result['has_json'])
        and result['has_pdiparams']
        and result['has_yml']
        and result['has_dict']
    )
    
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
        has_graph = result['has_pdmodel'] or result['has_json']
        status_icons = {
            'has_graph': '✅' if has_graph else '⚠️ ',
            'has_pdiparams': '✅' if result['has_pdiparams'] else '❌',
            'has_yml': '✅' if result['has_yml'] else '❌',
            'has_dict': '✅' if result['has_dict'] else '❌',
        }
        
        graph_name = 'inference.json' if result['has_json'] else 'inference.pdmodel'
        print(f"   {status_icons['has_graph']} {graph_name} {'(missing static graph)' if not has_graph else ''}")
        print(f"   {status_icons['has_pdiparams']} inference.pdiparams")
        print(f"   {status_icons['has_yml']} inference.yml {'(created)' if result['fixed_yml'] else ''}")
        print(f"   {status_icons['has_dict']} vin_dict.txt {'(created)' if result['fixed_dict'] else ''}")
        
        if result['fixed_yml'] or result['fixed_dict']:
            fixed_count += 1
        
        if result['usable']:
            usable_count += 1
            print(f"   ✅ USABLE with PaddleOCR v5 API")
        elif not result['stock_architecture'] and result['has_pdiparams']:
            print(f"   ⚠️  Repo-trained CUSTOM architecture - the stock "
                  f"PaddleOCR v5 API cannot construct it. Serve it with "
                  f"this repo's paddle/ONNX inference modules instead.")
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
