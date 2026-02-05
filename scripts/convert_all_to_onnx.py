#!/usr/bin/env python3
"""
Convert All PaddlePaddle Inference Models to ONNX
=================================================

This script finds all inference directories and converts them to ONNX format.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def convert_all_models(output_base: str = "output", onnx_dir: str = "output/onnx"):
    """Convert all PaddlePaddle inference models to ONNX."""
    import paddle2onnx
    import onnx
    
    output_base = Path(output_base)
    onnx_dir = Path(onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Converting PaddlePaddle Models to ONNX")
    print("=" * 60)
    print(f"  paddle2onnx version: {paddle2onnx.__version__}")
    print(f"  onnx version: {onnx.__version__}")
    print(f"  Output directory: {onnx_dir}")
    print()
    
    # Find all inference directories
    inference_dirs = []
    for item in output_base.iterdir():
        if item.is_dir():
            inference_subdir = item / 'inference'
            if inference_subdir.exists():
                inference_dirs.append(inference_subdir)
    
    print(f"Found {len(inference_dirs)} inference directories\n")
    
    converted = []
    failed = []
    
    for inference_dir in inference_dirs:
        model_name = inference_dir.parent.name
        print(f"📁 {model_name}")
        
        # Check for required files
        pdmodel_path = inference_dir / 'inference.pdmodel'
        pdiparams_path = inference_dir / 'inference.pdiparams'
        
        if not pdiparams_path.exists():
            print(f"   ❌ Missing inference.pdiparams")
            failed.append((model_name, "Missing pdiparams"))
            continue
        
        # Output path
        onnx_path = onnx_dir / f"{model_name}.onnx"
        
        try:
            if pdmodel_path.exists():
                # Full inference model - use paddle2onnx directly
                print(f"   Converting from pdmodel + pdiparams...")
                paddle2onnx.export(
                    str(pdmodel_path),
                    str(pdiparams_path),
                    str(onnx_path),
                    opset_version=11,
                    auto_upgrade_opset=True,
                    verbose=False,
                )
            else:
                # Only pdiparams - need to rebuild model and export
                print(f"   ⚠️  No pdmodel found, attempting to create static model...")
                
                # Try to load weights and export using paddle
                import paddle
                import paddle.nn as nn
                
                # Load our custom model architecture
                from src.vin_ocr.training.finetune_paddleocr import VINRecognitionModel
                
                # Create model with default config
                config = {
                    'Architecture': {'Neck': {'hidden_dim': 256}},
                }
                model = VINRecognitionModel(config, num_classes=34)
                
                # Load weights
                state_dict = paddle.load(str(pdiparams_path))
                model.set_state_dict(state_dict)
                model.eval()
                
                # Export to static graph first
                temp_model_path = inference_dir / 'temp_inference'
                input_spec = [
                    paddle.static.InputSpec(shape=[None, 3, 48, 320], dtype='float32', name='image')
                ]
                paddle.jit.save(model, str(temp_model_path), input_spec=input_spec)
                
                # Now convert to ONNX
                paddle2onnx.export(
                    str(temp_model_path) + '.pdmodel',
                    str(temp_model_path) + '.pdiparams',
                    str(onnx_path),
                    opset_version=11,
                    auto_upgrade_opset=True,
                    verbose=False,
                )
                
                # Clean up temp files
                for ext in ['.pdmodel', '.pdiparams', '.pdiparams.info']:
                    temp_file = Path(str(temp_model_path) + ext)
                    if temp_file.exists():
                        temp_file.unlink()
            
            # Verify ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Get file size
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            
            print(f"   ✅ Converted: {onnx_path.name} ({size_mb:.1f} MB)")
            converted.append((model_name, str(onnx_path)))
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed.append((model_name, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  ✅ Converted: {len(converted)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if converted:
        print(f"\n  ONNX models saved to: {onnx_dir}/")
        for name, path in converted:
            print(f"    - {name}.onnx")
    
    if failed:
        print(f"\n  Failed conversions:")
        for name, error in failed:
            print(f"    - {name}: {error[:50]}...")
    
    return converted, failed


def test_onnx_inference(onnx_path: str, test_image: str = None):
    """Test ONNX model inference."""
    import onnxruntime as ort
    import numpy as np
    
    print(f"\n  Testing: {onnx_path}")
    
    # Load model
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Get input info
    input_info = session.get_inputs()[0]
    print(f"    Input: {input_info.name} {input_info.shape}")
    
    # Create dummy input
    if input_info.shape[0] is None or isinstance(input_info.shape[0], str):
        batch = 1
    else:
        batch = input_info.shape[0]
    
    # Handle dynamic shapes
    shape = []
    for dim in input_info.shape:
        if dim is None or isinstance(dim, str):
            shape.append(1 if len(shape) == 0 else 48 if len(shape) == 2 else 320 if len(shape) == 3 else 3)
        else:
            shape.append(dim)
    
    dummy_input = np.random.randn(*shape).astype(np.float32)
    
    # Run inference
    try:
        outputs = session.run(None, {input_info.name: dummy_input})
        print(f"    Output shape: {outputs[0].shape}")
        print(f"    ✅ Inference OK")
        return True
    except Exception as e:
        print(f"    ❌ Inference failed: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert all models to ONNX')
    parser.add_argument('--output-base', default='output', help='Base output directory')
    parser.add_argument('--onnx-dir', default='output/onnx', help='ONNX output directory')
    parser.add_argument('--test', action='store_true', help='Test converted models')
    
    args = parser.parse_args()
    
    converted, failed = convert_all_models(args.output_base, args.onnx_dir)
    
    if args.test and converted:
        print("\n" + "=" * 60)
        print("Testing ONNX Models")
        print("=" * 60)
        for name, path in converted:
            test_onnx_inference(path)
