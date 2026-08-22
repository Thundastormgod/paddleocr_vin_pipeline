#!/usr/bin/env python3
"""
Re-export PaddlePaddle Models and Convert to ONNX
==================================================

This script:
1. Loads the saved .pdiparams weights
2. Recreates the model architecture
3. Exports to static graph (.pdmodel + .pdiparams)
4. Converts to ONNX

This fixes the issue where models only have .pdiparams without .pdmodel
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import paddle
import paddle.nn as nn
import numpy as np


def create_vin_model():
    """
    Build the canonical recognition model for re-export.

    This function previously re-declared the whole architecture inline; the
    copy had drifted and carried the batch-axis attention defect (paddle
    transformers are batch-first; feeding [T, B, C] attends across batch
    samples - see SVTREncoder in finetune_paddleocr.py). Architecture now
    has exactly one definition, imported here.

    legacy_batch_axis_attention=True because the stranded .pdiparams files
    this tool exists to re-export were all trained before the 2026-08-20
    fix; under the fixed forward those weights score 0.1113 val char
    accuracy instead of the semantics they were trained with (measured).
    """
    from src.vin_ocr.training.finetune_paddleocr import VINRecognitionModel

    config = {'Architecture': {'Neck': {'hidden_dim': 256}}}
    return VINRecognitionModel(
        config, num_classes=34, legacy_batch_axis_attention=True,
    )


def export_model_to_onnx(pdiparams_path: str, output_dir: str):
    """Export a single model to ONNX."""
    import paddle2onnx
    import onnx
    
    pdiparams_path = Path(pdiparams_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = pdiparams_path.parent.name
    
    print(f"\n📁 Processing: {model_name}")
    
    # Create model
    print("   Creating model architecture...")
    model = create_vin_model()
    
    # Load weights from .pdparams (checkpoint file)
    print(f"   Loading weights from: {pdiparams_path.name}")
    try:
        state_dict = paddle.load(str(pdiparams_path), return_numpy=True)
        # Convert numpy arrays to tensors
        state_dict_tensors = {k: paddle.to_tensor(v) for k, v in state_dict.items()}
        model.set_state_dict(state_dict_tensors)
        print(f"   ✅ Loaded {len(state_dict)} parameters")
    except Exception as e:
        print(f"   ❌ Failed to load weights: {e}")
        return None
    
    model.eval()
    
    # Export to static graph
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_model_path = temp_dir / "inference"
    
    print("   Exporting to static graph...")
    # Batch dim is dynamic (None): pinning it to 1 exported graphs that
    # rejected every recognize_batch() call with batch size > 1.
    input_spec = [
        paddle.static.InputSpec(shape=[None, 3, 48, 320], dtype='float32', name='x')
    ]
    
    try:
        paddle.jit.save(model, str(temp_model_path), input_spec=input_spec)
        print(f"   ✅ Static graph exported")
    except Exception as e:
        print(f"   ❌ Static graph export failed: {e}")
        return None
    
    # Convert to ONNX
    # Check which format was exported (.pdmodel for older paddle, .json for newer)
    pdmodel_path = str(temp_model_path) + ".pdmodel"
    pdjson_path = str(temp_model_path) + ".json"
    pdiparams_temp = str(temp_model_path) + ".pdiparams"
    onnx_path = output_dir / f"{model_name}.onnx"
    
    # Use .json if .pdmodel doesn't exist (newer Paddle versions)
    if os.path.exists(pdjson_path) and not os.path.exists(pdmodel_path):
        pdmodel_path = pdjson_path
    
    print("   Converting to ONNX...")
    print(f"      Model file: {os.path.basename(pdmodel_path)}")
    try:
        paddle2onnx.export(
            pdmodel_path,
            pdiparams_temp,
            str(onnx_path),
            opset_version=11,
            auto_upgrade_opset=True,
            verbose=False,
        )
    except Exception as e:
        print(f"   ❌ ONNX conversion failed: {e}")
        return None
    
    # Verify ONNX
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ ONNX model created: {onnx_path.name} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"   ⚠️  ONNX verification warning: {e}")
    
    # Cleanup temp files
    for f in temp_dir.glob("*"):
        f.unlink()
    temp_dir.rmdir()
    
    return str(onnx_path)


def test_onnx_model(onnx_path: str):
    """Test ONNX model with dummy input."""
    import onnxruntime as ort
    
    print(f"\n   Testing inference...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    input_info = session.get_inputs()[0]
    print(f"   Input: {input_info.name} {input_info.shape}")
    
    # Create test input
    dummy_input = np.random.randn(1, 3, 48, 320).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_info.name: dummy_input})
    print(f"   Output shape: {outputs[0].shape}")
    print(f"   ✅ Inference successful!")
    
    return True


def convert_all():
    """Convert all models in output directory."""
    output_base = Path("output")
    onnx_dir = Path("output/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Converting All Models to ONNX")
    print("=" * 60)
    
    # Convert the BEST checkpoint of each run when it exists;
    # latest.pdparams is merely the last epoch, not the selected model.
    pdiparams_files = []
    for run_dir in sorted(p for p in output_base.iterdir() if p.is_dir()):
        best_path = run_dir / "best_accuracy.pdparams"
        latest_path = run_dir / "latest.pdparams"
        if best_path.is_file():
            pdiparams_files.append(best_path)
        elif latest_path.is_file():
            print(f"⚠️  {run_dir.name}: no best_accuracy.pdparams found - "
                  f"falling back to latest.pdparams (LAST epoch, not the "
                  f"best-validation checkpoint)")
            pdiparams_files.append(latest_path)
    print(f"Found {len(pdiparams_files)} models to convert\n")
    
    converted = []
    failed = []
    
    for pdiparams_path in pdiparams_files:
        try:
            onnx_path = export_model_to_onnx(pdiparams_path, onnx_dir)
            if onnx_path:
                test_onnx_model(onnx_path)
                converted.append(onnx_path)
            else:
                failed.append(str(pdiparams_path))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed.append(str(pdiparams_path))
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  ✅ Converted: {len(converted)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if converted:
        print(f"\n  ONNX models saved to: {onnx_dir}/")
        for path in converted:
            print(f"    - {Path(path).name}")
    
    return converted, failed


if __name__ == "__main__":
    converted, failed = convert_all()
