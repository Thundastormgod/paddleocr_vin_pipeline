#!/usr/bin/env python3
"""
ONNX Model Export for VIN OCR
==============================

This script exports fine-tuned PaddleOCR models to ONNX format for:
- Production deployment
- Cross-platform inference
- Consistent evaluation results
- Faster inference in production

Usage:
    python -m src.vin_ocr.training.export_onnx --model-path output/vin_rec_finetune/best_accuracy
    python -m src.vin_ocr.training.export_onnx --model-path output/paddleocr_scratch_*/best_model

Author: JLR VIN Project
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check required dependencies for ONNX export."""
    missing = []
    
    try:
        import paddle
    except ImportError:
        missing.append("paddlepaddle")
    
    try:
        import paddle2onnx
    except ImportError:
        missing.append("paddle2onnx")
    
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    
    if missing:
        print("❌ Missing dependencies for ONNX export:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


def export_paddle_to_onnx(
    model_path: str,
    output_dir: str,
    model_name: str = None,
    opset_version: int = 11,
    input_shape: tuple = (1, 1, 32, 320),
) -> str:
    """
    Export PaddleOCR model to ONNX format.
    
    Args:
        model_path: Path to the .pdparams model file or directory
        output_dir: Directory to save ONNX model
        model_name: Optional name for the exported model
        opset_version: ONNX opset version (default: 11)
        input_shape: Input shape (batch, channels, height, width)
    
    Returns:
        Path to the exported ONNX model
    """
    import paddle
    import paddle2onnx
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model name
    if model_name is None:
        if model_path.is_file():
            model_name = model_path.stem
        else:
            model_name = model_path.name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_filename = f"{model_name}_{timestamp}.onnx"
    onnx_path = output_dir / onnx_filename
    
    print(f"\n{'='*60}")
    print("ONNX Model Export")
    print(f"{'='*60}")
    print(f"  Model path: {model_path}")
    print(f"  Output: {onnx_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")
    
    # Find model files
    if model_path.is_file() and model_path.suffix == '.pdparams':
        params_path = model_path
        pdmodel_path = model_path.with_suffix('.pdmodel')
    elif model_path.is_dir():
        # Look for inference model
        params_path = model_path / 'inference.pdiparams'
        pdmodel_path = model_path / 'inference.pdmodel'
        
        if not params_path.exists():
            # Try alternative names
            for name in ['model.pdparams', 'best_accuracy.pdparams', 'latest.pdparams']:
                alt_path = model_path / name
                if alt_path.exists():
                    params_path = alt_path
                    pdmodel_path = alt_path.with_suffix('.pdmodel')
                    break
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Check if inference model exists
    if pdmodel_path.exists() and params_path.exists():
        print(f"\n  Found inference model:")
        print(f"    - {pdmodel_path.name}")
        print(f"    - {params_path.name}")
        
        # Use paddle2onnx to convert
        paddle2onnx.export(
            str(pdmodel_path),
            str(params_path),
            str(onnx_path),
            opset_version=opset_version,
            auto_upgrade_opset=True,
            verbose=True,
        )
    else:
        print(f"\n  ⚠️ Inference model not found. Creating from training checkpoint...")
        print(f"    This requires loading the model architecture.")
        
        # For PaddleOCR recognition models, we need to recreate the model
        # This is more complex and requires knowing the model architecture
        raise NotImplementedError(
            "Direct .pdparams export not yet supported. "
            "Please export to inference model first using PaddleOCR tools:\n"
            "  python tools/export_model.py -c configs/rec/... -o output/inference/"
        )
    
    # Verify the exported model
    print(f"\n  Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ✅ ONNX model is valid!")
        
        # Print model info
        print(f"\n  Model Info:")
        print(f"    - Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"    - Outputs: {[out.name for out in onnx_model.graph.output]}")
        
    except Exception as e:
        print(f"  ⚠️ ONNX validation warning: {e}")
    
    # Test inference with ONNX Runtime
    print(f"\n  Testing inference with ONNX Runtime...")
    try:
        import onnxruntime as ort
        import numpy as np
        
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        
        # Create dummy input
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        print(f"  ✅ Inference test passed!")
        print(f"    - Output shape: {outputs[0].shape}")
        
    except Exception as e:
        print(f"  ⚠️ Inference test failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ ONNX model exported to: {onnx_path}")
    print(f"{'='*60}")
    
    # Save export metadata
    metadata_path = onnx_path.with_suffix('.json')
    import json
    metadata = {
        'source_model': str(model_path),
        'onnx_path': str(onnx_path),
        'export_timestamp': datetime.now().isoformat(),
        'opset_version': opset_version,
        'input_shape': list(input_shape),
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_path}")
    
    return str(onnx_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export PaddleOCR models to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export fine-tuned model
  python -m src.vin_ocr.training.export_onnx --model-path output/vin_rec_finetune/best_accuracy

  # Export with custom output directory
  python -m src.vin_ocr.training.export_onnx --model-path output/model --output-dir output/onnx

  # Export with specific input shape
  python -m src.vin_ocr.training.export_onnx --model-path output/model --input-height 48 --input-width 320
        """
    )
    
    parser.add_argument(
        '--model-path', '-m',
        required=True,
        help='Path to the PaddleOCR model (.pdparams file or directory)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='output/onnx',
        help='Output directory for ONNX model (default: output/onnx)'
    )
    parser.add_argument(
        '--model-name', '-n',
        default=None,
        help='Name for the exported model (default: derived from input)'
    )
    parser.add_argument(
        '--opset-version',
        type=int,
        default=11,
        help='ONNX opset version (default: 11)'
    )
    parser.add_argument(
        '--input-height',
        type=int,
        default=32,
        help='Input image height (default: 32)'
    )
    parser.add_argument(
        '--input-width',
        type=int,
        default=320,
        help='Input image width (default: 320)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Export model
    input_shape = (1, 1, args.input_height, args.input_width)
    
    try:
        onnx_path = export_paddle_to_onnx(
            model_path=args.model_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            opset_version=args.opset_version,
            input_shape=input_shape,
        )
        print(f"\n✅ Export complete: {onnx_path}")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
