#!/usr/bin/env python3
"""
ONNX Model Export and Inference Example
========================================

This script demonstrates the complete workflow for:
1. Exporting a trained PaddleOCR model to ONNX
2. Loading and running inference with the ONNX model

Usage:
    # Export a model to ONNX
    python scripts/onnx_example.py export --model output/vin_rec_finetune/inference
    
    # Run inference with ONNX model
    python scripts/onnx_example.py infer --model output/onnx/model.onnx --image test.jpg
    
    # Run batch inference
    python scripts/onnx_example.py infer --model model.onnx --dir test_images/
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import onnxruntime
        print(f"✅ onnxruntime: {onnxruntime.__version__}")
    except ImportError:
        missing.append("onnxruntime")
    
    try:
        import paddle2onnx
        print(f"✅ paddle2onnx: {paddle2onnx.__version__}")
    except ImportError:
        missing.append("paddle2onnx")
    
    try:
        import onnx
        print(f"✅ onnx: {onnx.__version__}")
    except ImportError:
        missing.append("onnx")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def export_to_onnx(model_path: str, output_dir: str = "output/onnx"):
    """
    Export PaddleOCR model to ONNX format.
    
    Args:
        model_path: Path to inference model directory or .pdparams file
        output_dir: Output directory for ONNX model
    """
    print("\n" + "=" * 60)
    print("ONNX Model Export")
    print("=" * 60)
    
    from src.vin_ocr.training.export_onnx import export_paddle_to_onnx
    
    try:
        onnx_path = export_paddle_to_onnx(
            model_path=model_path,
            output_dir=output_dir,
            opset_version=11,
            input_shape=(1, 3, 48, 320),  # Standard PaddleOCR input
        )
        print(f"\n✅ Model exported to: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_inference(model_path: str, image_path: str = None, image_dir: str = None):
    """
    Run inference with ONNX model.
    
    Args:
        model_path: Path to .onnx model file
        image_path: Single image path
        image_dir: Directory of images
    """
    print("\n" + "=" * 60)
    print("ONNX Inference")
    print("=" * 60)
    
    # Import the ONNX recognizer
    try:
        from src.vin_ocr.inference import ONNXVINRecognizer, ONNXInferenceConfig
    except ImportError:
        print("❌ Could not import ONNX inference module")
        print("   Using fallback inference method...")
        return run_inference_fallback(model_path, image_path, image_dir)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    config = ONNXInferenceConfig(
        input_height=48,
        input_width=320,
        input_channels=3,
        use_gpu=True,
    )
    
    try:
        recognizer = ONNXVINRecognizer(model_path, config=config)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get model info
    info = recognizer.get_model_info()
    print(f"  Provider: {info['provider']}")
    print(f"  Input shape: {info['input_shape']}")
    
    # Collect images
    images = []
    if image_path:
        images.append(image_path)
    if image_dir:
        import glob
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(glob.glob(os.path.join(image_dir, ext)))
            images.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not images:
        print("\n⚠️  No images provided. Use --image or --dir")
        return
    
    # Run inference
    print(f"\nProcessing {len(images)} images...\n")
    
    if len(images) == 1:
        # Single image
        result = recognizer.recognize(images[0])
        print(f"Image: {images[0]}")
        print(f"  VIN: {result['vin']}")
        print(f"  Raw text: {result['raw_text']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Valid: {result['is_valid']}")
        if result['error']:
            print(f"  Error: {result['error']}")
    else:
        # Batch inference
        results = recognizer.recognize_batch(images, batch_size=8)
        
        valid_count = sum(1 for r in results if r['is_valid'])
        
        print(f"{'Image':<40} {'VIN':<20} {'Conf':>8} {'Valid':>6}")
        print("-" * 76)
        
        for img, result in zip(images, results):
            img_name = Path(img).name[:38]
            status = "✅" if result['is_valid'] else "❌"
            print(f"{img_name:<40} {result['vin']:<20} {result['confidence']:>8.4f} {status:>6}")
        
        print("-" * 76)
        print(f"Summary: {valid_count}/{len(images)} valid VINs ({100*valid_count/len(images):.1f}%)")


def run_inference_fallback(model_path: str, image_path: str = None, image_dir: str = None):
    """Fallback inference using direct ONNX Runtime."""
    import onnxruntime as ort
    import cv2
    import numpy as np
    
    print(f"\nLoading model: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # Get input info
    input_info = session.get_inputs()[0]
    print(f"  Input: {input_info.name} {input_info.shape}")
    print(f"  Provider: {session.get_providers()[0]}")
    
    # Collect images
    images = []
    if image_path:
        images.append(image_path)
    if image_dir:
        import glob
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not images:
        print("\n⚠️  No images provided")
        return
    
    # VIN characters
    VIN_CHARSET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
    
    print(f"\nProcessing {len(images)} images...\n")
    
    for img_path in images:
        # Load and preprocess
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ {img_path}: Failed to load")
            continue
        
        # Resize to model input size
        target_h, target_w = 48, 320
        h, w = image.shape[:2]
        ratio = target_h / h
        new_w = min(int(w * ratio), target_w)
        image = cv2.resize(image, (new_w, target_h))
        
        # Pad if needed
        if new_w < target_w:
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            padded[:, :new_w] = image
            image = padded
        
        # Normalize and transpose
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, 0)  # Add batch
        
        # Run inference
        outputs = session.run(None, {input_info.name: image})
        output = outputs[0]
        
        # CTC decode
        pred_indices = np.argmax(output, axis=2)[0]
        decoded = []
        prev_idx = -1
        for idx in pred_indices:
            if idx != 0 and idx != prev_idx:  # 0 is blank
                if idx <= len(VIN_CHARSET):
                    decoded.append(VIN_CHARSET[idx - 1])
            prev_idx = idx
        
        vin = ''.join(decoded)
        confidence = float(np.mean(np.max(output, axis=2)))
        
        print(f"  {Path(img_path).name}: {vin} ({confidence:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='ONNX Model Export and Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model to ONNX')
    export_parser.add_argument('--model', '-m', required=True,
                               help='Path to PaddleOCR model (inference dir or .pdparams)')
    export_parser.add_argument('--output', '-o', default='output/onnx',
                               help='Output directory')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run ONNX inference')
    infer_parser.add_argument('--model', '-m', required=True,
                              help='Path to .onnx model')
    infer_parser.add_argument('--image', '-i', help='Single image path')
    infer_parser.add_argument('--dir', '-d', help='Directory of images')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check dependencies')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        check_dependencies()
    elif args.command == 'export':
        if not check_dependencies():
            sys.exit(1)
        export_to_onnx(args.model, args.output)
    elif args.command == 'infer':
        run_inference(args.model, args.image, args.dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
