#!/usr/bin/env python3
"""
VIN OCR CLI - Command Line Interface
====================================

Main CLI entry point for VIN OCR operations.

Usage:
    vin-ocr recognize <image>          Recognize VIN from image
    vin-ocr batch <folder>             Batch process folder
    vin-ocr serve                      Start web UI
    vin-ocr export <model> <output>    Export model to ONNX
"""

import argparse
import sys
from pathlib import Path


def cmd_recognize(args):
    """Recognize VIN from a single image."""
    from src.vin_ocr.inference import VINInference, ONNXVINRecognizer
    
    image_path = args.image
    model_path = args.model
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return 1
    
    # Determine model type
    if model_path.endswith('.onnx'):
        print(f"Using ONNX model: {model_path}")
        recognizer = ONNXVINRecognizer(model_path)
    else:
        print(f"Using Paddle model: {model_path}")
        recognizer = VINInference(model_path)
    
    result = recognizer.recognize(image_path)
    
    if result.get('error'):
        print(f"Error: {result['error']}")
        return 1
    
    print(f"VIN: {result['vin']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Raw text: {result['raw_text']}")
    
    if args.json:
        import json
        print(json.dumps(result, indent=2))
    
    return 0


def cmd_batch(args):
    """Batch process a folder of images."""
    from src.vin_ocr.inference import VINInference, ONNXVINRecognizer
    import json
    
    folder = Path(args.folder)
    model_path = args.model
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return 1
    
    # Get images
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    
    if not images:
        print(f"No images found in {folder}")
        return 1
    
    print(f"Processing {len(images)} images...")
    
    # Load model
    if model_path.endswith('.onnx'):
        recognizer = ONNXVINRecognizer(model_path)
    else:
        recognizer = VINInference(model_path)
    
    results = []
    for img in images:
        result = recognizer.recognize(str(img))
        result['filename'] = img.name
        results.append(result)
        print(f"  {img.name}: {result.get('vin', 'ERROR')}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return 0


def cmd_serve(args):
    """Start the Streamlit web UI."""
    import subprocess
    
    port = args.port or 8501
    app_path = Path(__file__).parent / "web" / "app.py"
    
    if not app_path.exists():
        app_path = Path("src/vin_ocr/web/app.py")
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    
    print(f"Starting VIN OCR Web UI on port {port}...")
    subprocess.run(cmd)
    
    return 0


def cmd_export(args):
    """Export model to ONNX format."""
    # Use the in-package exporter. The previous import target
    # (scripts.reexport_and_convert_onnx) is not an importable package —
    # scripts/ has no __init__.py and is not installed with the wheel.
    try:
        from src.vin_ocr.training.export_onnx import export_paddle_to_onnx
    except ImportError as exc:
        print(f"Error: ONNX export requires the 'onnx' extra: {exc}")
        print("Install with: pip install -e '.[onnx]'")
        return 1

    model_path = args.model
    output_path = args.output

    print(f"Exporting {model_path} to {output_path}...")

    try:
        result = export_paddle_to_onnx(
            model_path=model_path,
            output_dir=output_path,
            opset_version=args.opset,
        )
    except Exception as exc:
        print(f"Export failed: {exc}")
        return 1

    if result:
        print(f"Successfully exported to: {result}")
        return 0

    print("Export failed")
    return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='vin-ocr',
        description='VIN OCR Pipeline - Recognize Vehicle Identification Numbers from images',
    )
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument(
        '--no-trace', action='store_true',
        help='Disable MLflow tracing (on by default when the tracking '
             'extra is installed; traces land in experiment "vin_finetune" '
             'at MLFLOW_TRACKING_URI or the repo SQLite store)',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Recognize command
    recognize_parser = subparsers.add_parser('recognize', help='Recognize VIN from image')
    recognize_parser.add_argument('image', help='Path to image file')
    recognize_parser.add_argument('--model', '-m', default='output/onnx/final_test.onnx',
                                  help='Path to model (ONNX or Paddle inference dir)')
    recognize_parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process folder')
    batch_parser.add_argument('folder', help='Path to folder with images')
    batch_parser.add_argument('--model', '-m', default='output/onnx/final_test.onnx',
                              help='Path to model')
    batch_parser.add_argument('--output', '-o', help='Output JSON file')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web UI')
    serve_parser.add_argument('--port', '-p', type=int, default=8501, help='Port number')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model to ONNX')
    export_parser.add_argument('model', help='Path to Paddle model (.pdparams)')
    export_parser.add_argument('output', help='Output directory for ONNX')
    export_parser.add_argument('--opset', type=int, default=11,
                               help='ONNX opset version (default: 11)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Entry-point tracing (MLflow instrumentation-skill pattern for custom
    # code: no autolog exists for a paddle inference stack, so the pipeline
    # carries @mlflow.trace spans armed here). Loud no-op when the tracking
    # extra is absent; the recognizer never depends on it.
    if args.command in ('recognize', 'batch') and not args.no_trace:
        from src.vin_ocr.tracking.tracing import enable_tracing
        enable_tracing()
    
    commands = {
        'recognize': cmd_recognize,
        'batch': cmd_batch,
        'serve': cmd_serve,
        'export': cmd_export,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
