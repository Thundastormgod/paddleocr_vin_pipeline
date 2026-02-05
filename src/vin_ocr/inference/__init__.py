"""
VIN OCR Inference Module
========================

Provides high-performance inference backends for VIN recognition:
- ONNX Runtime inference (CPU, CUDA, CoreML)
- Direct PaddlePaddle inference
- Batch processing support

Usage:
    from src.vin_ocr.inference import ONNXVINRecognizer, VINInference
    
    # ONNX model
    recognizer = ONNXVINRecognizer("output/onnx/model.onnx")
    result = recognizer.recognize("image.jpg")
    
    # Paddle model
    inference = VINInference("output/model/inference")
    result = inference.recognize("image.jpg")
"""

from .onnx_inference import (
    ONNXVINRecognizer,
    ONNXInferenceConfig,
    load_onnx_model,
    VIN_CHARSET,
)

from .paddle_inference import (
    VINInference,
)

__all__ = [
    # ONNX
    'ONNXVINRecognizer',
    'ONNXInferenceConfig', 
    'load_onnx_model',
    'VIN_CHARSET',
    # Paddle
    'VINInference',
]
