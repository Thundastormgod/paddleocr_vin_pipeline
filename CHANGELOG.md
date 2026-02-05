# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-03

### Added
- Initial release of VIN OCR Pipeline
- **Training**
  - Fine-tuning PaddleOCR models (PP-OCRv3, PP-OCRv4, PP-OCRv5)
  - Training from scratch with multiple architectures (SVTR, CRNN, PP-OCRv5)
  - Hyperparameter tuning with Optuna
  - Data augmentation pipeline
- **Inference**
  - Direct PaddlePaddle inference (`VINInference`)
  - ONNX Runtime inference (`ONNXVINRecognizer`)
  - CoreML acceleration on Apple Silicon
  - Batch processing support
- **Evaluation**
  - Character-level and exact match accuracy
  - Multi-model comparison
  - Detailed metrics export (JSON, CSV)
- **Web UI** (Streamlit)
  - Single image recognition
  - Batch processing
  - Model training interface
  - Real-time evaluation dashboard
  - Model comparison tools
- **CLI Tools**
  - `vin-ocr recognize` - Single image recognition
  - `vin-ocr batch` - Batch processing
  - `vin-ocr serve` - Start web UI
  - `vin-ocr export` - ONNX export
- **Export**
  - ONNX model conversion
  - Inference model export
- **Documentation**
  - Architecture overview
  - API documentation
  - Training guide

### Technical Details
- PaddlePaddle 3.0+ support
- PaddleOCR v5 API compatibility
- ONNX opset 11 export
- Python 3.10+ support

## [Unreleased]

### Planned
- TensorRT optimization
- Mobile deployment (ONNX + CoreML)
- REST API server
- Active learning pipeline
- Multi-GPU training support
