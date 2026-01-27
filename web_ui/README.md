# VIN OCR Web UI

A user-friendly web interface for VIN recognition using multiple OCR models.

## Features

- **🔍 Single Image Recognition**: Upload and process individual VIN images
- **📊 Batch Evaluation**: Process folders of images with automatic metrics
- **🎯 Training Interface**: Configure and monitor model training
- **📈 Results Dashboard**: View and export recognition results

## Quick Start

```bash
# Install dependencies (from project root)
pip install -r requirements.txt

# Run the web UI
streamlit run web_ui/app.py

# Or with custom port
streamlit run web_ui/app.py --server.port 8080
```

## Available Models

| Model | Description | Requirements |
|-------|-------------|--------------|
| VIN Pipeline (PP-OCRv5) | Full pipeline with post-processing | Default |
| PaddleOCR PP-OCRv4 | Latest PaddleOCR model | paddleocr |
| PaddleOCR PP-OCRv3 | Previous generation | paddleocr |
| DeepSeek-OCR | Vision-language model | transformers, torch |

## Screenshots

### Recognition Page
Upload an image and get instant VIN recognition with confidence scores.

### Batch Evaluation
Compare multiple models on your dataset with detailed metrics.

### Training Interface
Configure and launch model training with custom parameters.

## Configuration

The web UI uses the same configuration files as the CLI tools:

- `configs/vin_finetune_config.yml` - PaddleOCR training config
- `configs/deepseek_finetune_config.yml` - DeepSeek training config

## API Usage

The web UI can also be embedded as an API:

```python
from web_ui.app import recognize_with_model

result = recognize_with_model("image.jpg", "VIN Pipeline (PP-OCRv5)")
print(result['vin'], result['confidence'])
```
