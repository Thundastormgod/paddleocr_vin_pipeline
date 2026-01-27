# Installation Guide

## Quick Start (Basic Installation)

For basic VIN recognition using PaddleOCR only:

```bash
pip install -r requirements.txt
```

This installs:
- PaddleOCR and PaddlePaddle (CPU version)
- OpenCV, NumPy, Pillow for image processing
- Pydantic for configuration management
- PyYAML for config files
- Streamlit, Plotly, Pandas for web UI
- pytest for testing

## Advanced Installation Options

### 1. GPU Acceleration (Recommended for Training)

Replace PaddlePaddle CPU with GPU version:

```bash
pip uninstall paddlepaddle
pip install paddlepaddle-gpu  # For CUDA 11.8+
```

### 2. DeepSeek-OCR Support

If you want to use DeepSeek-OCR models, install additional dependencies:

```bash
pip install transformers>=5.0.0
pip install accelerate>=1.12.0
pip install peft>=0.18.1
pip install bitsandbytes>=0.49.1
pip install safetensors>=0.7.0
pip install addict>=2.4.0
pip install matplotlib>=3.10.0
pip install einops>=0.8.0
pip install easydict>=1.13
pip install timm>=1.0.24

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Or install all at once from requirements_complete.txt:

```bash
pip install -r requirements_complete.txt
```

### 3. Web UI Only

If you only need the web interface:

```bash
pip install streamlit plotly pandas pillow
pip install paddleocr opencv-python numpy
```

## Dependency Overview

### Core Dependencies (Always Required)
- **paddleocr** - Main OCR engine
- **paddlepaddle** - Deep learning framework (CPU or GPU)
- **opencv-python** - Image processing
- **numpy** - Numerical computations
- **Pillow** - Image handling
- **pydantic** - Configuration validation
- **pyyaml** - YAML config parsing

### Web UI Dependencies
- **streamlit** - Web application framework
- **plotly** - Interactive visualizations
- **pandas** - Data analysis and export

### Optional Dependencies
- **pytest** - Testing framework
- **visualdl** - TensorBoard-like visualization for PaddlePaddle
- **transformers** - For DeepSeek-OCR and other Hugging Face models
- **accelerate** - Training acceleration utilities
- **peft** - Parameter-efficient fine-tuning
- **bitsandbytes** - 8-bit optimizers

## Installation Methods

### Method 1: Basic (PaddleOCR + Web UI)
```bash
# Install from main requirements
pip install -r requirements.txt

# Test installation
python -c "from paddleocr import PaddleOCR; print('✓ PaddleOCR installed')"
python -c "import streamlit; print('✓ Streamlit installed')"
```

### Method 2: Full Installation (All Features)
```bash
# Install everything including DeepSeek
pip install -r requirements_complete.txt

# Test installation
python -c "from paddleocr import PaddleOCR; import transformers; print('✓ All dependencies installed')"
```

### Method 3: Custom Installation
```bash
# Core only (no web UI)
pip install paddleocr paddlepaddle opencv-python numpy pillow pydantic pyyaml

# Add web UI later
pip install streamlit plotly pandas
```

## Verification

After installation, verify everything works:

```bash
# Test basic OCR
python example_usage.py

# Test web UI
streamlit run web_ui/app.py

# Run tests
pytest tests/
```

## Troubleshooting

### Issue: PaddlePaddle CUDA version mismatch
```bash
# Check CUDA version
nvidia-smi

# Install matching PaddlePaddle version
# For CUDA 11.8:
pip install paddlepaddle-gpu

# For CUDA 12.0+:
pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

### Issue: OpenCV import errors
```bash
# Try headless version
pip uninstall opencv-python
pip install opencv-python-headless
```

### Issue: Streamlit not found
```bash
# Make sure you're in the right environment
pip install streamlit --upgrade
```

## Requirements Files Reference

- **requirements.txt** - Basic installation (PaddleOCR + Web UI)
- **requirements_complete.txt** - Full installation (includes DeepSeek)
- **requirements_frozen.txt** - Exact versions from working environment
- **web_ui/requirements.txt** - Documentation only (now in main requirements.txt)

## System Requirements

- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk**: 2GB for basic installation, 10GB+ for all models
- **GPU**: Optional but recommended for training (CUDA 11.8+)

## Next Steps

After installation:
1. Read [README.md](README.md) for usage examples
2. Check [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for fine-tuning
3. Run the web UI: `streamlit run web_ui/app.py`
4. See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
