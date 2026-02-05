#!/bin/bash
# Quick Setup Script for Linux/Mac
# =================================
# This script sets up the complete environment for the VIN Pipeline

echo "========================================"
echo "PaddleOCR VIN Pipeline - Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

echo ""
echo "Choose installation method:"
echo "1. Full installation with GPU support (recommended for training)"
echo "2. CPU-only installation (lightweight, inference only)"
echo "3. Development installation (includes testing tools)"
read -p "Enter choice (1-3): " choice

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install based on choice
case $choice in
    1)
        echo ""
        echo "Installing full environment with GPU support..."
        
        # Install PaddlePaddle GPU
        echo "Installing PaddlePaddle GPU..."
        pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
        
        # Install other dependencies
        echo "Installing remaining dependencies..."
        pip install -r requirements.txt
        
        # Verify GPU
        echo ""
        echo "Verifying GPU support..."
        python -c "import paddle; print('PaddlePaddle version:', paddle.__version__); print('GPU available:', paddle.device.cuda.device_count() > 0)"
        ;;
    2)
        echo ""
        echo "Installing CPU-only environment..."
        pip install -r requirements.txt
        ;;
    3)
        echo ""
        echo "Installing development environment..."
        pip install -r requirements.txt
        pip install playwright pytest-playwright
        ;;
    *)
        echo "Invalid choice. Installing basic environment..."
        pip install -r requirements.txt
        ;;
esac

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import paddle; print('✓ PaddlePaddle:', paddle.__version__)"
python -c "import paddleocr; print('✓ PaddleOCR installed')"
python -c "import streamlit; print('✓ Streamlit:', streamlit.__version__)"
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)"

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To run the web UI:"
echo "  streamlit run src/vin_ocr/web/app.py"
echo ""
echo "To activate this environment later:"
echo "  source venv/bin/activate"
echo ""
