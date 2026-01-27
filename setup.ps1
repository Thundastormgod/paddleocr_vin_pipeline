# Quick Setup Script for Windows
# ===============================
# This script sets up the complete environment for the VIN Pipeline

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PaddleOCR VIN Pipeline - Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host $pythonVersion

if ($pythonVersion -notmatch "Python 3\.1[1-9]") {
    Write-Host "Warning: Python 3.11+ recommended. Current version may not be compatible." -ForegroundColor Red
}

Write-Host ""
Write-Host "Choose installation method:" -ForegroundColor Yellow
Write-Host "1. Full installation with GPU support (recommended for training)"
Write-Host "2. CPU-only installation (lightweight, inference only)"
Write-Host "3. Development installation (includes testing tools)"
$choice = Read-Host "Enter choice (1-3)"

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install based on choice
switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Installing full environment with GPU support..." -ForegroundColor Green
        
        # Install PaddlePaddle GPU
        Write-Host "Installing PaddlePaddle GPU..." -ForegroundColor Yellow
        pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
        
        # Install other dependencies
        Write-Host "Installing remaining dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
        
        # Verify GPU
        Write-Host ""
        Write-Host "Verifying GPU support..." -ForegroundColor Yellow
        python -c "import paddle; print('PaddlePaddle version:', paddle.__version__); print('GPU available:', paddle.device.cuda.device_count() > 0)"
    }
    "2" {
        Write-Host ""
        Write-Host "Installing CPU-only environment..." -ForegroundColor Green
        pip install -r requirements.txt
    }
    "3" {
        Write-Host ""
        Write-Host "Installing development environment..." -ForegroundColor Green
        pip install -r requirements.txt
        pip install playwright pytest-playwright
    }
    default {
        Write-Host "Invalid choice. Installing basic environment..." -ForegroundColor Red
        pip install -r requirements.txt
    }
}

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c "import paddle; print('[OK] PaddlePaddle:', paddle.__version__)"
python -c "import paddleocr; print('[OK] PaddleOCR installed')"
python -c "import streamlit; print('[OK] Streamlit:', streamlit.__version__)"
python -c "import cv2; print('[OK] OpenCV:', cv2.__version__)"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the web UI:" -ForegroundColor Cyan
Write-Host "  streamlit run web_ui/app.py" -ForegroundColor White
Write-Host ""
Write-Host "To activate this environment later:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
