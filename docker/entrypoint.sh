#!/bin/bash
# =============================================================================
# VIN OCR Pipeline - Docker Entrypoint Script
# =============================================================================
# Handles runtime device detection and environment setup
# Automatically detects available hardware: CUDA GPU, Apple MPS, or CPU
# =============================================================================

set -e

echo "=============================================="
echo "VIN OCR Pipeline - Starting Container"
echo "=============================================="

# -----------------------------------------------------------------------------
# Hardware Detection
# -----------------------------------------------------------------------------

detect_device() {
    echo "Detecting available hardware..."
    
    # Check for NVIDIA GPU (CUDA)
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1)
            GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
            echo "✅ NVIDIA GPU detected: $GPU_NAME (${GPU_MEMORY}MB)"
            export VIN_OCR_DEVICE="gpu"
            export PADDLE_DEVICE="gpu"
            return 0
        fi
    fi
    
    # Check for CUDA through Python/PyTorch
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo "✅ CUDA GPU detected via PyTorch: $GPU_NAME"
        export VIN_OCR_DEVICE="gpu"
        export PADDLE_DEVICE="gpu"
        return 0
    fi
    
    # Check for Apple MPS (Metal Performance Shaders)
    if python -c "import torch; exit(0 if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        echo "✅ Apple MPS (Metal) detected"
        echo "   Note: PaddleOCR uses CPU, PyTorch models can use MPS"
        export VIN_OCR_DEVICE="mps"
        export PADDLE_DEVICE="cpu"  # PaddlePaddle doesn't support MPS
        return 0
    fi
    
    # Fallback to CPU
    echo "ℹ️  No GPU detected, using CPU"
    export VIN_OCR_DEVICE="cpu"
    export PADDLE_DEVICE="cpu"
    return 0
}

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

setup_environment() {
    echo ""
    echo "Setting up environment..."
    
    # Set PaddlePaddle device flags
    if [ "$PADDLE_DEVICE" = "gpu" ]; then
        export FLAGS_use_cuda=1
        export FLAGS_use_cudnn=1
        echo "  PaddlePaddle: GPU mode (CUDA)"
    else
        export FLAGS_use_cuda=0
        export FLAGS_use_cudnn=0
        echo "  PaddlePaddle: CPU mode"
    fi
    
    # Streamlit configuration
    export STREAMLIT_SERVER_HEADLESS=true
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    
    # Create directories if they don't exist
    mkdir -p /app/output /app/data /app/dataset /app/models /app/logs /app/finetune_data 2>/dev/null || true
    
    echo "  Working directory: $(pwd)"
    echo "  Device: $VIN_OCR_DEVICE"
}

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

validate_installation() {
    echo ""
    echo "Validating installation..."
    
    # Check PaddlePaddle
    if python -c "import paddle; print(f'  PaddlePaddle: {paddle.__version__}')" 2>/dev/null; then
        :
    else
        echo "  ⚠️  PaddlePaddle not found"
    fi
    
    # Check PaddleOCR
    if python -c "import paddleocr; print(f'  PaddleOCR: {paddleocr.__version__}')" 2>/dev/null; then
        :
    else
        echo "  ⚠️  PaddleOCR not found"
    fi
    
    # Check Streamlit
    if python -c "import streamlit; print(f'  Streamlit: {streamlit.__version__}')" 2>/dev/null; then
        :
    else
        echo "  ⚠️  Streamlit not found"
    fi
    
    # Check PyTorch (optional for DeepSeek)
    if python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null; then
        :
    else
        echo "  ℹ️  PyTorch not installed (optional, for DeepSeek-OCR)"
    fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    detect_device
    setup_environment
    validate_installation
    
    echo ""
    echo "=============================================="
    echo "Starting application..."
    echo "=============================================="
    echo ""
    
    # Execute the command passed to docker run
    exec "$@"
}

main "$@"
