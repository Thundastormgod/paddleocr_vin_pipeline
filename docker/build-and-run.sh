#!/bin/bash
# =============================================================================
# VIN OCR Pipeline - Docker Build & Run Script
# =============================================================================
# Automatically detects hardware and builds/runs the appropriate container
#
# Usage:
#   ./docker/build-and-run.sh          # Auto-detect and run
#   ./docker/build-and-run.sh cpu      # Force CPU mode
#   ./docker/build-and-run.sh gpu      # Force GPU mode (requires NVIDIA)
#   ./docker/build-and-run.sh build    # Build only (no run)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="vin-ocr"
CONTAINER_NAME="vin-ocr-app"
PORT=${PORT:-8501}

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BLUE}=============================================="
    echo -e "VIN OCR Pipeline - Docker Setup"
    echo -e "==============================================${NC}"
    echo ""
}

detect_gpu() {
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
            echo -e "${GREEN}✅ NVIDIA GPU detected: $GPU_INFO${NC}"
            return 0
        fi
    fi
    
    # Check for nvidia-docker
    if docker info 2>/dev/null | grep -q nvidia; then
        echo -e "${GREEN}✅ NVIDIA Docker runtime available${NC}"
        return 0
    fi
    
    return 1
}

build_cpu() {
    echo -e "${YELLOW}Building CPU image...${NC}"
    docker build -f docker/Dockerfile.cpu -t ${IMAGE_NAME}:cpu .
    echo -e "${GREEN}✅ CPU image built: ${IMAGE_NAME}:cpu${NC}"
}

build_gpu() {
    echo -e "${YELLOW}Building GPU image...${NC}"
    docker build -f docker/Dockerfile.gpu -t ${IMAGE_NAME}:gpu .
    echo -e "${GREEN}✅ GPU image built: ${IMAGE_NAME}:gpu${NC}"
}

build_universal() {
    echo -e "${YELLOW}Building universal image...${NC}"
    docker build -t ${IMAGE_NAME}:latest .
    echo -e "${GREEN}✅ Universal image built: ${IMAGE_NAME}:latest${NC}"
}

run_cpu() {
    echo -e "${YELLOW}Starting CPU container...${NC}"
    
    # Stop existing container if running
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8501 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/dataset:/app/dataset" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/finetune_data:/app/finetune_data" \
        -v "$(pwd)/configs:/app/configs:ro" \
        -e VIN_OCR_DEVICE=cpu \
        ${IMAGE_NAME}:cpu
    
    echo -e "${GREEN}✅ Container started: ${CONTAINER_NAME}${NC}"
    echo -e "${BLUE}   Access the app at: http://localhost:${PORT}${NC}"
}

run_gpu() {
    echo -e "${YELLOW}Starting GPU container...${NC}"
    
    # Stop existing container if running
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    
    docker run -d \
        --name ${CONTAINER_NAME} \
        --gpus all \
        -p ${PORT}:8501 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/dataset:/app/dataset" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/finetune_data:/app/finetune_data" \
        -v "$(pwd)/configs:/app/configs:ro" \
        -e VIN_OCR_DEVICE=gpu \
        -e NVIDIA_VISIBLE_DEVICES=all \
        ${IMAGE_NAME}:gpu
    
    echo -e "${GREEN}✅ Container started with GPU: ${CONTAINER_NAME}${NC}"
    echo -e "${BLUE}   Access the app at: http://localhost:${PORT}${NC}"
}

show_logs() {
    echo -e "${YELLOW}Container logs:${NC}"
    docker logs -f ${CONTAINER_NAME}
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  (none)    Auto-detect hardware and build+run"
    echo "  cpu       Force CPU mode"
    echo "  gpu       Force GPU mode (requires NVIDIA GPU)"
    echo "  build     Build images only (no run)"
    echo "  stop      Stop running container"
    echo "  logs      Show container logs"
    echo "  help      Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  PORT      Port to expose (default: 8501)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Auto-detect and run"
    echo "  $0 cpu                # Build and run CPU version"
    echo "  $0 gpu                # Build and run GPU version"
    echo "  PORT=9000 $0 cpu      # Run on port 9000"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

print_header

case "${1:-auto}" in
    cpu)
        build_cpu
        run_cpu
        ;;
    gpu)
        if detect_gpu; then
            build_gpu
            run_gpu
        else
            echo -e "${RED}❌ No NVIDIA GPU detected. Use 'cpu' mode instead.${NC}"
            exit 1
        fi
        ;;
    build)
        build_universal
        if detect_gpu; then
            build_gpu
        fi
        build_cpu
        echo -e "${GREEN}✅ All images built successfully${NC}"
        ;;
    stop)
        echo -e "${YELLOW}Stopping container...${NC}"
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
        echo -e "${GREEN}✅ Container stopped${NC}"
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    auto|*)
        # Auto-detect
        if detect_gpu; then
            echo -e "${GREEN}GPU mode selected${NC}"
            build_gpu
            run_gpu
        else
            echo -e "${YELLOW}No GPU detected, using CPU mode${NC}"
            build_cpu
            run_cpu
        fi
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
