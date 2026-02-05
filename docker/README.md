# Docker Deployment Guide

This guide explains how to deploy the VIN OCR Pipeline using Docker with automatic hardware detection.

## Quick Start

### Option 1: Auto-Detect Hardware (Recommended)

```bash
# Make the script executable
chmod +x docker/build-and-run.sh

# Auto-detect hardware and run
./docker/build-and-run.sh
```

### Option 2: Docker Compose

```bash
# CPU only
docker-compose up vin-ocr-cpu

# GPU (requires NVIDIA Docker)
docker-compose up vin-ocr-gpu
```

### Option 3: Manual Docker Commands

```bash
# Build universal image
docker build -t vin-ocr .

# Run
docker run -p 8501:8501 vin-ocr
```

## Available Docker Images

| Image | Dockerfile | Use Case |
|-------|------------|----------|
| `vin-ocr:latest` | `Dockerfile` | Universal, auto-detects hardware |
| `vin-ocr:cpu` | `docker/Dockerfile.cpu` | CPU-only deployments |
| `vin-ocr:gpu` | `docker/Dockerfile.gpu` | NVIDIA GPU with CUDA |

## Hardware Support

### CPU (All Platforms)
- Works on any machine (Linux, macOS, Windows)
- No special requirements
- Slower inference speed

### NVIDIA GPU (Linux)
- Requires NVIDIA GPU with CUDA support
- Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Fastest inference speed

```bash
# Install NVIDIA Container Toolkit (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Apple Silicon (M1/M2/M3)
- PaddleOCR runs on CPU (MPS not supported by PaddlePaddle)
- PyTorch-based models (DeepSeek) can use MPS
- Use `vin-ocr:cpu` image

## Volume Mounts

The following directories are mounted to persist data:

| Local Path | Container Path | Purpose |
|------------|----------------|---------|
| `./data` | `/app/data` | Input images |
| `./dataset` | `/app/dataset` | Training datasets |
| `./output` | `/app/output` | Training output |
| `./models` | `/app/models` | Trained models |
| `./finetune_data` | `/app/finetune_data` | Fine-tuning data |
| `./configs` | `/app/configs` | Configuration (read-only) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIN_OCR_DEVICE` | auto | Device: `cpu`, `gpu`, or `mps` |
| `PADDLE_DEVICE` | auto | PaddlePaddle device |
| `STREAMLIT_SERVER_PORT` | 8501 | Web UI port |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU index (for multi-GPU) |

## Commands

### Build Images

```bash
# Build all images
./docker/build-and-run.sh build

# Build specific image
docker build -f docker/Dockerfile.cpu -t vin-ocr:cpu .
docker build -f docker/Dockerfile.gpu -t vin-ocr:gpu .
```

### Run Container

```bash
# CPU mode
./docker/build-and-run.sh cpu

# GPU mode
./docker/build-and-run.sh gpu

# Custom port
PORT=9000 ./docker/build-and-run.sh cpu
```

### Manage Container

```bash
# View logs
./docker/build-and-run.sh logs
# or
docker logs -f vin-ocr-app

# Stop container
./docker/build-and-run.sh stop
# or
docker stop vin-ocr-app

# Shell into container
docker exec -it vin-ocr-app /bin/bash
```

## Training in Docker

### Using Docker Compose

```bash
# Start training service
docker-compose --profile training run vin-ocr-training \
    python -m src.vin_ocr.training.finetune_paddleocr \
    --epochs 50 \
    --batch-size 32 \
    --output /app/output/training
```

### Using Docker Run

```bash
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/finetune_data:/app/finetune_data \
    vin-ocr:gpu \
    python -m src.vin_ocr.training.finetune_paddleocr \
    --epochs 50 --batch-size 32
```

## Production Deployment

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml vin-ocr
```

### Kubernetes

See `kubernetes/` directory for Kubernetes manifests (if available).

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs vin-ocr-app

# Check if port is in use
lsof -i :8501
```

### GPU not detected

```bash
# Verify NVIDIA Docker is installed
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check Docker runtime
docker info | grep nvidia
```

### Out of memory

- Reduce batch size in training
- Use CPU mode for inference
- Use QLoRA for DeepSeek fine-tuning

### Permission denied on mounted volumes

```bash
# Fix ownership
sudo chown -R $USER:$USER ./data ./output ./models
```

## Health Check

The container includes a health check that verifies Streamlit is running:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' vin-ocr-app
```

## Security Notes

- Container runs as non-root user (`vinuser`)
- Configs are mounted read-only
- No sensitive data in image
- Use Docker secrets for API keys in production
