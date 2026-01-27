# VIN OCR Training Guide

This guide explains how to train the PaddleOCR VIN recognition model for your specific dataset.

## Training Methods Overview

| Method | When to Use | Speed | Accuracy Improvement |
|--------|-------------|-------|---------------------|
| **Rule-Based** (`rules`) | Quick improvements, <1000 images | Fast (minutes) | 5-15% |
| **Neural Fine-Tuning** (`finetune`) | Production systems, 1000+ images | Slow (hours) | 20-40% |

---

## Method 1: Rule-Based Learning (Quick Start)

Best for:
- Quick accuracy improvements
- Limited training data (<1000 images)
- CPU-only environments
- Rapid experimentation

```bash
# Basic usage
python train_pipeline.py --dataset-dir ./data --method rules

# With verbose output
python train_pipeline.py --dataset-dir ./data --method rules

# Output: Learned correction rules saved to training_output/
```

**How it works:**
1. Runs pretrained OCR on training images
2. Compares predictions to ground truth
3. Learns character confusion patterns (e.g., "O" often misread as "0")
4. Generates deterministic correction rules

---

## Method 2: Neural Network Fine-Tuning (Production)

**Best for:**
- Large datasets (1,000+ images, recommended 10,000+)
- Maximum accuracy requirements
- Production VIN recognition systems
- GPU-equipped machines

### Prerequisites

1. **GPU Setup** (Recommended):
```bash
# Uninstall CPU version
pip uninstall paddlepaddle

# Install GPU version (CUDA 11.x)
pip install paddlepaddle-gpu

# Verify GPU
python -c "import paddle; print(paddle.device.get_device())"
```

2. **Dataset Format**:
```
data/
├── images/
│   ├── 1G1YY22G965104876.jpg
│   ├── WVWZZZ3CZWE123456.png
│   └── ...
└── train_labels.txt  # Format: image_path\tVIN_LABEL
```

Example `train_labels.txt`:
```
images/1G1YY22G965104876.jpg	1G1YY22G965104876
images/WVWZZZ3CZWE123456.png	WVWZZZ3CZWE123456
```

### Running Fine-Tuning

```bash
# Basic fine-tuning (uses config defaults)
python train_pipeline.py --dataset-dir ./data --method finetune

# Full configuration
python train_pipeline.py \
    --dataset-dir ./data \
    --method finetune \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --output-dir ./output/vin_model

# CPU training (slower)
python train_pipeline.py --dataset-dir ./data --method finetune --no-gpu
```

### Direct Fine-Tuning Script

For more control, use the dedicated fine-tuning script:

```bash
# Using default config
python finetune_paddleocr.py --config configs/vin_finetune_config.yml

# With custom settings
python finetune_paddleocr.py \
    --config configs/vin_finetune_config.yml \
    --epochs 150 \
    --batch-size 32 \
    --lr 0.00005 \
    --output ./my_model

# Resume interrupted training
python finetune_paddleocr.py \
    --config configs/vin_finetune_config.yml \
    --resume ./output/vin_rec_finetune/latest
```

### Multi-GPU Training

For large datasets (10,000+ images), use distributed training:

```bash
# 4 GPUs
python -m paddle.distributed.launch --gpus '0,1,2,3' \
    finetune_paddleocr.py --config configs/vin_finetune_config.yml

# 2 GPUs with larger batch size
python -m paddle.distributed.launch --gpus '0,1' \
    finetune_paddleocr.py \
    --config configs/vin_finetune_config.yml \
    --batch-size 128
```

---

## Training Configuration

### YAML Configuration (`configs/vin_finetune_config.yml`)

Key parameters to tune:

```yaml
Global:
  epoch_num: 100          # More epochs = better accuracy (up to a point)
  use_gpu: true
  use_amp: true           # Mixed precision (faster training)
  
Train:
  loader:
    batch_size_per_card: 64  # Reduce if OOM errors
    num_workers: 4           # Increase for faster data loading
    
Optimizer:
  lr:
    learning_rate: 0.0001    # Lower = more stable, higher = faster convergence
    warmup_epoch: 5          # Gradual LR increase at start
```

### Recommended Settings by Dataset Size

| Dataset Size | Epochs | Batch Size | Learning Rate |
|--------------|--------|------------|---------------|
| 1,000-5,000 | 150 | 32 | 0.0001 |
| 5,000-20,000 | 100 | 64 | 0.0001 |
| 20,000-100,000 | 80 | 128 | 0.00005 |
| 100,000+ | 50 | 256 | 0.00005 |

---

## Using Your Trained Model

### After Fine-Tuning

```python
from paddleocr import PaddleOCR
from vin_pipeline import VINOCRPipeline

# Option 1: Use inference model directly
ocr = PaddleOCR(
    rec_model_dir='./output/model/inference',
    rec_char_dict_path='./configs/vin_dict.txt',
    use_gpu=True
)

# Option 2: Use with VINOCRPipeline (recommended)
pipeline = VINOCRPipeline(
    use_gpu=True,
    # Custom model path
    rec_model_dir='./output/model/inference'
)

result = pipeline.process_image('test_vin.jpg')
print(result['vin'])
```

---

## Troubleshooting

### Out of Memory (OOM) Errors
```bash
# Reduce batch size
--batch-size 16

# Enable gradient checkpointing (in config)
# use_amp: true
```

### Poor Accuracy
1. Check dataset quality (blurry images, incorrect labels)
2. Increase epochs
3. Try lower learning rate (0.00005)
4. Add more diverse training data

### Training Too Slow
1. Use GPU: `pip install paddlepaddle-gpu`
2. Increase batch size (if memory allows)
3. Reduce `num_workers` if I/O bound
4. Use mixed precision: `use_amp: true`

### Model Not Improving
1. Check for data leakage between train/val
2. Verify labels are correct
3. Try different learning rate
4. Increase training data diversity

---

## Training Metrics

During training, watch these metrics:

| Metric | Good Value | Action if Bad |
|--------|------------|---------------|
| Train Loss | Decreasing | Training is working |
| Val Loss | Stable/Decreasing | No overfitting |
| Val Accuracy | 90%+ for VINs | Increase epochs/data |
| LR | Scheduled decay | Normal behavior |

Output files after training:
```
output/
├── model/
│   ├── best_accuracy.pdparams  # Best checkpoint
│   ├── latest.pdparams         # Latest checkpoint
│   └── inference/              # Export for deployment
│       ├── inference.pdmodel
│       ├── inference.pdiparams
│       └── vin_dict.txt
├── runtime_config.yml          # Config used
└── logs/                       # Training logs
```

---

## Quick Reference

```bash
# Rule-based (fast)
python train_pipeline.py --dataset-dir ./data --method rules

# Neural fine-tuning (accurate)
python train_pipeline.py --dataset-dir ./data --method finetune --epochs 100

# Full fine-tuning with all options
python train_pipeline.py \
    --dataset-dir ./data \
    --method finetune \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --output-dir ./my_model

# Resume training
python finetune_paddleocr.py --config configs/vin_finetune_config.yml --resume ./output/latest

# Multi-GPU
python -m paddle.distributed.launch --gpus '0,1' finetune_paddleocr.py --config configs/vin_finetune_config.yml
```
