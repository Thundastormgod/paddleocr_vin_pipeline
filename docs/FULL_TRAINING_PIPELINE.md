# VIN OCR Training Pipeline - Complete Reference

> **Comprehensive documentation for fine-tuning PaddleOCR and DeepSeek-OCR models for Vehicle Identification Number (VIN) recognition.**

**Version:** 2.0  
**Last Updated:** January 2026  
**Author:** JRL-VIN Project

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Data Preparation](#data-preparation)
5. [Training PaddleOCR](#training-paddleocr)
6. [Training DeepSeek-OCR](#training-deepseek-ocr)
7. [Unified Training Interface](#unified-training-interface)
8. [Configuration Reference](#configuration-reference)
9. [Monitoring & Logging](#monitoring--logging)
10. [Model Export & Deployment](#model-export--deployment)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

---

## Overview

This training pipeline provides two complementary approaches to VIN OCR:

| Model | Framework | Approach | Best For |
|-------|-----------|----------|----------|
| **PaddleOCR** | PaddlePaddle | CTC-based recognition | Production deployment, speed |
| **DeepSeek-OCR** | PyTorch/Transformers | Vision-Language model with LoRA | Accuracy, complex conditions |

### Key Features

- ✅ **Production-Ready Architecture** - PP-OCRv4 compatible weights
- ✅ **Memory-Efficient Training** - LoRA, QLoRA, gradient checkpointing
- ✅ **Mixed Precision** - AMP/bf16/fp16 support
- ✅ **Distributed Training** - Multi-GPU support
- ✅ **Checkpoint Management** - Save/resume training
- ✅ **VIN-Specific Augmentation** - Optimized for engraved plates

### Training Infrastructure Status

| Component | Status | Notes |
|-----------|--------|-------|
| PaddleOCR Training | ✅ Ready | `PPOCR_TRAIN_AVAILABLE=True` |
| DeepSeek Training | ✅ Ready | `PEFT_AVAILABLE=True` |
| Data Preparation | ✅ Ready | Script at `scripts/prepare_finetune_data.py` |
| Configuration | ✅ Complete | Both YAML configs present |
| Character Dict | ✅ Correct | 33 VIN-valid characters |
| Export/Inference | ✅ Implemented | Static graph export for PaddleOCR |

---

## Architecture

### Training Pipeline Structure

```
train_vin_model.py (Unified Entry Point - 301 lines)
│
├── finetune_paddleocr.py (1,115 lines)
│   ├── VINRecognitionModel (PP-OCRv4 architecture)
│   │   ├── PPLCNetV3Backbone (MobileNet-like with SE blocks)
│   │   ├── SVTREncoder (2-layer Transformer neck)
│   │   └── CTCHead (CTC output layer)
│   ├── VINRecognitionDataset (Custom dataset with augmentation)
│   └── VINFineTuner (Training loop with AMP support)
│
├── finetune_deepseek.py (637 lines)
│   ├── DeepSeekVINTrainer (HuggingFace Trainer wrapper)
│   ├── VINDeepSeekDataset (Vision-language dataset)
│   └── LoRA/QLoRA configuration (PEFT integration)
│
└── scripts/prepare_finetune_data.py (445 lines)
    ├── VIN extraction from filenames
    ├── Image preprocessing (resize, CLAHE)
    └── Train/val split generation
```

### Model Architecture Comparison

#### PaddleOCR (PP-OCRv4)

```
Input Image [B, 3, 48, 320]
         │
         ▼
┌─────────────────────────────┐
│   PPLCNetV3 Backbone        │  ← Depthwise separable convs + SE blocks
│   - Stem: Conv2D + BN       │
│   - 11 DSConv stages        │
│   - Output: 512 channels    │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   SVTR Transformer Neck     │  ← Sequence modeling
│   - AdaptiveAvgPool (H→1)   │
│   - Linear projection       │
│   - Positional encoding     │
│   - 2× TransformerEncoder   │
│   - Output: 256 dim         │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   CTC Head                  │  ← Classification
│   - FC1: 256 → 256          │
│   - Dropout: 0.1            │
│   - FC2: 256 → 33 classes   │
└─────────────────────────────┘
         │
         ▼
Output Logits [B, T, 33]
```

#### DeepSeek-OCR with LoRA

```
Input Image + Prompt
         │
         ▼
┌─────────────────────────────┐
│   Vision Encoder            │
│   (Frozen during LoRA)      │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Language Model Layers     │  ← LoRA adapters applied to:
│   + LoRA Adapters           │
│                             │     Attention:
│   Config:                   │     - q_proj (query)
│   - r = 16 (rank)           │     - k_proj (key)
│   - α = 32 (scaling)        │     - v_proj (value)
│   - dropout = 0.05          │     - o_proj (output)
│                             │
│                             │     FFN:
│                             │     - gate_proj
│                             │     - up_proj
│                             │     - down_proj
└─────────────────────────────┘
         │
         ▼
Output Text (VIN)

Trainable Parameters: ~1% of total
Memory Reduction: 50-75%
```

---

## Prerequisites

### Hardware Requirements

| Training Type | GPU Memory | Recommended GPU | Training Time (10k images) |
|--------------|------------|-----------------|---------------------------|
| PaddleOCR (full) | 8-16 GB | RTX 3080/4080 | ~4-8 hours |
| DeepSeek + LoRA + 8-bit | ~16 GB | RTX 4090 | ~6-10 hours |
| DeepSeek + LoRA + bf16 | ~24 GB | A100 40GB | ~4-6 hours |
| DeepSeek full fine-tune | ~48 GB | A100 80GB | ~12-24 hours |

### Software Requirements

```bash
# Core dependencies (already installed)
pip install paddlepaddle-gpu>=2.5.0  # or paddlepaddle for CPU
pip install paddleocr>=2.7.0
pip install torch>=2.0
pip install transformers>=4.36
pip install peft>=0.6

# Optional (for quantization)
pip install bitsandbytes  # 8-bit/4-bit quantization

# Verify installation
python -c "
from finetune_paddleocr import PADDLE_AVAILABLE, PPOCR_TRAIN_AVAILABLE
from finetune_deepseek import TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, PEFT_AVAILABLE
print(f'PaddlePaddle: {PADDLE_AVAILABLE}')
print(f'PPOCR Training: {PPOCR_TRAIN_AVAILABLE}')
print(f'PyTorch: {TORCH_AVAILABLE}')
print(f'Transformers: {TRANSFORMERS_AVAILABLE}')
print(f'PEFT (LoRA): {PEFT_AVAILABLE}')
"
```

**Current Status (Verified):**
```
PaddlePaddle: True
PPOCR Training: True
PyTorch: True
Transformers: True
PEFT (LoRA): True
```

### Dataset Requirements

For optimal fine-tuning results:

| Dataset Size | Expected Accuracy | Training Time | Recommendation |
|-------------|-------------------|---------------|----------------|
| 1,000 images | ~85% | ~1 hour | Minimum viable |
| 5,000 images | ~92% | ~4 hours | Good for testing |
| **11,000+ images** | **~97%+** | ~8 hours | **Production recommended** |
| 50,000 images | ~99% | ~24 hours | Enterprise grade |

---

## Data Preparation

### Step 1: Organize Raw Images

Place VIN images in the `./data/` directory with VINs in filenames:

```
data/
├── 1-VIN-SAL1A2A40SA606662.jpg
├── 2-VIN-WVWZZZ3CZWE123456.jpg
├── vehicle_5N1AR2MN9JC123456_front.png
└── ...
```

**Supported filename patterns:**
- `*-VIN-{VIN}.*` → e.g., `1-VIN-SAL1A2A40SA606662.jpg`
- `*_{VIN}_*.*` → e.g., `car_SAL1A2A40SA606662_photo.jpg`
- `*-{VIN}.*` → e.g., `scan-SAL1A2A40SA606662.png`

### Step 2: Run Data Preparation Script

```bash
python scripts/prepare_finetune_data.py \
    --input-dir ./data \
    --output-dir ./finetune_data \
    --train-ratio 0.9 \
    --val-ratio 0.1 \
    --apply-clahe \
    --target-height 48 \
    --max-width 320

# Output structure:
# finetune_data/
# ├── train_labels.txt    # image_path\tVIN_LABEL
# ├── val_labels.txt
# └── images/
#     ├── train/
#     └── val/
```

### Step 3: Verify Data Format

```bash
# Check label files
head -5 ./finetune_data/train_labels.txt

# Expected format (tab-separated):
# images/train/img_001.jpg	SAL1A2A40SA606662
# images/train/img_002.jpg	WVWZZZ3CZWE123456

# Count samples
wc -l ./finetune_data/train_labels.txt
wc -l ./finetune_data/val_labels.txt
```

### VIN Character Set

The training uses a 33-character vocabulary (excluding I, O, Q which are invalid in VINs):

```
# configs/vin_dict.txt
0 1 2 3 4 5 6 7 8 9
A B C D E F G H J K L M N P R S T U V W X Y Z
```

**Note:** Characters I, O, Q are excluded per ISO 3779 VIN standard to avoid confusion with 1, 0, and other characters.

---

## Training PaddleOCR

### Quick Start

```bash
# Basic training
python train_vin_model.py --model paddleocr

# With custom parameters
python train_vin_model.py \
    --model paddleocr \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0005 \
    --output ./output/my_vin_model
```

### Direct Script Usage

```bash
# Using finetune_paddleocr.py directly
python finetune_paddleocr.py \
    --config configs/vin_finetune_config.yml \
    --epochs 100

# Resume from checkpoint
python finetune_paddleocr.py \
    --config configs/vin_finetune_config.yml \
    --resume output/vin_rec_finetune/latest

# Multi-GPU training
python -m paddle.distributed.launch --gpus '0,1,2,3' \
    finetune_paddleocr.py --config configs/vin_finetune_config.yml

# CPU training (slower)
python finetune_paddleocr.py \
    --config configs/vin_finetune_config.yml \
    --cpu
```

### Configuration Details

**`configs/vin_finetune_config.yml`** (137 lines):

```yaml
Global:
  debug: false
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/vin_rec_finetune
  save_epoch_step: 5
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: null  # Set to load pretrained PP-OCRv4
  checkpoints: null
  save_inference_dir: ./inference/vin_rec
  use_visualdl: true
  visualdl_log_dir: ./output/vdl_logs
  character_dict_path: ./configs/vin_dict.txt
  max_text_length: 17
  use_space_char: false
  distributed: false

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0005
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 0.00001

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform: null
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 120
            depth: 2
            hidden_dims: 120
            kernel_size: [1, 3]
            use_guide: true
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: 17

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./finetune_data/
    label_file_list:
      - ./finetune_data/train_labels.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecConAug:
          prob: 0.5
          ext_data_num: 2
          image_shape: [48, 320, 3]
          max_text_length: 17
      - RecAug:
      - MultiLabelEncode:
          max_text_length: 17
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_sar
            - length
            - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: 64
    drop_last: true
    num_workers: 8
    use_shared_memory: true

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./finetune_data/
    label_file_list:
      - ./finetune_data/val_labels.txt
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 128
    num_workers: 4
```

### Training Output Structure

```
output/vin_rec_finetune/
├── epoch_5.pdparams       # Model weights at epoch 5
├── epoch_5.pdopt          # Optimizer state at epoch 5
├── epoch_5_info.json      # Checkpoint metadata
├── epoch_10.pdparams
├── ...
├── latest.pdparams        # Latest checkpoint (for resume)
├── latest.pdopt
├── best_accuracy.pdparams # Best model by validation accuracy
└── inference/             # Exported inference model
    ├── inference.pdmodel      # Model structure
    ├── inference.pdiparams    # Model weights
    └── vin_dict.txt           # Character dictionary
```

---

## Training DeepSeek-OCR

### Quick Start with LoRA

```bash
# LoRA training (recommended - memory efficient)
python train_vin_model.py --model deepseek --lora

# With custom parameters
python train_vin_model.py \
    --model deepseek \
    --lora \
    --epochs 10 \
    --batch-size 4 \
    --lr 2e-5 \
    --output ./output/my_deepseek_vin
```

### Direct Script Usage

```bash
# LoRA fine-tuning (default)
python finetune_deepseek.py \
    --config configs/deepseek_finetune_config.yml \
    --lora

# Full fine-tuning (requires 48GB+ VRAM)
python finetune_deepseek.py \
    --config configs/deepseek_finetune_config.yml \
    --full

# Resume training
python finetune_deepseek.py \
    --config configs/deepseek_finetune_config.yml \
    --resume output/deepseek_vin/checkpoint-1000
```

### Configuration Details

**`configs/deepseek_finetune_config.yml`** (67 lines):

```yaml
# DeepSeek-OCR Fine-Tuning Configuration for VIN Recognition
# ===========================================================
#
# GPU Memory Requirements:
#   - LoRA + 8-bit: ~16GB VRAM
#   - LoRA + bf16: ~24GB VRAM  
#   - Full fine-tuning: ~48GB VRAM

# Model Configuration
model_name: "deepseek-ai/DeepSeek-OCR"

# Output
output_dir: "./output/deepseek_vin_finetune"

# Training Hyperparameters
num_epochs: 10
batch_size: 4
gradient_accumulation_steps: 8  # Effective batch size = 4 * 8 = 32
learning_rate: 2.0e-5
weight_decay: 0.01
warmup_ratio: 0.1
max_grad_norm: 1.0

# LoRA Configuration (Memory-Efficient Fine-Tuning)
use_lora: true
lora_r: 16              # LoRA rank (higher = more capacity, more memory)
lora_alpha: 32          # LoRA alpha (scaling factor, typically 2x rank)
lora_dropout: 0.05      # Dropout for regularization
lora_target_modules:    # Modules to apply LoRA
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"

# Quantization (for lower memory usage)
use_8bit: false         # 8-bit quantization (requires bitsandbytes)
use_4bit: false         # 4-bit quantization (QLoRA, requires bitsandbytes)

# Data Paths
train_data_path: "./finetune_data/train_labels.txt"
val_data_path: "./finetune_data/val_labels.txt"
data_dir: "./finetune_data/"
max_length: 32          # Max token length for VIN output

# Precision
bf16: true              # BFloat16 (recommended for Ampere+ GPUs)
fp16: false             # Float16 (use if bf16 not supported)

# Logging and Checkpointing
logging_steps: 10
eval_steps: 100
save_steps: 500
save_total_limit: 3     # Keep only last N checkpoints

# Early Stopping
early_stopping_patience: 5
early_stopping_threshold: 0.001

# Seed
seed: 42
```

### LoRA Memory Comparison

| Configuration | VRAM Usage | Trainable Params | Speed |
|--------------|------------|------------------|-------|
| Full Fine-tune | ~48 GB | 100% (~7B) | Baseline |
| LoRA (r=16) + bf16 | ~24 GB | ~1% (~70M) | 2x faster |
| LoRA (r=16) + 8-bit | ~16 GB | ~1% (~70M) | 1.5x faster |
| QLoRA (r=16) + 4-bit | ~12 GB | ~1% (~70M) | 1.2x faster |

### Training Output Structure

```
output/deepseek_vin_finetune/
├── checkpoint-500/
│   ├── trainer_state.json
│   ├── optimizer.pt
│   └── scheduler.pt
├── checkpoint-1000/
├── checkpoint-1500/
├── final_model/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer/
├── lora_weights/          # LoRA adapter weights (if using LoRA)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── inference/             # Merged model for inference
    ├── config.json
    ├── model.safetensors
    └── tokenizer/
```

---

## Unified Training Interface

### `train_vin_model.py` - Single Entry Point

The unified interface simplifies training by providing a single command for both models:

```bash
# Train PaddleOCR
python train_vin_model.py --model paddleocr

# Train DeepSeek with LoRA
python train_vin_model.py --model deepseek --lora

# Train both models sequentially
python train_vin_model.py --model all

# Full options
python train_vin_model.py \
    --model paddleocr \
    --config configs/vin_finetune_config.yml \
    --output ./my_output \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --resume ./output/checkpoint
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model`, `-m` | str | `paddleocr` | Model to train: `paddleocr`, `deepseek`, or `all` |
| `--config`, `-c` | str | auto-detect | Path to config file |
| `--output`, `-o` | str | from config | Output directory |
| `--epochs` | int | from config | Number of training epochs |
| `--batch-size` | int | from config | Batch size per device |
| `--lr` | float | from config | Learning rate |
| `--resume`, `-r` | str | None | Checkpoint path to resume training |
| `--lora` | flag | True | Use LoRA for DeepSeek (default) |
| `--full` | flag | False | Full fine-tuning for DeepSeek (no LoRA) |

### Training Both Models

```bash
# Train both models sequentially
python train_vin_model.py --model all

# Output:
# Phase 1: Training PaddleOCR
# Phase 2: Training DeepSeek-OCR
# Summary with results for both
```

---

## Configuration Reference

### Hyperparameter Comparison

| Parameter | PaddleOCR | DeepSeek | Notes |
|-----------|-----------|----------|-------|
| **Epochs** | 100 | 10 | LoRA converges faster |
| **Batch Size** | 64 | 4 (×8 = 32 effective) | Limited by GPU memory |
| **Learning Rate** | 5e-4 | 2e-5 | VLMs need lower LR |
| **Warmup** | 5 epochs | 10% of steps | Stabilizes training |
| **Optimizer** | Adam | AdamW | Weight decay handling |
| **Precision** | AMP (optional) | bf16/fp16 | Mixed precision |
| **Loss** | CTC | Cross-Entropy | Different architectures |
| **Image Size** | 48×320 | Model-dependent | PP-OCRv4 standard |

### PaddleOCR Recommended Ranges

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| `epoch_num` | 100 | 50-200 | More for smaller datasets |
| `learning_rate` | 5e-4 | 1e-4 to 1e-3 | Lower for fine-tuning |
| `warmup_epoch` | 5 | 3-10 | Prevents early divergence |
| `batch_size` | 64 | 16-128 | Max by GPU memory |
| `weight_decay` | 1e-5 | 1e-6 to 1e-4 | L2 regularization |

### DeepSeek LoRA Recommended Ranges

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| `num_epochs` | 10 | 5-20 | LoRA converges quickly |
| `learning_rate` | 2e-5 | 1e-5 to 5e-5 | Too high causes instability |
| `lora_r` | 16 | 8-64 | Higher = more capacity |
| `lora_alpha` | 32 | 16-64 | Typically 2× lora_r |
| `gradient_accumulation` | 8 | 4-16 | Simulates larger batch |

---

## Monitoring & Logging

### TensorBoard (PaddleOCR via VisualDL)

```bash
# Start TensorBoard with VisualDL logs
tensorboard --logdir=./output/vdl_logs

# Or use VisualDL directly
pip install visualdl
visualdl --logdir=./output/vdl_logs --port 8080
```

### TensorBoard (DeepSeek)

```bash
# HuggingFace Trainer logs to TensorBoard by default
tensorboard --logdir=./output/deepseek_vin_finetune

# Open http://localhost:6006 in browser
```

### Key Metrics to Monitor

| Metric | PaddleOCR | DeepSeek | Target | Warning Threshold |
|--------|-----------|----------|--------|-------------------|
| **Accuracy** | `val_acc` | `eval_accuracy` | >95% | <80% after 50% training |
| **Loss** | `train_loss` | `loss` | <0.1 | >1.0 not decreasing |
| **CER** | N/A | `eval_cer` | <0.02 | >0.1 |
| **Learning Rate** | `lr` | `learning_rate` | Decay curve | Flat or increasing |

### Sample Training Log

```
============================================================
Starting VIN Recognition Fine-Tuning
  Epochs: 100
  Device: gpu
  Output: ./output/vin_rec_finetune
============================================================

Epoch [1] Batch [0/156] Loss: 2.3456 LR: 0.000005
Epoch [1] Batch [10/156] Loss: 1.8234 LR: 0.000010
Epoch [1] Batch [20/156] Loss: 1.4567 LR: 0.000015
...
Epoch [1/100] Train Loss: 1.2345 Val Loss: 1.1234 Val Acc: 0.7823 Time: 300.0s
Saved best model with accuracy: 0.7823

Epoch [2] Batch [0/156] Loss: 1.0123 LR: 0.000050
...
Epoch [2/100] Train Loss: 0.8765 Val Loss: 0.7654 Val Acc: 0.8567 Time: 295.0s
Saved best model with accuracy: 0.8567

...

============================================================
Training Complete!
  Total time: 8.25 hours
  Best accuracy: 0.9723
  Model saved to: ./output/vin_rec_finetune
============================================================
```

---

## Model Export & Deployment

### PaddleOCR Export

The training automatically exports an inference model. For manual export:

```python
from finetune_paddleocr import VINFineTuner, load_config

config = load_config('configs/vin_finetune_config.yml')
trainer = VINFineTuner(config, output_dir='./output/vin_rec_finetune')
trainer.export_inference_model()

# Output files:
# ./output/vin_rec_finetune/inference/
# ├── inference.pdmodel      # Model structure (static graph)
# ├── inference.pdiparams    # Model weights
# └── vin_dict.txt           # Character dictionary
```

### Using Exported PaddleOCR Model

```python
from paddleocr import PaddleOCR

# Use fine-tuned model for inference
ocr = PaddleOCR(
    rec_model_dir='./output/vin_rec_finetune/inference',
    rec_char_dict_path='./output/vin_rec_finetune/inference/vin_dict.txt',
    use_angle_cls=False
)

# Run OCR
result = ocr.ocr('vin_image.jpg', det=True, rec=True, cls=False)
vin = result[0][0][1][0]  # Extract recognized text
print(f"Recognized VIN: {vin}")
```

### DeepSeek Export

```python
from finetune_deepseek import DeepSeekVINTrainer, load_config

config = load_config('configs/deepseek_finetune_config.yml')
trainer = DeepSeekVINTrainer(config)

# Merge LoRA weights into base model for inference
trainer.export_for_inference(merge_lora=True)

# Output files:
# ./output/deepseek_vin_finetune/inference/
# ├── config.json
# ├── model.safetensors      # Merged model weights
# └── tokenizer/
```

### Using Exported DeepSeek Model

```python
from transformers import AutoModel, AutoProcessor
from PIL import Image

# Load fine-tuned model
model = AutoModel.from_pretrained(
    './output/deepseek_vin_finetune/inference',
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    './output/deepseek_vin_finetune/inference',
    trust_remote_code=True
)

# Inference
image = Image.open('vin_image.jpg')
inputs = processor(images=image, text="OCR this VIN:", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20)
vin = processor.decode(output[0], skip_special_tokens=True)
print(f"Recognized VIN: {vin}")
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
paddle.fluid.core_avx.EnforceNotMet: CUDA error(2)
```

**Solutions:**
```bash
# Option 1: Reduce batch size
python train_vin_model.py --model paddleocr --batch-size 32

# Option 2: Enable gradient checkpointing (DeepSeek - already default)
# In config: gradient_checkpointing: true

# Option 3: Use 8-bit quantization (DeepSeek)
# In deepseek_finetune_config.yml:
# use_8bit: true

# Option 4: Use 4-bit QLoRA (DeepSeek)
# use_4bit: true
```

#### 2. Training Data Not Found

**Symptoms:**
```
ERROR: Training data not found!
FileNotFoundError: ./finetune_data/train_labels.txt
```

**Solution:**
```bash
# Step 1: Prepare data
python scripts/prepare_finetune_data.py \
    --input-dir ./data \
    --output-dir ./finetune_data

# Step 2: Verify files exist
ls -la ./finetune_data/
# Should show: train_labels.txt, val_labels.txt, images/

# Step 3: Check file contents
head -3 ./finetune_data/train_labels.txt
```

#### 3. PPOCR_TRAIN_AVAILABLE = False

**Symptoms:**
```
WARNING: PaddleOCR training components not available
```

**Solution:**
```bash
# Clone PaddleOCR repository (already done)
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git

# Verify
python -c "from finetune_paddleocr import PPOCR_TRAIN_AVAILABLE; print(PPOCR_TRAIN_AVAILABLE)"
# Should print: True
```

#### 4. Low Accuracy (< 80%)

**Possible causes:**
- Insufficient training data (<1,000 images)
- Learning rate too high or too low
- Not enough epochs
- Poor image quality in dataset

**Solutions:**
```bash
# Option 1: More epochs
python train_vin_model.py --model paddleocr --epochs 200

# Option 2: Lower learning rate
python train_vin_model.py --model paddleocr --lr 0.0001

# Option 3: Apply CLAHE enhancement during data prep
python scripts/prepare_finetune_data.py \
    --input-dir ./data \
    --output-dir ./finetune_data \
    --apply-clahe

# Option 4: Increase dataset size (most effective)
# Add more labeled VIN images to ./data/
```

#### 5. NaN Loss During Training

**Symptoms:**
```
Epoch [5] Batch [10/156] Loss: nan
```

**Solutions:**
```bash
# Option 1: Lower learning rate significantly
python train_vin_model.py --model paddleocr --lr 0.00001

# Option 2: Disable AMP
# In vin_finetune_config.yml:
# use_amp: false

# Option 3: Add gradient clipping (DeepSeek - already default)
# max_grad_norm: 1.0
```

#### 6. Slow Training Speed

**Solutions:**
```bash
# Option 1: Increase number of workers
# In config: num_workers: 16

# Option 2: Enable shared memory
# use_shared_memory: true

# Option 3: Use mixed precision
# use_amp: true (PaddleOCR)
# bf16: true (DeepSeek)

# Option 4: Multi-GPU training
python -m paddle.distributed.launch --gpus '0,1' finetune_paddleocr.py
```

---

## Best Practices

### Data Quality Checklist

- [ ] **Minimum 11,000 images** for production quality
- [ ] **Diverse conditions**: Various lighting, angles, wear levels
- [ ] **Balanced distribution** across different VIN patterns/manufacturers
- [ ] **Verified labels** - Spot-check at least 100 random samples
- [ ] **Image resolution** - At least 200px width for VIN region
- [ ] **Clean filenames** - VIN clearly extractable from filename

### Training Strategy

1. **Start with PaddleOCR** - Faster iteration, easier debugging
2. **Use LoRA for DeepSeek** - Memory efficient, good accuracy
3. **Monitor validation accuracy** - Stop if overfitting (val loss increasing)
4. **Save checkpoints frequently** - Enable resume on failures
5. **Use early stopping** - Prevent wasted compute

### Hyperparameter Tuning Order

```bash
# 1. First, find good learning rate
for lr in 0.0001 0.0003 0.0005 0.001; do
    python train_vin_model.py --model paddleocr --lr $lr --epochs 20
done

# 2. Then optimize batch size (max that fits in memory)
# 3. Adjust epochs based on convergence curve
# 4. Fine-tune regularization (weight_decay)
```

### Production Deployment Checklist

- [ ] Validate exported model loads correctly
- [ ] Test on held-out test set (not validation set)
- [ ] Measure inference latency (target: <100ms per image)
- [ ] Compare accuracy with baseline model
- [ ] Document model version, training data, and hyperparameters
- [ ] Set up model versioning (MLflow, DVC, etc.)
- [ ] Create fallback to baseline if fine-tuned model fails

---

## File Reference

### Training Scripts

| File | Purpose | Lines | Dependencies |
|------|---------|-------|--------------|
| `train_vin_model.py` | Unified entry point | 301 | Both models |
| `finetune_paddleocr.py` | PaddleOCR training | 1,115 | PaddlePaddle, ppocr |
| `finetune_deepseek.py` | DeepSeek LoRA training | 637 | PyTorch, transformers, peft |
| `scripts/prepare_finetune_data.py` | Data preparation | 445 | OpenCV, vin_utils |

### Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| `configs/vin_finetune_config.yml` | PaddleOCR training config | YAML |
| `configs/deepseek_finetune_config.yml` | DeepSeek training config | YAML |
| `configs/vin_dict.txt` | Character dictionary | Plain text (33 chars) |

### Output Directories

| Directory | Contents |
|-----------|----------|
| `./output/vin_rec_finetune/` | PaddleOCR checkpoints and exports |
| `./output/deepseek_vin_finetune/` | DeepSeek checkpoints and exports |
| `./output/vdl_logs/` | VisualDL/TensorBoard logs |
| `./finetune_data/` | Prepared training data |

---

## Quick Reference Card

```bash
# ============================================================
# VIN OCR TRAINING - QUICK REFERENCE
# ============================================================

# 1. PREPARE DATA (Required first step)
python scripts/prepare_finetune_data.py \
    --input-dir ./data \
    --output-dir ./finetune_data

# 2. TRAIN PADDLEOCR (Production, Fast)
python train_vin_model.py --model paddleocr --epochs 100

# 3. TRAIN DEEPSEEK (Accuracy, LoRA)
python train_vin_model.py --model deepseek --lora --epochs 10

# 4. TRAIN BOTH
python train_vin_model.py --model all

# 5. RESUME TRAINING
python train_vin_model.py \
    --model paddleocr \
    --resume output/vin_rec_finetune/latest

# 6. MONITOR TRAINING
tensorboard --logdir=./output/vdl_logs

# 7. VERIFY INSTALLATION
python -c "
from finetune_paddleocr import PPOCR_TRAIN_AVAILABLE
from finetune_deepseek import PEFT_AVAILABLE
print(f'PaddleOCR Training: {PPOCR_TRAIN_AVAILABLE}')
print(f'DeepSeek+LoRA: {PEFT_AVAILABLE}')
"

# 8. RUN TESTS
python -m pytest tests/ -q
```

---

## Version History

### Version 2.0 (January 2026)
- Added DeepSeek-OCR with LoRA fine-tuning
- Created unified training interface (`train_vin_model.py`)
- Integrated local PaddleOCR repository
- Added comprehensive configuration files
- Full documentation rewrite

### Version 1.0 (December 2025)
- Initial PaddleOCR PP-OCRv4 fine-tuning
- Basic data preparation script
- Rule-based learning support

---

*For questions or issues, refer to the project's GitHub repository: https://github.com/Thundastormgod/paddleocr_vin_pipeline*
