# VIN OCR Pipeline - Model Registry & Documentation

This document provides a comprehensive overview of all models used throughout the VIN OCR pipeline, including their purposes, configurations, and where they are used.

---

## ⚠️ Important: Recognition vs Fine-tuned Evaluation

### Recognition Evaluation (Default)
Tests raw OCR capability using **default/pretrained weights**. This does NOT evaluate fine-tuned model performance.

### Fine-tuned Model Evaluation
For proper evaluation of custom-trained models, **export to ONNX format** for consistent, production-ready results:

```bash
# Export fine-tuned model to ONNX
python -m src.vin_ocr.training.export_onnx --model-path output/vin_rec_finetune/best_accuracy

# Run production evaluation with ONNX models
python -m src.vin_ocr.evaluation.multi_model_evaluation --mode production
```

---

## 📋 Model Registry Summary

| Model Key | Model Name | Type | Evaluation Type | Purpose |
|-----------|------------|------|-----------------|---------|
| `paddleocr_v4` | PaddleOCR PP-OCRv4 | `paddleocr` | Recognition Only | Default OCR engine (pretrained) |
| `paddleocr_v3` | PaddleOCR PP-OCRv3 | `paddleocr` | Recognition Only | Legacy OCR engine (pretrained) |
| `vin_pipeline` | VIN Pipeline | `vin_pipeline` | Recognition Only | PaddleOCR + VIN-specific processing |
| `finetuned` | Fine-tuned VIN Model | `finetuned` | Fine-tuned | Custom PaddleOCR model (Paddle format) |
| `finetuned_onnx` | Fine-tuned ONNX | `finetuned_onnx` | Production | Custom PaddleOCR model (ONNX format) |
| `deepseek` | DeepSeek-OCR | `deepseek` | Recognition Only | Vision-Language Model (pretrained) |
| `deepseek_finetuned` | Fine-tuned DeepSeek | `deepseek_finetuned` | Fine-tuned | DeepSeek VLM (HPC/CUDA trained) |
| `deepseek_finetuned_onnx` | DeepSeek ONNX | `deepseek_finetuned_onnx` | Production | DeepSeek VLM (ONNX exported) |
| `onnx_*` | ONNX Models | `onnx` | Production | Generic exported models |

---

## 🔄 Evaluation Modes

| Mode | Description | Models Tested |
|------|-------------|---------------|
| `recognition` | Tests raw OCR with default weights | paddleocr_v4, paddleocr_v3, vin_pipeline, deepseek |
| `finetuned` | Evaluates custom trained models | finetuned, deepseek_finetuned, paddleocr_* for comparison |
| `production` | Tests ONNX models for deployment | onnx_*, finetuned_onnx, deepseek_finetuned_onnx |

```bash
# Recognition evaluation (default)
python -m src.vin_ocr.evaluation.multi_model_evaluation --mode recognition

# Fine-tuned evaluation (includes DeepSeek if trained on HPC)
python -m src.vin_ocr.evaluation.multi_model_evaluation --mode finetuned

# Production/ONNX evaluation (all exported models)
python -m src.vin_ocr.evaluation.multi_model_evaluation --mode production
```

---

## 🌐 Using Trained Models in Web UI

The Web UI automatically discovers and lists fine-tuned models for inference:

### Automatic Model Discovery

Fine-tuned models are automatically detected from these directories:
- `output/vin_rec_finetune/` - PaddleOCR fine-tuned models
- `output/paddleocr_scratch/` - Models trained from scratch  
- `output/deepseek_finetune/` - DeepSeek fine-tuned models
- `models/finetuned/` - Manually placed models
- `output/onnx/`, `models/onnx/` - ONNX exported models

### Model Selection for Inference

In the Web UI Recognition page:
1. The **Select OCR Model** dropdown shows all available models
2. Fine-tuned models appear with 🎯 prefix (e.g., "🎯 Fine-tuned: best_accuracy")
3. ONNX models appear with 📦 prefix (e.g., "📦 ONNX: vin_recognition")

### Expected Output Structure

After training, your output directory should look like:
```
output/vin_rec_finetune/
├── latest.pdparams          # Latest checkpoint
├── latest.pdopt             # Optimizer state
├── best_accuracy.pdparams   # Best accuracy checkpoint
├── best_accuracy.pdopt
├── epoch_10.pdparams        # Epoch checkpoint
└── config.yml               # Training config
```

These will appear in the dropdown as:
- 🎯 Fine-tuned: latest
- 🎯 Fine-tuned: best_accuracy
- 🎯 Fine-tuned: epoch_10

---

## 🤖 Model Details

### 1. PaddleOCR PP-OCRv4 (`paddleocr_v4`)

**Type:** `paddleocr`  
**Evaluation:** Recognition Only (pretrained weights)  
**Description:** PaddleOCR base model using the PP-OCR v4 architecture - the latest and most accurate version.

**Configuration:**
```python
PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    text_det_thresh=0.3,
    text_det_box_thresh=0.5,
)
```

**Used In:**
- Multi-model evaluation (`multi_model_evaluation.py`)
- VIN Pipeline as base engine
- Web UI inference

---

### 2. PaddleOCR PP-OCRv3 (`paddleocr_v3`)

**Type:** `paddleocr`  
**Evaluation:** Recognition Only (pretrained weights)  
**Description:** PaddleOCR with PP-OCR v3 architecture - included for backward compatibility and comparison.

**Configuration:**
```python
PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    ocr_version='PP-OCRv3',
)
```

**Used In:**
- Multi-model evaluation (comparison baseline)

---

### 3. VIN Pipeline (`vin_pipeline`)

**Type:** `vin_pipeline`  
**Evaluation:** Recognition Only (uses pretrained PaddleOCR)  
**Description:** Complete VIN recognition pipeline combining PaddleOCR with VIN-specific preprocessing and postprocessing.

**Components:**
1. **Preprocessing:**
   - Grayscale conversion
   - CLAHE contrast normalization
   - Bilateral filtering
   - Optimized for engraved metal plates

2. **OCR Engine:** PaddleOCR PP-OCRv4

3. **Postprocessing:**
   - Artifact removal (`*`, `#`, `X`, `I`, `Y`, `T`, `F`, `A` prefixes/suffixes)
   - Invalid character fixes (`I→1`, `O→0`, `Q→0`)
   - Position-based corrections
   - VIN validation (17 characters)

**Used In:**
- Main inference (`vin_pipeline.py`)
- Web UI VIN recognition
- Multi-model evaluation

---

### 4. Fine-tuned VIN Model (`finetuned`)

**Type:** `finetuned`  
**Evaluation:** Fine-tuned (custom weights)  
**Description:** PaddleOCR model fine-tuned specifically on VIN images for improved accuracy.

**Training Configuration:**
- Base: CRNN or SVTR architecture
- Dataset: VIN-specific images
- Output: `output/vin_rec_finetune/best_accuracy.pdparams`

**Training Options:**
1. **Fine-tuning** (`finetune_paddleocr.py`): Start from pretrained weights
2. **From Scratch** (`train_from_scratch.py`): Train new model entirely

**⚠️ Important:** For consistent evaluation results, export to ONNX format:
```bash
python -m src.vin_ocr.training.export_onnx --model-path output/vin_rec_finetune/best_accuracy
```

**Used In:**
- Multi-model evaluation (when available)
- Production inference (if better than baseline)

---

### 5. DeepSeek-OCR (`deepseek`)

**Type:** `deepseek`  
**Evaluation:** Recognition Only  
**Description:** Vision-Language Model (VLM) that can perform OCR using natural language understanding.

**Requirements:**
- `transformers>=4.46.0`
- `torch`
- GPU recommended

**Used In:**
- Multi-model evaluation (optional)
- Alternative OCR provider

---

### 6. Fine-tuned DeepSeek (`deepseek_finetuned`)

**Type:** `deepseek_finetuned`  
**Evaluation:** Fine-tuned (HPC/CUDA required)  
**Description:** DeepSeek-OCR model fine-tuned on VIN-specific data for superior recognition accuracy.

**⚠️ Requirements:**
- **HPC with NVIDIA RTX 3090 (24GB VRAM)**
- `transformers>=4.46.0`
- `torch` with CUDA support
- `accelerate`, `peft` (for efficient fine-tuning)

**Training Workflow (on HPC with RTX 3090):**
```bash
# 1. SSH to HPC cluster
ssh user@hpc-cluster

# 2. Activate CUDA environment
module load cuda/11.8
source /path/to/conda/bin/activate deepseek_env

# 3. Fine-tune DeepSeek on VIN data (optimized for 24GB VRAM)
python -m src.vin_ocr.training.finetune_deepseek \
    --data-dir data/vin_images \
    --output-dir output/deepseek_finetune \
    --epochs 10 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --gradient-checkpointing \
    --fp16

# 4. Export to ONNX for portable inference
python -m src.vin_ocr.training.export_deepseek_onnx \
    --model-path output/deepseek_finetune/best_model \
    --output-dir output/deepseek_finetune/onnx
```

**Memory Optimization for RTX 3090 (24GB):**
- Use `--gradient-checkpointing` to reduce memory usage
- Use `--fp16` for mixed precision training
- Batch size 4-8 recommended for DeepSeek-VL 7B
- Consider `--load-in-8bit` for larger models

**Model Output Locations:**
- PyTorch checkpoint: `output/deepseek_finetune/best_model/pytorch_model.bin`
- Safetensors format: `output/deepseek_finetune/best_model/model.safetensors`
- Config: `output/deepseek_finetune/best_model/config.json`

**Used In:**
- Multi-model evaluation (when GPU available)
- High-accuracy VIN recognition

---

### 7. Fine-tuned DeepSeek ONNX (`deepseek_finetuned_onnx`)

**Type:** `deepseek_finetuned_onnx`  
**Evaluation:** Production  
**Description:** ONNX-exported fine-tuned DeepSeek model for portable, production-ready inference without PyTorch/transformers dependencies.

**Benefits:**
- No PyTorch/transformers dependency required
- Cross-platform deployment
- Faster inference with ONNX Runtime
- Can run on CPU (though GPU still faster)

**Export from HPC:**
```bash
# On HPC cluster with the fine-tuned model
python -m src.vin_ocr.training.export_deepseek_onnx \
    --model-path output/deepseek_finetune/best_model \
    --output-dir models/deepseek_onnx

# Copy ONNX files to local machine
scp -r user@hpc:output/deepseek_finetune/onnx/ ./models/deepseek_onnx/
```

**Local Inference Requirements:**
```bash
pip install onnx onnxruntime
# Or with GPU: pip install onnxruntime-gpu
```

**Used In:**
- Production deployment
- Multi-model evaluation (`--mode production`)
- Environments without GPU

---

### 8. ONNX Exported Models (`onnx_*`, `finetuned_onnx`)

**Type:** `onnx` / `finetuned_onnx`  
**Evaluation:** Production  
**Description:** Models exported to ONNX format for production deployment and consistent evaluation.

**Benefits:**
- Cross-platform deployment (no PaddlePaddle dependency)
- Faster inference with ONNX Runtime
- Consistent results across environments
- Professional-grade production deployment

**Export Command:**
```bash
# Export fine-tuned model
python -m src.vin_ocr.training.export_onnx \
    --model-path output/vin_rec_finetune/best_accuracy \
    --output-dir output/onnx

# Export with custom input size
python -m src.vin_ocr.training.export_onnx \
    --model-path output/model \
    --input-height 48 \
    --input-width 320
```

**Requirements:**
```bash
pip install onnx onnxruntime paddle2onnx
```

**Used In:**
- Production evaluation (`--mode production`)
- Deployment to non-Python environments
- Cross-platform inference

---

## 📊 Model Usage by Component
### Training Components

| Component | Models Used | Purpose |
|-----------|-------------|---------|
| `finetune_paddleocr.py` | PaddleOCR base | Fine-tune on VIN data |
| `train_from_scratch.py` | CRNN, SVTR_Tiny | Train new recognition model |
| `optuna_tuning.py` | PaddleOCR, DeepSeek | Hyperparameter optimization |

### Evaluation Components

| Component | Models Used | Purpose |
|-----------|-------------|---------|
| `multi_model_evaluation.py` | All registered models | Compare model performance |
| `evaluate.py` | VIN Pipeline | Single model evaluation |

### Inference Components

| Component | Models Used | Purpose |
|-----------|-------------|---------|
| `vin_pipeline.py` | VIN Pipeline | Production inference |
| `app.py` (Web UI) | All available | Interactive testing |

---

## 🔧 Model Configuration

### Hyperparameter Search Spaces

**PaddleOCR Models:**
```python
{
    'architecture': ['CRNN', 'SVTR_Tiny'],
    'learning_rate': (1e-5, 1e-2),
    'batch_size': [4, 8, 16, 32],
    'epochs': (5, 50),
    'optimizer': ['Adam', 'AdamW', 'SGD'],
    'weight_decay': (1e-6, 1e-2),
    'warmup_epochs': (0, 5),
    'label_smoothing': (0.0, 0.2),
    'image_height': [32, 48],
    'image_width': [128, 192, 256, 320],
    'dropout': (0.0, 0.5),
}
```

---

## 📁 Model Artifacts

### Output Locations

| Model Type | Output Directory | Key Files |
|------------|------------------|-----------|
| Fine-tuned | `output/vin_rec_finetune/` | `best_accuracy.pdparams`, `latest.pdparams` |
| Scratch | `output/paddleocr_scratch_*/` | `best_model/`, `training_log.json` |
| Tuning | `output/*_tuning_*/` | `optimization_results.json`, `trial_history.csv` |

### Evaluation Results

| File | Contents |
|------|----------|
| `results/multi_model_evaluation.json` | Full evaluation metrics per model |
| `results/model_comparison.csv` | Summary comparison table |
| `results/sample_results.csv` | Per-image results with model attribution |

---

## 🔄 Model Selection Logic

The pipeline automatically selects models based on availability:

```python
# Priority order for inference
1. Fine-tuned model (if exists and validated)
2. VIN Pipeline (default)
3. PaddleOCR PP-OCRv4 (fallback)
```

For evaluation, all available models are tested and compared.

---

## 📝 Adding New Models

To add a new model to the evaluation:

1. **Register in `multi_model_evaluation.py`:**
```python
# In load_models() method
self.models['new_model_key'] = {
    'name': 'New Model Name',
    'engine': model_instance,
    'type': 'model_type',  # Add to MODEL_TYPE_DESCRIPTIONS
}
```

2. **Add run method:**
```python
def run_new_model(self, engine, image_path: str) -> Tuple[str, float]:
    # Implementation
    return prediction, confidence
```

3. **Update MODEL_TYPE_DESCRIPTIONS:**
```python
MODEL_TYPE_DESCRIPTIONS = {
    ...
    'model_type': 'Description of the new model',
}
```

---

## 📈 Model Performance Tracking

All evaluation runs include:
- **Model attribution**: Each result linked to specific model
- **Timestamp**: When evaluation was run
- **Configuration**: Model settings used
- **Metrics**: Accuracy, F1, confidence, processing time

Results are saved with full model provenance in:
- `multi_model_evaluation.json` - Complete metrics
- `sample_results.csv` - Per-image with model column
