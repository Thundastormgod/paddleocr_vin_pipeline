# PaddleOCR VIN Recognition Pipeline

A complete OCR pipeline for Vehicle Identification Number (VIN) recognition 
from engraved metal plates, using PaddleOCR with specialized preprocessing 
and postprocessing.

---

## Key Metrics (Current Performance)

| Metric                   | Value | Baseline | Improvement |
|--------------------------|-------|----------|-------------|
| **Character-Level F1**   | 55%   | 43%      | +29%        |
| **Exact Match Rate**     | 25%   | 5%       | +400%       |
| **Precision**            | 57%   | 45%      | +27%        |
| **Recall**               | 54%   | 42%      | +29%        |

> **Note:** Industry target is 95%+ exact match and 98%+ F1.  
> This pipeline establishes a baseline for further development.

---

## Table of Contents

1. [Experiment Summary](#experiment-summary)
2. [CLI Testing Results](#cli-testing-results)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Web UI](#web-ui)
6. [Multi-Model Evaluation](#multi-model-evaluation)
7. [Training & Fine-Tuning](#training--fine-tuning)
8. [Pipeline Architecture](#pipeline-architecture)
9. [Configuration](#configuration)
10. [Character Confusion Handling](#character-confusion-handling)
11. [VIN Format Reference](#vin-format-reference)
12. [Documentation](#documentation)
13. [For the Team](#for-the-team)
14. [License](#license)

---

## Experiment Summary

### Dataset

| Property       | Value                                    |
|----------------|------------------------------------------|
| Total Images   | 382 VIN plate images                     |
| Source         | DagsHub bucket (JRL-VIN project)         |
| Image Type     | Engraved metal VIN plates from vehicles  |
| Ground Truth   | Manual annotations with verified VINs    |

### Industry Metrics Achieved

| Metric              | Baseline       | With Pipeline  | Improvement | Industry Target |
|---------------------|----------------|----------------|-------------|-----------------|
| Exact Match Rate    | 5% (19/382)    | 25% (96/382)   | +400%       | 95%+            |
| Character-Level F1  | 43%            | 55%            | +29%        | 98%+            |
| Precision           | 45%            | 57%            | +27%        | 98%+            |
| Recall              | 42%            | 54%            | +29%        | 98%+            |
| Detection Rate      | 99.7%          | 99.7%          | --          | 99%+            |
| Avg Processing Time | ~1.6s/image    | ~2.3s/image    | +44%        | <5s             |

### Additional Metrics to Explore

| Metric                      | Formula                                       | Status          |
|-----------------------------|-----------------------------------------------|-----------------|
| CER (Character Error Rate)  | (S + D + I) / N                               | To calculate    |
| NED (Normalized Edit Dist)  | edit_distance / max(len_pred, len_gt)         | To calculate    |
| Word Error Rate (WER)       | Errors at VIN level                           | Have (1-exact)  |
| Per-Position Accuracy       | Accuracy at each of 17 positions              | To calculate    |
| Levenshtein Distance (Avg)  | Mean edits needed to correct                  | To calculate    |

### Why These Results Matter

**1. Baseline Performance Gap**

Raw PaddleOCR achieves only 5% exact match on engraved plates due to:
- Metal surface reflections and lighting variations
- Character confusions (O/0, I/1, S/5) common on stamped text
- Artifact characters from plate borders and stamps

**2. Pipeline Improvements**

Our preprocessing + postprocessing pipeline achieves 5x improvement:
- CLAHE contrast enhancement handles lighting variations
- Artifact removal strips border characters (*, #, X prefixes)
- Invalid character correction (I→1, O→0, Q→0 per VIN standard)
- Position-based correction (digits in sequential section)

**3. Gap to Production**

Current 25% exact match is NOT production-ready (industry requires 95%+).
This baseline establishes:
- A validated preprocessing approach for engraved plates
- Identified failure modes for targeted improvements
- A foundation for the team to build upon

### Recommended Next Steps (Not Yet Implemented)

| Priority | Action                                     | Expected Impact     | Status      |
|----------|--------------------------------------------|---------------------|-------------|
| High     | Fine-tune detection model on VIN plates    | +20-30% exact match | Not started |
| High     | Train custom recognition model on charset  | +15-25% exact match | Not started |
| Medium   | Implement confidence-weighted voting       | +5-10% exact match  | Not started |
| Medium   | Add manufacturer-specific WMI validation   | +3-5% exact match   | Not started |
| Low      | Multi-angle image capture                  | +5-10% exact match  | Not started |

---

## CLI Testing Results (January 2026)

### Test Environment

| Component            | Value                            |
|----------------------|----------------------------------|
| PaddleOCR Version    | 3.x (PP-OCRv5)                   |
| Preprocessing Mode   | engraved (CLAHE + bilateral)     |
| Python               | 3.12                             |
| Platform             | macOS (Apple Silicon)            |

### Images Tested

1. `1-VIN_-_SAL119E90SA606112_.jpg`
2. `10-VIN_-_SAL1A2A40SA606645_.jpg`
3. `1000-VIN_-_SAL1P9EU2SA606633_.jpg`
4. `1001-VIN_-_SAL1P9EU2SA606664_.jpg`

### Preprocessing Pipeline

| Step | Operation                                              |
|------|--------------------------------------------------------|
| 1    | Load image (BGR format)                                |
| 2    | Convert to grayscale                                   |
| 3    | Apply CLAHE (clip_limit=2.0, tile_size=8x8)            |
| 4    | Bilateral filter (d=5, sigmaColor=50, sigmaSpace=50)   |
| 5    | Convert back to BGR (3-channel) for PaddleOCR          |

### Model Configuration

| Parameter            | Value                  |
|----------------------|------------------------|
| Detection Model      | PP-OCRv5_server_det    |
| Recognition Model    | en_PP-OCRv5_mobile_rec |
| Language             | English                |
| text_det_box_thresh  | 0.3                    |

### Test Results

| Image                             | Expected VIN        | Predicted VIN          | Conf | Match |
|-----------------------------------|---------------------|------------------------|------|-------|
| 1-VIN_-_SAL119E90SA606112_.jpg    | SAL119E90SA606112   | 2ESAL119E90SA606112    | 55%  | No    |
| 10-VIN_-_SAL1A2A40SA606645_.jpg   | SAL1A2A40SA606645   | SAL1A2K40SR606E45M     | 69%  | No    |
| 1000-VIN_-_SAL1P9EU2SA606633_.jpg | SAL1P9EU2SA606633   | 1401SA10EH/SA5066331   | 33%  | No    |
| 1001-VIN_-_SAL1P9EU2SA606664_.jpg | SAL1P9EU2SA606664   | SAL1P9EU2SA606664      | 96%  | Yes   |

**Summary:** 1/4 exact matches (25%) - consistent with full dataset results

### Errors Encountered During Development

**1. Deprecated Parameter Error**
```
Error: DeprecationWarning: det_db_box_thresh has been deprecated
Fix:   Changed to text_det_box_thresh in PaddleOCR 3.x
```

**2. Invalid Parameter Error**
```
Error: ValueError: Unknown argument: rec_thresh
Fix:   Removed rec_thresh parameter (no longer supported in PaddleOCR 3.x)
```

**3. Image Dimension Error**
```
Error: ValueError: not enough values to unpack (expected 3, got 2)
Cause: PaddleOCR expects 3-channel BGR images, preprocessing returned grayscale
Fix:   Added cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) after preprocessing
```

**4. OCR Artifact Characters**
```
Issue:   Raw OCR output contains *, #, X, / characters from plate borders
Example: "*SAL1P9EU2SA606664" -> "SAL1P9EU2SA606664"
Note:    "/" character not yet filtered (seen in test image 1000)
```

### Observed Failure Modes

- **Prefix artifacts:** "2E*", "I" prepended to VIN
- **Character confusion:** A↔K, 6↔E, 9↔0
- **Slash insertion:** "/" appearing mid-VIN from scratches/reflections
- **Low confidence:** (<50%) correlates with incorrect predictions

---

## Installation

### Basic Installation (Recommended)

For most users (PaddleOCR + Web UI):

```bash
pip install -r requirements.txt
```

This installs all core dependencies including:
- PaddleOCR for VIN recognition
- Streamlit web interface
- Image processing libraries (OpenCV, Pillow)
- Data analysis tools (Pandas, Plotly)

### Advanced Options

For detailed installation options including:
- GPU acceleration setup
- DeepSeek-OCR support
- Custom installations
- Troubleshooting

See [INSTALLATION.md](INSTALLATION.md) for the complete installation guide.

---

## Quick Start

```python
from vin_pipeline import VINOCRPipeline

pipeline = VINOCRPipeline()
result = pipeline.recognize('path/to/vin_image.jpg')

print(result['vin'])           # "SAL1P9EU2SA606664"
print(result['confidence'])    # 0.91
print(result['raw_ocr'])       # "XSAL1P9EU2SA606664*"
```

---

## Web UI

A Streamlit-based web interface for easy interaction with all models.

### Launch Web UI

```bash
# Install web UI dependencies
pip install -r web_ui/requirements.txt

# Run the web interface
streamlit run src/vin_ocr/web/app.py

# Or with custom port
streamlit run src/vin_ocr/web/app.py --server.port 8080
```

### Features

| Page | Description |
|------|-------------|
| 🔍 **Recognition** | Upload single images for VIN extraction |
| 📊 **Batch Evaluation** | Process folders with metrics comparison |
| 🎯 **Training** | Configure and monitor model training |
| 📈 **Dashboard** | View results and export data |

---

## Multi-Model Evaluation

Compare different OCR models on your dataset:

```bash
# Evaluate all available models on test images
python -m src.vin_ocr.evaluation.multi_model_evaluation --max-images 100

# Specify custom image folder
python -m src.vin_ocr.evaluation.multi_model_evaluation --image-folder ./my_images --max-images 50

# Output to specific directory
python -m src.vin_ocr.evaluation.multi_model_evaluation --output-dir ./results/experiment1
```

### Available Models

| Model | Type | Description |
|-------|------|-------------|
| **VIN Pipeline** | Local | PP-OCRv5 with post-processing |
| **PaddleOCR v4** | Local | Latest PaddleOCR release |
| **PaddleOCR v3** | Local | Previous generation |
| **DeepSeek-OCR** | Local | Vision-language model (requires GPU) |

### Output Metrics

- **F1 Micro/Macro** - Character-level F1 scores
- **Exact Match Accuracy** - Full VIN match rate
- **Character Accuracy** - Per-character accuracy
- **Per-sample results** - CSV with detailed breakdown

---

## Training & Fine-Tuning

### PaddleOCR Fine-Tuning

Fine-tune PP-OCRv4/v5 recognition model on VIN data:

```bash
# Run with default config
python -m src.vin_ocr.training.finetune_paddleocr --config configs/vin_finetune_config.yml

# Resume from checkpoint
python -m src.vin_ocr.training.finetune_paddleocr --config configs/vin_finetune_config.yml \
    --resume output/vin_rec_finetune/latest

# Multi-GPU training
python -m paddle.distributed.launch --gpus '0,1' -m src.vin_ocr.training.finetune_paddleocr \
    --config configs/vin_finetune_config.yml
```

### DeepSeek-OCR Fine-Tuning (HPC with RTX 3090)

Fine-tune DeepSeek vision-language model on HPC with NVIDIA RTX 3090 (24GB VRAM):

```bash
# SSH to HPC cluster
ssh user@hpc-cluster

# LoRA fine-tuning (optimized for RTX 3090 24GB)
python -m src.vin_ocr.training.finetune_deepseek \
    --config configs/deepseek_finetune_config.yml \
    --lora \
    --gradient-checkpointing \
    --bf16

# If running out of memory, use 8-bit quantization
python -m src.vin_ocr.training.finetune_deepseek \
    --config configs/deepseek_finetune_config.yml \
    --lora \
    --load-in-8bit

# Export to ONNX for portable inference
python -m src.vin_ocr.training.export_deepseek_onnx \
    --model-path output/deepseek_vin_finetune/best_model \
    --output-dir models/deepseek_onnx
```

**RTX 3090 (24GB) Memory Guidelines:**
| Method | VRAM Usage | Recommended Settings |
|--------|------------|---------------------|
| LoRA + bf16 | ~20-24GB | `batch_size=4, gradient_accumulation=8` |
| LoRA + 8-bit | ~14-16GB | Use if bf16 causes OOM |
| Full fine-tuning | ~48GB+ | Not recommended for RTX 3090 |

### Use Fine-Tuned Models at Inference

Load a full fine-tuned DeepSeek model or LoRA adapters via the provider factory:

```python
from ocr_providers import OCRProviderFactory

# Full fine-tuned model (local path)
provider = OCRProviderFactory.create(
    "deepseek",
    finetuned_model_path="/path/to/fine_tuned_model"
)

# PEFT adapter (LoRA/QLoRA)
provider = OCRProviderFactory.create(
    "deepseek",
    adapter_path="/path/to/adapter",
    merge_adapter=False
)
```

If using vLLM, note that PEFT adapters are ignored (vLLM backend does not load adapters).

### Training Data Format

Place images and labels in the data directory:

```
dagshub_data/
├── train/
│   ├── images/
│   │   ├── SAL1A2A40SA605902_train_8.jpg
│   │   └── ...
│   └── train_labels.txt
└── test/
    ├── images/
    │   └── ...
    └── test_labels.txt
```

Label file format (`train_labels.txt`):
```
images/SAL1A2A40SA605902_train_8.jpg	SAL1A2A40SA605902
images/WBY1Z2C55KV304518_train_12.jpg	WBY1Z2C55KV304518
```

---

## Training, Testing & Evaluation Pipeline

Run complete experiments with industry-standard metrics:

```bash
# Quick test with current data
python test_pipeline.py

# Full experiment with train/val/test splits
python run_experiment.py --data-dir data --output-dir experiments

# Custom split ratios
python run_experiment.py --data-dir data \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### Image Naming Convention

Images must be named as: `NUMBER-VIN -VINCODE.jpg`

Examples:
- `1-VIN -SAL1A2A40SA606662.jpg`
- `42-VIN -1HGBH41JXMN109186.jpg`

### Metrics Calculated

| Metric | Description |
|--------|-------------|
| **Exact Match** | % of VINs predicted 100% correctly |
| **F1 Score** | Harmonic mean of precision & recall |
| **CER** | Character Error Rate |
| **NED** | Normalized Edit Distance |
| **Per-position** | Accuracy at each of 17 VIN positions |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed pipeline documentation.

---

## Pipeline Architecture

```
+---------------------------------------------------------------------+
|                     VIN OCR PIPELINE                                |
+---------------------------------------------------------------------+
|  +------------+    +------------+    +----------------------+       |
|  | PREPROCESS |--->| PADDLEOCR  |--->|   POSTPROCESSOR      |       |
|  +------------+    +------------+    +----------------------+       |
|  | - Grayscale|    | - PP-OCRv5 |    | - Artifact Removal   |       |
|  | - CLAHE    |    | - Detection|    | - Invalid Char Fix   |       |
|  | - Bilateral|    | - Recogn.  |    | - Checksum Validate  |       |
|  +------------+    +------------+    +----------------------+       |
+---------------------------------------------------------------------+
```

---

## Files

```
paddleocr_vin_pipeline/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── vin_pipeline.py           # Complete pipeline (single-file)
├── example_usage.py          # Usage examples
├── ARCHITECTURE.md           # Detailed architecture docs
├── LICENSE                   # Apache 2.0 License
├── NOTICE                    # Attribution notices
│
├── # Training & Evaluation Pipeline
├── prepare_dataset.py        # Dataset splitting & label generation
├── train_pipeline.py         # Training configuration & execution
├── run_experiment.py         # End-to-end experiment runner
├── test_pipeline.py          # Quick pipeline test
├── src/vin_ocr/evaluation/evaluate.py      # Evaluation with metrics
├── src/vin_ocr/utils/validate_dataset.py   # Dataset validation
│
├── tests/
│   └── test_vin_pipeline.py  # Test suite (52+ tests)
├── data/                     # VIN images (add your images here)
├── experiments/              # Experiment outputs
└── results/
    ├── experiment_summary.json
    ├── detailed_metrics.json
    └── sample_results.csv
```

---

## Configuration

```python
pipeline = VINOCRPipeline(
    preprocess_mode='engraved',  # 'none', 'fast', 'balanced', 'engraved'
    enable_postprocess=True,     # Enable VIN correction
    verbose=False                # Print processing steps
)
```

---

## Character Confusion Handling

| Confusion | Solution       | Reason                            |
|-----------|----------------|-----------------------------------|
| I → 1     | Auto-fix       | I is invalid in VIN               |
| O → 0     | Auto-fix       | O is invalid in VIN               |
| Q → 0     | Auto-fix       | Q is invalid in VIN               |
| S ↔ 5     | Position-based | Prefer digits in sequential section |
| L ↔ 1     | Position-based | Check surrounding characters      |
| * # X     | Remove         | Artifact characters               |

---

## VIN Format Reference

```
Position:  1  2  3  | 4  5  6  7  8 | 9 | 10 | 11 | 12 13 14 15 16 17
              |             |         |    |    |           |
           WMI          VDS        Check Year Plant   Sequential
      (Manufacturer) (Descriptor)  Digit            Number
```

**Valid Characters:** 0-9, A-H, J-N, P, R-Z (NO: I, O, Q)

---

## Documentation

### Developer Documentation
All technical documentation is located in the [`dev/`](dev/) directory:

- **[Architecture](dev/ARCHITECTURE.md)** - System design and component overview
- **[Environment Setup](dev/ENVIRONMENT.md)** - Complete installation guide with multiple methods
- **[Code Citations](dev/CODE_CITATIONS.md)** - Attribution for third-party code
- **[Training Guides](dev/docs/)** - Deep dive into training pipeline, techniques, and algorithms

### Quick Links
- **[Training Guide](dev/docs/TRAINING_GUIDE.md)** - Step-by-step training instructions
- **[Training Deep Dive](dev/docs/TRAINING_DEEP_DIVE.md)** - Advanced training concepts
- **[Algorithm Complexity](dev/docs/ALGORITHM_COMPLEXITY.md)** - Performance analysis
- **[Finetuning Techniques](dev/docs/FINETUNING_TECHNIQUES.md)** - Model adaptation strategies

---

## For the Team

### Quick Setup

```bash
git clone https://github.com/Thundastormgod/paddleocr_vin_pipeline.git
cd paddleocr_vin_pipeline
pip install -r requirements.txt
pytest tests/test_vin_pipeline.py -v
```

### Data Access

The 382 test images are in DagsHub: `Thundastormgod/jrl-vin`  
Path: `data/paddleocr_sample/`

### Known Limitations

- **NOT PRODUCTION-READY:** 25% vs 95% industry target
- **ENGRAVED PLATES ONLY:** Not for printed labels
- **SINGLE VIN PER IMAGE**

---

## License

**Apache License 2.0** - See [LICENSE](LICENSE) and [dev/NOTICE](dev/NOTICE)

| Requires                    | Permits                              |
|-----------------------------|--------------------------------------|
| Attribution                 | Commercial use                       |
| State changes               | Modification                         |
| Include license             | Distribution                         |
|                             | Patent use                           |

---

## Citation

If you use this software in research, please cite:

```
PaddleOCR VIN Recognition Pipeline
JRL-VIN Project, 2025
https://github.com/Thundastormgod/paddleocr_vin_pipeline
```


---

## Evaluation & Validation Tools

### Dataset Validation

Validate ground truth quality before training/evaluation:

```bash
# Validate dataset
python -m src.vin_ocr.utils.validate_dataset --data-dir /path/to/paddleocr_sample

# Output report to JSON
python -m src.vin_ocr.utils.validate_dataset --data-dir /path/to/data --output report.json
```

**What it checks:**
- VIN extraction from filenames
- Filename vs label file consistency
- VIN format validity (length, characters, checksum)
- Duplicate detection

### Model Evaluation

Evaluate pipeline performance with train/val/test splits:

```bash
# Evaluate on all data
python -m src.vin_ocr.evaluation.evaluate --data-dir /path/to/images

# Create 70/15/15 splits and evaluate test set
python -m src.vin_ocr.evaluation.evaluate --data-dir /path/to/images --create-splits --split test

# Evaluate validation set with existing splits
python -m src.vin_ocr.evaluation.evaluate --data-dir /path/to/images --split-dir ./splits --split val

# Export results
python -m src.vin_ocr.evaluation.evaluate --data-dir /path/to/images --output results.json --csv predictions.csv

# Quick test with limited samples
python -m src.vin_ocr.evaluation.evaluate --data-dir /path/to/images --max-samples 50
```

### Metrics Calculated

| Metric | Description |
|--------|-------------|
| Exact Match Rate | Percentage of perfectly predicted VINs |
| Character-Level F1 | Harmonic mean of precision and recall |
| Character Precision | Correct chars / Total predicted chars |
| Character Recall | Correct chars / Total reference chars |
| CER | Character Error Rate (lower is better) |
| NED | Normalized Edit Distance |
| Per-Position Accuracy | Accuracy at each of 17 VIN positions |

### Dataset Splits

The evaluation tool supports professional ML workflow with train/val/test splits:

| Split | Default Ratio | Purpose |
|-------|---------------|---------|
| Train | 70% | Model training |
| Val | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation |

Split files are saved to `splits/` directory:
- `train_split.txt`
- `val_split.txt`  
- `test_split.txt`

### CI/CD Integration

The evaluate script outputs machine-readable metrics:

```
EXACT_MATCH_RATE=0.2500
CHARACTER_F1=0.5500
CER=0.4500
```
