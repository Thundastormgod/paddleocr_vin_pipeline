# PaddleOCR VIN Recognition Pipeline

A complete OCR pipeline for Vehicle Identification Number (VIN) recognition
from engraved metal plates: PaddleOCR engine + VIN-specific preprocessing,
postprocessing (artifact stripping, charset fixes, ISO-3779 checksum),
training, tracked evaluation, and a web UI.

---

## Measured Performance (2026-08-20)

Every number below was measured by a tracked MLflow run in experiment
`model_comparison` (one run per model, identical VIN-disjoint splits,
canonical character metrics, single-image basis). No other performance
numbers in this repository's history are citable; see `LOGBOOK.md` for the
full measurement record and the removal of the fabrication-era documents.

| Model | val-102 char / exact | test-40 char / exact |
|---|---|---|
| **PP-OCRv3-mobile + pipeline (production default)** | **82.70% / 25.5%** | **87.79% / 37.5%** |
| PP-OCRv3-mobile + pipeline, postprocessor OFF | 77.51% / 8.8% | 81.03% / 5.0% |
| LCNetV3-SVTR-CTC-ep44 +postproc | 68.28% / 0% | 68.38% / 0% |
| LCNetV3-SVTR-CTC-ep44 (registry v3, alias `best`) | 66.61% / 0% | 68.53% / 0% |
| LCNetV3-SVTR-CTC-ep19 best-val-loss (v2) | 64.13% / 0% | 66.91% / 0% |
| LCNetV3-SVTR-CTC-ep25 (v1) | 59.98% / 0% | 60.88% / 0% |
| LCNetV3-SVTR-CTC-ep36 warm-restart (refuted) | 38.06% / 0% | 42.94% / 0% |

Reading the table:

- **Production default** is the stock PaddleOCR `PP-OCRv3_mobile_det` +
  `en_PP-OCRv3_mobile_rec` engine wrapped in this repo's preprocessing and
  postprocessing. The postprocessor alone contributes +16.7pp (val) /
  +32.5pp (test) exact match.
- **LCNetV3-SVTR-CTC** is the custom from-scratch model family (3.23M
  params). It is a refuted route - kept in the MLflow registry
  (`vin-lcnetv3-svtr-ctc`) as the honest record - because 2,387 training
  crops cannot compete with industrial-scale pretraining.
- Industry target for this domain is ~95%+ exact match; nothing here is
  production-ready yet. The measured improvement ladder (GPU + pretrained
  warm start + input resolution) lives in `LOGBOOK.md`.
- Dataset: 2,529 crops from DagsHub `Thundastormgod/jlr-vin-ocr`,
  VIN-grouped splits 2,387/102/40, 100% checksum-valid labels, pinned by
  `finetune_data.dvc`.

Reproduce the table:

```bash
python scripts/compare_models.py            # ~6 min CPU, logs to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

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

### Development Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # or ".[all]" for web + training + onnx
pytest                        # full suite; no GPU or model weights required
```

Copy `.env.example` to `.env` and fill in credentials (DagsHub tokens, device
and threshold overrides). `.env` is gitignored and must never be committed.

### Optional Extras

| Extra      | Installs                                  |
|------------|-------------------------------------------|
| `web`      | Streamlit, Plotly, Pandas                 |
| `training` | VisualDL, Optuna                          |
| `onnx`     | onnx, onnxruntime, paddle2onnx            |
| `dev`      | pytest, black, isort, flake8, mypy        |
| `all`      | everything above                          |

> **OpenCV:** install exactly one of `opencv-python`, `opencv-contrib-python`
> or `opencv-python-headless`. They all provide `cv2`; installing more than one
> leaves whichever pip unpacked last, at an unpredictable version.

---

## Quick Start

```python
from src.vin_ocr.pipeline.vin_pipeline import VINOCRPipeline

pipeline = VINOCRPipeline()
result = pipeline.recognize('path/to/vin_image.jpg')

print(result['vin'])           # "SAL1P9EU2SA606664"
print(result['confidence'])    # 0.91
print(result['raw_ocr'])       # "XSAL1P9EU2SA606664*"
```

Validation helpers are importable without the heavy OCR backends:

```python
from src.vin_ocr.core import validate_vin, extract_vin_from_filename

validate_vin("SAL1A2A40SA606662").is_fully_valid   # True
extract_vin_from_filename("1-VIN -SAL1A2A40SA606662.jpg")
```

CLI equivalents:

```bash
vin-ocr recognize image.jpg          # single image
vin-ocr batch ./images -o out.json   # folder
vin-ocr serve                        # Streamlit UI on :8501
vin-train finetune --help            # training commands
vin-evaluate single --help           # evaluation commands
```

---

## Web UI

A Streamlit-based web interface for easy interaction with all models.

### Launch Web UI

```bash
# Install web UI dependencies
pip install -e ".[web]"        # or: pip install -r src/vin_ocr/web/requirements.txt

# Run the web interface
streamlit run src/vin_ocr/web/app.py     # or: make run  /  vin-ocr serve

# Or with custom port
streamlit run src/vin_ocr/web/app.py --server.port 8080
```

### Features

| Page | Description |
|------|-------------|
| 📁 **Data Management** | Prepare datasets and labels |
| 🎯 **Training** | Configure and monitor model training |
| 🔍 **Inference** | Upload images for VIN extraction |
| 📈 **Results Dashboard** | View results and export data |

---

## Multi-Model Evaluation

Compare different OCR models on your dataset:

```bash
# Evaluate all available models on test images
python -m src.vin_ocr.evaluation.multi_model_evaluation --max-images 100

# Specify custom image folder
python -m src.vin_ocr.evaluation.multi_model_evaluation --image-folder ./my_images --max-images 50

# Output to specific directory
python -m src.vin_ocr.evaluation.multi_model_evaluation --model paddleocr_v3 --output-dir ./eval_runs/experiment1
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

```

> **Multi-GPU is NOT supported.** `finetune_paddleocr.py` contains no
> `DataParallel`, `fleet` or `init_parallel_env` call, so
> `paddle.distributed.launch` would start N independent single-GPU processes
> that overwrite each other's checkpoints. Use a single device.

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
from src.vin_ocr.providers.ocr_providers import OCRProviderFactory

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
# Quick sanity check (unit tests, no GPU or weights required)
pytest

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

See [Pipeline Architecture](#pipeline-architecture) below and
[docs/SYSTEM_MAPS.md](docs/SYSTEM_MAPS.md) for the generated architecture maps.

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
├── src/vin_ocr/              # Installable package
│   ├── core/                 # VIN spec, validation, checksum, charset
│   │   ├── vin_utils.py      #   validation, correction, extraction
│   │   └── charset.py        #   char<->index map (training + inference)
│   ├── preprocessing/        # CLAHE / engraved-plate strategies
│   ├── providers/            # PaddleOCR + DeepSeek backends, ensemble
│   ├── pipeline/             # VINOCRPipeline, MultiProviderVINPipeline
│   ├── inference/            # ONNX and Paddle inference backends
│   ├── training/             # Fine-tune, scratch, ONNX export, Optuna
│   │   └── cli.py            #   `vin-train`
│   ├── evaluation/           # Metrics, single- and multi-model evaluation
│   │   └── cli.py            #   `vin-evaluate`
│   ├── utils/                # Dataset prep/validation, hardware detection
│   ├── web/app.py            # Streamlit UI
│   └── cli.py                # `vin-ocr`
│
├── configs/                  # Training configs + vin_dict.txt charset
├── scripts/                  # Standalone ONNX / data utilities
├── tests/                    # regression suite (no GPU or weights needed)
├── docker/                   # CPU + GPU images, compose, entrypoint
├── .github/workflows/ci.yml  # Tests, lint, secrets scan, build
│
├── run_experiment.py         # End-to-end experiment runner
├── train_pipeline.py         # Training configuration & execution
├── config.py                 # Env-var-driven settings singleton
├── .env.example              # Credential + tuning template
│
├── data/                     # VIN images (DVC-managed, gitignored)
├── output/                   # Checkpoints (gitignored)
└── eval_runs/                # Evaluation outputs (created on demand)
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

| Document | Contents |
|----------|----------|
| [LOGBOOK.md](LOGBOOK.md) | The measurement record: every number with its run ID and basis |
| [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md) | How to run tracked training, resume, register checkpoints |
| [ENTERPRISE_TRAINING_READINESS.md](ENTERPRISE_TRAINING_READINESS.md) | Trainer readiness audit and its fixes |
| [docs/SYSTEM_MAPS.md](docs/SYSTEM_MAPS.md) | Generated architecture/dependency maps (`make maps` regenerates) |
| [DAGSHUB_SETUP.md](DAGSHUB_SETUP.md) | DagsHub + DVC data setup |
| [docker/README.md](docker/README.md) | Container build and deployment |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution workflow |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

> The fabrication-era documents (simulated architecture benchmarks, status
> summaries, extrapolated results) were removed on 2026-08-20; they exist in
> git history only. Nothing in this repository cites them as evidence.

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

Recognition crops (2,529 images; VIN-grouped splits 2,387/102/40) come from
the DagsHub datasource `Thundastormgod/jlr-vin-ocr`; the staged copy is
pinned by `finetune_data.dvc`. See DAGSHUB_SETUP.md for credentials.

### Known Limitations

- **NOT PRODUCTION-READY:** best measured exact match is 37.5% (test-40,
  stock engine + pipeline) vs the ~95% industry target
- **ENGRAVED PLATES ONLY:** Not for printed labels
- **SINGLE VIN PER IMAGE**

---

## License

> **⚠️ UNRESOLVED — do not rely on this section yet.**
>
> The licensing of this project is currently self-contradictory:
>
> - `LICENSE` is a **corrupted file**: Apache-2.0 text with MIT License text
>   spliced into it mid-sentence (it literally begins
>   `Apache LicenseMIT License`, and the MIT grant is pasted into the middle of
>   the Apache "TERMS AND CONDITIONS" heading).
> - `pyproject.toml` declares `license = {text = "MIT"}`.
> - This README previously asserted Apache-2.0.
> - The referenced `NOTICE` file does not exist.
>
> Apache-2.0 and MIT impose different obligations (notably attribution/NOTICE
> and patent grants), so this must be settled by the copyright holder before
> distribution. Once decided, replace `LICENSE` with the clean upstream text,
> align `pyproject.toml`, and add a `NOTICE` file if Apache-2.0 is chosen.

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
