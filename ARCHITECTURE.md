# PaddleOCR VIN Pipeline - Complete Architecture Documentation

> **Last Updated**: January 2026  
> **Version**: 2.0  
> **Status**: Production-Ready

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Dependency Graph](#2-module-dependency-graph)
3. [Core Components](#3-core-components)
4. [Data Flow Architecture](#4-data-flow-architecture)
5. [Module Reference](#5-module-reference)
6. [Configuration System](#6-configuration-system)
7. [Processing Pipeline Details](#7-processing-pipeline-details)
8. [Training & Evaluation Pipeline](#8-training--evaluation-pipeline)
9. [Error Handling Strategy](#9-error-handling-strategy)
10. [Performance Characteristics](#10-performance-characteristics)
11. [Extension Points](#11-extension-points)

---

## 1. System Overview

This application is a **VIN (Vehicle Identification Number) OCR Pipeline** optimized for reading 17-character VINs from engraved metal plates. It combines PaddleOCR with domain-specific preprocessing and rule-based post-processing.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VIN OCR PIPELINE SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   config    │    │  vin_utils  │    │vin_pipeline │    │  evaluate   │      │
│  │   (cfg)     │◄───│   (utils)   │◄───│   (core)    │───▶│  (metrics)  │      │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘    └─────────────┘      │
│                            │                  │                                  │
│                            │                  │                                  │
│                            ▼                  ▼                                  │
│                     ┌─────────────┐    ┌─────────────┐                          │
│                     │  prepare_   │    │   train_    │                          │
│                     │  dataset    │    │  pipeline   │                          │
│                     └──────┬──────┘    └──────┬──────┘                          │
│                            │                  │                                  │
│                            └────────┬─────────┘                                  │
│                                     ▼                                            │
│                            ┌─────────────────┐                                   │
│                            │  run_experiment │                                   │
│                            │   (orchestrator)│                                   │
│                            └─────────────────┘                                   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                        EXTERNAL DEPENDENCIES                              │   │
│  ├──────────────────────────────────────────────────────────────────────────┤   │
│  │  PaddleOCR 3.0  │  PaddlePaddle 3.0  │  OpenCV  │  NumPy  │  Pydantic   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Single Source of Truth** | All VIN constants/utilities in `vin_utils.py` |
| **Centralized Configuration** | All settings in `config.py` with env var support |
| **Fail-Safe Error Handling** | Explicit exception handling, no bare `except:` |
| **Domain-Specific Optimization** | Rules tuned for engraved metal VIN plates |
| **Testability** | 104 unit tests, dependency injection ready |

---

## 2. Module Dependency Graph

```
                                 ┌─────────────────┐
                                 │   config.py     │
                                 │  (Configuration)│
                                 └────────┬────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
           ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
           │  vin_utils.py  │   │ vin_pipeline.py│   │train_pipeline.py│
           │ (VIN Utilities)│◄──│ (Core Pipeline)│   │  (Training)    │
           └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
                   │                    │                    │
       ┌───────────┼───────────┬───────┘                    │
       │           │           │                            │
       ▼           ▼           ▼                            │
┌────────────┐┌────────────┐┌────────────┐                  │
│evaluate.py ││prepare_    ││run_        │◄─────────────────┘
│(Evaluation)││dataset.py  ││experiment.py│
└────────────┘└────────────┘└─────────────┘

LEGEND:
  ──▶  imports from / depends on
  ◄──  provides services to
```

### Import Matrix

| Module | Imports From | Imported By |
|--------|--------------|-------------|
| `config.py` | (stdlib only) | `vin_utils`, `vin_pipeline`, `train_pipeline`, `evaluate`, `prepare_dataset`, `run_experiment` |
| `vin_utils.py` | `config` | `vin_pipeline`, `train_pipeline`, `evaluate`, `prepare_dataset`, `run_experiment` |
| `vin_pipeline.py` | `vin_utils`, PaddleOCR | `evaluate`, `train_pipeline`, `run_experiment` |
| `evaluate.py` | `vin_utils`, `config`, `vin_pipeline` | `run_experiment` |
| `prepare_dataset.py` | `vin_utils`, `config` | `run_experiment` |
| `train_pipeline.py` | `vin_utils`, `config`, `vin_pipeline` | `run_experiment` |
| `run_experiment.py` | `vin_utils`, `config`, `vin_pipeline` | (entry point) |

---

## 3. Core Components

### 3.1 File Structure

```
paddleocr_vin_pipeline/
│
├── 📄 vin_pipeline.py          # Core OCR pipeline (1105 lines)
│   ├── VINOCRPipeline          #   Main pipeline class
│   ├── VINImagePreprocessor    #   Image enhancement
│   └── VINPostProcessor        #   VIN correction
│
├── 📄 vin_utils.py             # Shared utilities (610 lines)
│   ├── VINConstants            #   ISO 3779 constants
│   ├── RuleBasedCorrector      #   Character correction
│   ├── extract_vin_from_filename()
│   ├── validate_vin()
│   ├── calculate_check_digit()
│   └── validate_vin_checksum()
│
├── 📄 config.py                # Configuration (266 lines)
│   ├── PreprocessingConfig     #   CLAHE, bilateral filter
│   ├── OCRConfig               #   PaddleOCR settings
│   ├── AugmentationConfig      #   Data augmentation
│   ├── TrainingConfig          #   Training hyperparameters
│   └── PipelineConfig          #   Master config
│
├── 📄 evaluate.py              # Evaluation (800 lines)
│   ├── EvaluationMetrics       #   Metrics dataclass
│   ├── VINEvaluator            #   Evaluation runner
│   └── calculate_metrics()     #   F1, CER, NED, etc.
│
├── 📄 prepare_dataset.py       # Dataset prep (302 lines)
│   ├── split_dataset()         #   Train/val/test splits
│   └── create_label_files()    #   PaddleOCR format
│
├── 📄 train_pipeline.py        # Rule Learning (395 lines)
│   ├── VINTrainingPipeline     #   Training orchestrator
│   ├── create_augmented_dataset()
│   └── train_rule_learning()   #   Learns correction rules (NOT neural network training)
│
├── 📄 run_experiment.py        # Experiment runner (570 lines)
│   └── run_full_experiment()   #   End-to-end pipeline
│
├── 📁 tests/                   # Unit tests (104 tests)
│   ├── test_vin_pipeline.py
│   ├── test_evaluate.py
│   └── test_validate_dataset.py
│
├── 📁 data/                    # Input images
│   └── 1-VIN -SAL1A2A40SA606662.jpg
│
├── 📁 results/                 # Evaluation output
│   ├── detailed_metrics.json
│   └── experiment_summary.json
│
├── 📄 requirements.txt         # Dependencies (pinned)
├── 📄 ARCHITECTURE.md          # This document
└── 📄 README.md                # Usage guide
```

### 3.2 Class Hierarchy

```
                        ┌─────────────────────┐
                        │    <<interface>>    │
                        │   OCR Pipeline      │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
           ┌────────▼────────┐          ┌────────▼────────┐
           │ VINOCRPipeline  │          │VINTrainingPipeline│
           │   (Runtime)     │          │   (Training)     │
           └────────┬────────┘          └──────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
┌───────▼───────┐ ┌─▼─────────┐ ┌▼──────────────┐
│VINImage       │ │PaddleOCR  │ │VINPost        │
│Preprocessor   │ │(external) │ │Processor      │
└───────────────┘ └───────────┘ └───────┬───────┘
                                        │
                              ┌─────────▼─────────┐
                              │ RuleBasedCorrector│
                              │   (vin_utils)     │
                              └───────────────────┘
```

---

## 4. Data Flow Architecture

### 4.1 Recognition Flow (Single Image)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         RECOGNITION DATA FLOW                                 │
└──────────────────────────────────────────────────────────────────────────────┘

  Input                    Processing                              Output
  ─────                    ──────────                              ──────

┌─────────┐           ┌─────────────────────────────────────┐    ┌──────────┐
│  Image  │           │         VINOCRPipeline              │    │  Result  │
│ (path)  │──────────▶│                                     │───▶│  (dict)  │
└─────────┘           │  ┌───────────┐                      │    └──────────┘
     │                │  │   LOAD    │                      │         │
     │                │  │cv2.imread │                      │         │
     │                │  └─────┬─────┘                      │         │
     │                │        │ BGR array                  │         │
     │                │        ▼                            │         │
     │                │  ┌───────────────────────────────┐  │         │
     │                │  │     VINImagePreprocessor      │  │         │
     │                │  │  ┌─────────────────────────┐  │  │         │
     │                │  │  │ 1. Convert to grayscale │  │  │         │
     │                │  │  │ 2. Apply CLAHE          │  │  │         │
     │                │  │  │ 3. Bilateral filter     │  │  │         │
     │                │  │  └─────────────────────────┘  │  │         │
     │                │  └──────────────┬────────────────┘  │         │
     │                │                 │ Enhanced gray     │         │
     │                │                 ▼                   │         │
     │                │  ┌───────────────────────────────┐  │         │
     │                │  │        PaddleOCR              │  │         │
     │                │  │  ┌──────────┐  ┌──────────┐   │  │         │
     │                │  │  │ Detection│─▶│Recognition│  │  │         │
     │                │  │  │  (DB++)  │  │  (SVTR)  │   │  │         │
     │                │  │  └──────────┘  └──────────┘   │  │         │
     │                │  └──────────────┬────────────────┘  │         │
     │                │                 │ (text, conf)      │         │
     │                │                 ▼                   │         │
     │                │  ┌───────────────────────────────┐  │         │
     │                │  │      VINPostProcessor         │  │         │
     │                │  │  ┌─────────────────────────┐  │  │         │
     │                │  │  │ 1. Remove artifacts     │  │  │         │
     │                │  │  │ 2. Fix I→1, O→0, Q→0    │  │  │         │
     │                │  │  │ 3. Position correction  │  │  │         │
     │                │  │  │ 4. Validate checksum    │  │  │         │
     │                │  │  └─────────────────────────┘  │  │         │
     │                │  └──────────────────────────────┘   │         │
     │                │                                     │         │
     │                └─────────────────────────────────────┘         │
     │                                                                │
     └────────────────────────────────────────────────────────────────┘

  Output Structure:
  {
    'vin': 'SAL1A2A40SA606662',      # Corrected VIN
    'raw_ocr': 'SALIA2A4OSA606662',   # Original OCR output
    'confidence': 0.87,               # OCR confidence
    'is_valid_length': True,          # 17 chars?
    'checksum_valid': True,           # Position 9 check
    'corrections': [...],             # Applied fixes
    'preprocessing_mode': 'engraved'
  }
```

### 4.2 Training Flow (Rule Learning, NOT Transfer Learning)

> **IMPORTANT**: The training pipeline does NOT fine-tune neural network weights.
> It learns character correction rules from OCR error patterns.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     RULE-BASED CORRECTION LEARNING                            │
│            (This is NOT neural network transfer learning)                     │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐
  │   Raw Images    │
  │  (data/*.jpg)   │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        prepare_dataset.py                                │
  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
  │  │Extract VIN from │───▶│  Split Dataset  │───▶│ Create Labels   │     │
  │  │   filenames     │    │ 70/15/15        │    │ (PaddleOCR fmt) │     │
  │  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
  └─────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        train_pipeline.py                                 │
  │                   (Rule Learning - NO weight updates)                    │
  │                                                                          │
  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
  │  │  Data Augment   │───▶│  Run Pretrained │───▶│ Build Confusion │     │
  │  │  (if < 50 imgs) │    │  PP-OCRv5       │    │    Matrix       │     │
  │  └─────────────────┘    └─────────────────┘    └────────┬────────┘     │
  │                                                          │              │
  │  Augmentation:               ┌───────────────────────────┘              │
  │  • Rotation (±3°)            │                                          │
  │  • Brightness (±20%)         ▼                                          │
  │  • Gaussian blur       ┌─────────────────┐                              │
  │  • Contrast adjust     │ Generate Rules  │                              │
  │                        │ from Confusions │                              │
  │                        └────────┬────────┘                              │
  │                                 │                                        │
  │  What this DOES:                │     What this does NOT do:            │
  │  ✓ Learns char→char mappings   │     ✗ Fine-tune model weights         │
  │  ✓ Builds confusion matrix      │     ✗ Backpropagation                 │
  │  ✓ Deterministic rules          │     ✗ Gradient descent                │
  │                                 │     ✗ Update neural network           │
  │                                 │                                        │
  └─────────────────────────────────┼───────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                          Learned Rules (JSON)                            │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  {                                                               │   │
  │  │    "type": "rule_based",                                        │   │
  │  │    "rules": {                                                    │   │
  │  │      "I": "1",  // OCR often mistakes I for 1                   │   │
  │  │      "O": "0",  // OCR often mistakes O for 0                   │   │
  │  │      "S": "5",  // In positions 12-17 (serial number)           │   │
  │  │      "B": "8",  // Visual similarity                            │   │
  │  │      ...                                                         │   │
  │  │    },                                                            │   │
  │  │    "note": "Rules learned from OCR error patterns"              │   │
  │  │  }                                                               │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────┘

  These rules are then applied in VINPostProcessor during inference.
```

### 4.3 Evaluation Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION DATA FLOW                                 │
└──────────────────────────────────────────────────────────────────────────────┘

   ┌───────────┐       ┌───────────┐
   │  Images   │       │  Ground   │
   │   (test)  │       │  Truth    │
   └─────┬─────┘       └─────┬─────┘
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │                          evaluate.py                                   │
   │                                                                        │
   │   For each image:                                                      │
   │   ┌────────────────────────────────────────────────────────────────┐  │
   │   │  1. Run VINOCRPipeline.recognize()                             │  │
   │   │  2. Compare prediction vs ground truth                         │  │
   │   │  3. Calculate per-sample metrics                               │  │
   │   └────────────────────────────────────────────────────────────────┘  │
   │                                                                        │
   │   Aggregate metrics:                                                   │
   │   ┌────────────────────────────────────────────────────────────────┐  │
   │   │  • Exact Match Rate     = correct / total                      │  │
   │   │  • F1 Score             = 2PR / (P+R)                          │  │
   │   │  • Character Error Rate = (S+D+I) / N                          │  │
   │   │  • Normalized Edit Dist = edit_dist / max_len                  │  │
   │   │  • Per-position accuracy[17]                                   │  │
   │   └────────────────────────────────────────────────────────────────┘  │
   │                                                                        │
   └─────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │                          Output Files                                  │
   │                                                                        │
   │   results/                                                             │
   │   ├── detailed_metrics.json    # Full metrics breakdown               │
   │   ├── experiment_summary.json  # Summary statistics                   │
   │   ├── sample_results.csv       # Per-image predictions                │
   │   └── confusion_matrix.json    # Character confusion analysis         │
   │                                                                        │
   └───────────────────────────────────────────────────────────────────────┘
```

---

## 5. Module Reference

### 5.1 `vin_utils.py` - Shared Utilities

**Purpose**: Single source of truth for VIN-related constants, validation, and correction.

```python
# Constants
VIN_LENGTH = 17
VIN_VALID_CHARS = frozenset("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
VIN_INVALID_CHARS = frozenset("IOQ")

# Key Functions
extract_vin_from_filename(filename: str) -> Optional[str]
validate_vin(vin: str) -> VINValidationResult
validate_vin_format(vin: str) -> bool
validate_vin_checksum(vin: str) -> bool
calculate_check_digit(vin: str) -> Optional[str]
correct_vin(raw_text: str, confidence: float) -> Dict

# Key Classes
class VINConstants:
    """ISO 3779 / NHTSA VIN specification constants"""
    
class RuleBasedCorrector:
    """Deterministic character correction for OCR errors"""
    
class VINValidationResult:
    """Structured validation result with all checks"""
```

### 5.2 `config.py` - Configuration Management

**Purpose**: Centralized configuration with environment variable support.

```python
# Environment Variable Prefix: VIN_

# Example overrides:
# VIN_CLAHE_CLIP_LIMIT=3.0
# VIN_USE_GPU=false
# VIN_LOG_LEVEL=DEBUG

# Configuration Classes
@dataclass
class PreprocessingConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 5
    max_image_dimension: int = 4096
    default_mode: str = 'engraved'

@dataclass
class OCRConfig:
    language: str = 'en'
    det_db_box_thresh: float = 0.3
    rec_thresh: float = 0.3
    use_gpu: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 0.0001
    augmentation_multiplier: int = 50

@dataclass
class PipelineConfig:
    preprocessing: PreprocessingConfig
    ocr: OCRConfig
    training: TrainingConfig
    logging: LoggingConfig

# Usage
from config import get_config
config = get_config()
print(config.preprocessing.clahe_clip_limit)  # 2.0
```

### 5.3 `vin_pipeline.py` - Core Pipeline

**Purpose**: Main OCR pipeline combining preprocessing, recognition, and postprocessing.

```python
class VINOCRPipeline:
    """Main pipeline for VIN recognition."""
    
    def __init__(self, preprocess_mode: str = 'engraved'):
        self.preprocessor = VINImagePreprocessor(mode=preprocess_mode)
        self.ocr = PaddleOCR(...)
        self.postprocessor = VINPostProcessor()
    
    def recognize(self, image_path: str) -> Dict:
        """Recognize VIN from image."""
        image = self._load_image(image_path)
        enhanced = self.preprocessor.preprocess(image)
        raw_result = self.ocr.ocr(enhanced)
        text, conf = self._extract_best_result(raw_result)
        return self.postprocessor.process(text, conf)
    
    def batch_recognize(self, image_paths: List[str]) -> List[Dict]:
        """Process multiple images."""
        return [self.recognize(p) for p in image_paths]


class VINImagePreprocessor:
    """Image enhancement for engraved metal plates."""
    
    MODES = ['none', 'fast', 'balanced', 'engraved']
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.mode == 'engraved':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
            return denoised
        ...


class VINPostProcessor:
    """VIN-specific OCR correction."""
    
    def process(self, raw_text: str, confidence: float) -> Dict:
        corrector = RuleBasedCorrector()
        return corrector.correct(raw_text, confidence)
```

---

## 6. Configuration System

### 6.1 Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION HIERARCHY                               │
└─────────────────────────────────────────────────────────────────────────────┘

  Priority (highest to lowest):
  
  1. ┌─────────────────────────────────────────┐
     │ Environment Variables                   │  VIN_CLAHE_CLIP_LIMIT=3.0
     │ (Runtime overrides)                     │  VIN_USE_GPU=false
     └─────────────────────────────────────────┘
                        │
                        ▼
  2. ┌─────────────────────────────────────────┐
     │ config.json (if present)               │  {"preprocessing": {"clahe_clip_limit": 2.5}}
     │ (Project-level customization)           │
     └─────────────────────────────────────────┘
                        │
                        ▼
  3. ┌─────────────────────────────────────────┐
     │ Dataclass Defaults                      │  @dataclass PreprocessingConfig:
     │ (Hardcoded in config.py)                │      clahe_clip_limit: float = 2.0
     └─────────────────────────────────────────┘
```

### 6.2 Environment Variables Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VIN_CLAHE_CLIP_LIMIT` | float | 2.0 | CLAHE contrast limiting |
| `VIN_MAX_IMAGE_DIM` | int | 4096 | Max image dimension (OOM protection) |
| `VIN_PREPROCESS_MODE` | str | 'engraved' | Default preprocessing mode |
| `VIN_DET_BOX_THRESH` | float | 0.3 | OCR detection threshold |
| `VIN_REC_THRESH` | float | 0.3 | OCR recognition threshold |
| `VIN_USE_GPU` | bool | true | Enable GPU acceleration |
| `VIN_LOG_LEVEL` | str | 'INFO' | Logging verbosity |
| `VIN_AUGMENTATION_THRESHOLD` | int | 50 | Min images before augmentation |
| `VIN_AUGMENTATION_MULTIPLIER` | int | 50 | Augmentation factor |

---

## 7. Processing Pipeline Details

### 7.1 Preprocessing Stage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input: BGR Image (numpy.ndarray)                                          │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 1: GRAYSCALE CONVERSION                                        │   │
│   │                                                                      │   │
│   │   cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                           │   │
│   │                                                                      │   │
│   │   Why: Reduces 3 channels to 1, focuses on intensity                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)    │   │
│   │                                                                      │   │
│   │   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))       │   │
│   │   enhanced = clahe.apply(gray)                                      │   │
│   │                                                                      │   │
│   │   Why: Enhances local contrast in engraved characters               │   │
│   │   Parameters:                                                        │   │
│   │   • clipLimit=2.0: Prevents over-amplification of noise             │   │
│   │   • tileGridSize=(8,8): 64 regions for local adaptation            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 3: BILATERAL FILTER (Denoising)                                │   │
│   │                                                                      │   │
│   │   denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50,     │   │
│   │                                  sigmaSpace=50)                     │   │
│   │                                                                      │   │
│   │   Why: Reduces noise while preserving character edges               │   │
│   │   Parameters:                                                        │   │
│   │   • d=5: Diameter of pixel neighborhood                             │   │
│   │   • sigmaColor=50: Color similarity weight                          │   │
│   │   • sigmaSpace=50: Spatial proximity weight                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Output: Enhanced Grayscale Image                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Rule-Based Correction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RULE-BASED CORRECTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input: Raw OCR Text (e.g., "**SALIA2A4OSA6O6662#")                        │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 1: ARTIFACT REMOVAL                                            │   │
│   │                                                                      │   │
│   │   Patterns removed:                                                  │   │
│   │   • ^[*#XYT]+     → Start artifacts                                 │   │
│   │   • [*#]+$        → End artifacts                                   │   │
│   │   • ^[IYTFA][*#]+ → Prefix + artifacts                              │   │
│   │                                                                      │   │
│   │   "**SALIA2A4OSA6O6662#" → "SALIA2A4OSA6O6662"                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 2: INVALID CHARACTER REPLACEMENT                               │   │
│   │                                                                      │   │
│   │   VIN prohibits I, O, Q (confusion with 1, 0):                      │   │
│   │   • I → 1                                                            │   │
│   │   • O → 0                                                            │   │
│   │   • Q → 0                                                            │   │
│   │                                                                      │   │
│   │   "SALIA2A4OSA6O6662" → "SAL1A2A40SA606662"                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 3: POSITION-BASED CORRECTION                                   │   │
│   │                                                                      │   │
│   │   VIN Structure:                                                     │   │
│   │   Position: 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17      │   │
│   │             └─ WMI ─┘  └──── VDS ────┘ │  │  │  └── SERIAL ──┘     │   │
│   │                                       chk yr plt   (must be digits) │   │
│   │                                                                      │   │
│   │   Positions 12-17 letter→digit corrections:                         │   │
│   │   • S → 5, G → 6, B → 8, A → 4, L → 1, Z → 2                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 4: CHECKSUM VALIDATION                                         │   │
│   │                                                                      │   │
│   │   ISO 3779 / NHTSA checksum at position 9:                          │   │
│   │                                                                      │   │
│   │   weights = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]                   │   │
│   │   values  = {A:1, B:2, ..., 0:0, 1:1, ...}                         │   │
│   │   sum     = Σ (value[char[i]] × weight[i])                          │   │
│   │   check   = sum mod 11  (10 becomes 'X')                            │   │
│   │                                                                      │   │
│   │   Validation: vin[8] == calculated_check_digit                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Output: Corrected VIN "SAL1A2A40SA606662"                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Training & Evaluation Pipeline

### 8.1 End-to-End Experiment Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      run_experiment.py ORCHESTRATION                         │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ PHASE 1: DATA PREPARATION                          prepare_dataset.py    │
  ├──────────────────────────────────────────────────────────────────────────┤
  │                                                                           │
  │  data/                                                                    │
  │  └── *.jpg  ──────▶  extract_vin_from_filename()  ──────▶  Split:       │
  │                                                            ├── train/ 70%│
  │                                                            ├── val/   15%│
  │                                                            └── test/  15%│
  │                                                                           │
  │  Output: dataset/{train,val,test}/ + *_labels.txt                        │
  └────────────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ PHASE 2: BASELINE EVALUATION                            evaluate.py      │
  ├──────────────────────────────────────────────────────────────────────────┤
  │                                                                           │
  │  Run pretrained PP-OCRv5 on test split                                   │
  │  Calculate: Exact Match, F1, CER, NED, Per-position accuracy             │
  │                                                                           │
  │  Output: baseline_metrics.json                                            │
  └────────────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ PHASE 3: RULE LEARNING (Optional)                  train_pipeline.py     │
  ├──────────────────────────────────────────────────────────────────────────┤
  │                                                                           │
  │  ⚠️  NOTE: This is NOT neural network fine-tuning!                       │
  │                                                                           │
  │  If train images < 50:                                                    │
  │    └── Augment to 50× (rotation, brightness, blur, contrast)             │
  │                                                                           │
  │  Rule Learning (NOT transfer learning):                                   │
  │    1. Run pretrained PP-OCRv5 on all training images                     │
  │    2. Compare predictions to ground truth                                 │
  │    3. Build character confusion matrix                                    │
  │    4. Generate correction rules from most common errors                   │
  │                                                                           │
  │  Output: model.json (with learned rules), NOT model weights              │
  └────────────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ PHASE 4: FINAL EVALUATION                               evaluate.py      │
  ├──────────────────────────────────────────────────────────────────────────┤
  │                                                                           │
  │  Evaluate on all splits with learned corrections:                         │
  │  ├── Train split (sanity check)                                          │
  │  ├── Val split (hyperparameter selection)                                │
  │  └── Test split (final reported metrics)                                 │
  │                                                                           │
  │  Output: {train,val,test}_metrics.json, experiment_summary.json          │
  └──────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Metrics Calculation

| Metric | Formula | Description |
|--------|---------|-------------|
| **Exact Match** | `correct_vins / total_vins` | % of VINs with 100% correct |
| **Precision** | `TP / (TP + FP)` | Character-level precision |
| **Recall** | `TP / (TP + FN)` | Character-level recall |
| **F1 Score** | `2 × P × R / (P + R)` | Harmonic mean |
| **CER** | `(S + D + I) / N` | Character Error Rate |
| **NED** | `edit_distance / max_length` | Normalized Edit Distance |
| **Position Accuracy** | `correct[i] / total[i]` | Per-position (1-17) |

---

## 9. Error Handling Strategy

### 9.1 Error Categories and Handling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ERROR HANDLING MATRIX                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Category          │ Example               │ Handling                       │
│   ──────────────────┼───────────────────────┼───────────────────────────────│
│   File I/O          │ Image not found       │ Return error dict, log warning│
│   Image Format      │ Corrupted JPEG        │ Return error dict, log warning│
│   Image Content     │ Empty/blank image     │ Raise ValueError              │
│   OCR Failure       │ No text detected      │ Return empty VIN, low conf    │
│   Validation        │ Invalid VIN length    │ Return with is_valid=False    │
│   Configuration     │ Invalid env var       │ Log warning, use default      │
│   Training          │ Augmentation error    │ Log warning, skip sample      │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ CRITICAL RULE: No bare `except:` - always capture exception type    │   │
│   │                                                                      │   │
│   │ ✗ BAD:   except:                                                    │   │
│   │              pass                                                    │   │
│   │                                                                      │   │
│   │ ✓ GOOD:  except Exception as e:                                     │   │
│   │              logger.warning(f"Error processing {path}: {e}")        │   │
│   │              continue                                                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Error Response Structure

```python
# Success response
{
    'vin': 'SAL1A2A40SA606662',
    'confidence': 0.87,
    'is_valid_length': True,
    'checksum_valid': True,
    'corrections': ['I→1 at pos 4'],
    'error': None
}

# Error response
{
    'vin': '',
    'confidence': 0.0,
    'is_valid_length': False,
    'checksum_valid': False,
    'corrections': [],
    'error': 'Could not load image: FileNotFoundError'
}
```

---

## 10. Performance Characteristics

### 10.1 Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Throughput (CPU)** | 0.3-0.5 img/sec | Intel i7, single thread |
| **Throughput (GPU)** | 5-10 img/sec | NVIDIA T4 |
| **Model Load Time** | 3-5 sec | First call only |
| **Memory (Idle)** | ~500 MB | After model load |
| **Memory (Peak)** | ~1 GB | During inference |
| **Preprocessing** | 10-20 ms/img | CLAHE + bilateral |
| **OCR Inference** | 2-3 sec/img (CPU) | PP-OCRv5 |
| **Postprocessing** | <1 ms/img | Rule application |

### 10.2 Accuracy Metrics (Observed)

| Metric | Baseline (PP-OCRv5) | With Pipeline | Improvement |
|--------|---------------------|---------------|-------------|
| Exact Match | 5% | 25% | +400% |
| F1 Score | 43% | 55% | +28% |
| CER | 12% | 8% | -33% |

---

## 11. Extension Points

### 11.1 Adding New Preprocessing Modes

```python
# In vin_pipeline.py

class VINImagePreprocessor:
    MODES = ['none', 'fast', 'balanced', 'engraved', 'custom']  # Add here
    
    def preprocess(self, image):
        if self.mode == 'custom':
            # Implement custom preprocessing
            return self._custom_preprocess(image)
        ...
    
    def _custom_preprocess(self, image):
        # Your implementation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ... custom operations
        return processed
```

### 11.2 Adding Correction Rules

```python
# In vin_utils.py

class RuleBasedCorrector:
    # Add to SEQUENTIAL_POSITION_RULES
    SEQUENTIAL_POSITION_RULES = {
        'S': '5',
        'G': '6',
        'YOUR_CHAR': 'REPLACEMENT',  # Add here
        ...
    }
    
    # Or add learned rules at runtime
    corrector = RuleBasedCorrector()
    corrector.add_learned_rules({'X': 'K'})
```

### 11.3 Custom Configuration

```python
# Via environment variables
export VIN_CLAHE_CLIP_LIMIT=3.0
export VIN_USE_GPU=false

# Via code
from config import get_config, PipelineConfig

config = get_config()
config.preprocessing.clahe_clip_limit = 3.0
```

### 11.4 Adding New Metrics

```python
# In evaluate.py

def calculate_metrics(predictions):
    # Add custom metric
    custom_metric = calculate_custom(predictions)
    
    return {
        'exact_match': ...,
        'f1_score': ...,
        'custom_metric': custom_metric,  # Add here
    }
```

---

## Appendix A: VIN Specification Reference

### A.1 VIN Structure (ISO 3779)

```
Position:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
           └─ WMI ─┘  └───── VDS ─────┘  │  └────── VIS ──────┘
                                        Check
                                        Digit

WMI (1-3):  World Manufacturer Identifier
            • Position 1: Country/region
            • Position 2: Manufacturer
            • Position 3: Division/type

VDS (4-8):  Vehicle Descriptor Section
            • Positions 4-8: Vehicle attributes (model, body, engine)

Check (9):  Check Digit
            • Calculated from all other positions
            • Value: 0-9 or X (for 10)

VIS (10-17): Vehicle Identifier Section
            • Position 10: Model year
            • Position 11: Plant code
            • Positions 12-17: Sequential number (usually digits only)
```

### A.2 Invalid Characters

| Character | Reason | Replacement |
|-----------|--------|-------------|
| I | Confused with 1 | 1 |
| O | Confused with 0 | 0 |
| Q | Confused with 0 | 0 |

### A.3 Check Digit Calculation

```python
WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
VALUES = {
    'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8,
    'J':1, 'K':2, 'L':3, 'M':4, 'N':5, 'P':7, 'R':9,
    'S':2, 'T':3, 'U':4, 'V':5, 'W':6, 'X':7, 'Y':8, 'Z':9,
    '0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9
}

def calculate_check_digit(vin):
    total = sum(VALUES[c] * WEIGHTS[i] for i, c in enumerate(vin) if i != 8)
    remainder = total % 11
    return 'X' if remainder == 10 else str(remainder)
```

---

## Appendix B: File Format Specifications

### B.1 Image Naming Convention

```
{NUMBER}-VIN -{VINCODE}.{ext}

Examples:
  1-VIN -SAL1A2A40SA606662.jpg
  42-VIN -1HGBH41JXMN109186.png
  100-VIN -WVWZZZ3CZWE123456.jpeg
```

### B.2 PaddleOCR Label Format

```
# labels.txt
train/1-VIN -SAL1A2A40SA606662.jpg	SAL1A2A40SA606662
train/42-VIN -1HGBH41JXMN109186.jpg	1HGBH41JXMN109186
```

### B.3 Metrics Output Format (JSON)

```json
{
  "split": "test",
  "sample_count": 15,
  "exact_match_rate": 0.25,
  "f1_score": 0.55,
  "precision": 0.58,
  "recall": 0.52,
  "character_error_rate": 0.08,
  "normalized_edit_distance": 0.12,
  "position_accuracy": {
    "1": 0.95, "2": 0.90, "3": 0.85, ...
  },
  "confusion_matrix": {
    "I": {"1": 12, "L": 2},
    "O": {"0": 15}
  }
}
```

---

*End of Architecture Documentation*
