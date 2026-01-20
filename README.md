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
5. [Pipeline Architecture](#pipeline-architecture)
6. [Configuration](#configuration)
7. [Character Confusion Handling](#character-confusion-handling)
8. [VIN Format Reference](#vin-format-reference)
9. [For the Team](#for-the-team)
10. [License](#license)

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

```bash
pip install -r requirements.txt
```

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
├── tests/
│   └── test_vin_pipeline.py  # Test suite (52+ tests)
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

**Apache License 2.0** - See LICENSE and NOTICE files

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
python validate_dataset.py --data-dir /path/to/paddleocr_sample

# Output report to JSON
python validate_dataset.py --data-dir /path/to/data --output report.json
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
python evaluate.py --data-dir /path/to/images

# Create 70/15/15 splits and evaluate test set
python evaluate.py --data-dir /path/to/images --create-splits --split test

# Evaluate validation set with existing splits
python evaluate.py --data-dir /path/to/images --split-dir ./splits --split val

# Export results
python evaluate.py --data-dir /path/to/images --output results.json --csv predictions.csv

# Quick test with limited samples
python evaluate.py --data-dir /path/to/images --max-samples 50
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
