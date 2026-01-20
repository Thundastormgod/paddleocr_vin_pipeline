# PaddleOCR VIN Pipeline - Architecture Documentation

## Overview

This document describes the internal architecture of the VIN OCR Pipeline, designed for recognizing Vehicle Identification Numbers from engraved metal plates.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VINOCRPipeline                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────┐                                                         │
│   │   Input Image   │                                                         │
│   │  (BGR/Gray)     │                                                         │
│   └───────┬────────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                    PREPROCESSING STAGE                              │    │
│   ├────────────────────────────────────────────────────────────────────┤    │
│   │                                                                     │    │
│   │   ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐      │    │
│   │   │  Grayscale  │───▶│    CLAHE    │───▶│ Bilateral Filter │      │    │
│   │   │ Conversion  │    │  (Contrast) │    │   (Denoise)      │      │    │
│   │   └─────────────┘    └─────────────┘    └──────────────────┘      │    │
│   │                                                                     │    │
│   │   Mode Options:                                                     │    │
│   │   • none     - Skip all preprocessing                              │    │
│   │   • fast     - Normalize only                                      │    │
│   │   • balanced - CLAHE only                                          │    │
│   │   • engraved - CLAHE + Bilateral (recommended)                     │    │
│   │                                                                     │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│           │                                                                  │
│           ▼                                                                  │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      OCR STAGE (PaddleOCR)                          │    │
│   ├────────────────────────────────────────────────────────────────────┤    │
│   │                                                                     │    │
│   │   Model: PP-OCRv4 (English)                                        │    │
│   │                                                                     │    │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │    │
│   │   │  Detection  │───▶│ Recognition │───▶│  Output     │           │    │
│   │   │  (DB++)     │    │  (SVTR)     │    │ (text,conf) │           │    │
│   │   └─────────────┘    └─────────────┘    └─────────────┘           │    │
│   │                                                                     │    │
│   │   Optimized Settings:                                               │    │
│   │   • det_db_box_thresh = 0.3                                        │    │
│   │   • rec_thresh = 0.3                                               │    │
│   │   • use_doc_orientation_classify = False                           │    │
│   │   • use_doc_unwarping = False                                      │    │
│   │                                                                     │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│           │                                                                  │
│           ▼                                                                  │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                   POSTPROCESSING STAGE                              │    │
│   ├────────────────────────────────────────────────────────────────────┤    │
│   │                                                                     │    │
│   │   1. ARTIFACT REMOVAL                                               │    │
│   │      ├── Remove prefix: *, #, X, Y, T, F, A, I                     │    │
│   │      └── Remove suffix: *, #                                       │    │
│   │                                                                     │    │
│   │   2. INVALID CHARACTER FIX                                          │    │
│   │      ├── I → 1 (I not allowed in VIN)                              │    │
│   │      ├── O → 0 (O not allowed in VIN)                              │    │
│   │      └── Q → 0 (Q not allowed in VIN)                              │    │
│   │                                                                     │    │
│   │   3. POSITION-BASED CORRECTION                                      │    │
│   │      └── Positions 12-17 (sequential): Prefer digits               │    │
│   │          ├── S → 5                                                  │    │
│   │          ├── G → 6                                                  │    │
│   │          ├── B → 8                                                  │    │
│   │          ├── A → 4                                                  │    │
│   │          ├── L → 1                                                  │    │
│   │          └── Z → 2                                                  │    │
│   │                                                                     │    │
│   │   4. VALIDATION                                                     │    │
│   │      ├── Length check (must be 17)                                 │    │
│   │      └── Checksum validation (position 9)                          │    │
│   │                                                                     │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│           │                                                                  │
│           ▼                                                                  │
│   ┌────────────────┐                                                         │
│   │    OUTPUT      │                                                         │
│   │  {vin, conf,   │                                                         │
│   │   valid, etc}  │                                                         │
│   └────────────────┘                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. VINImagePreprocessor

**Purpose**: Enhance image quality for better OCR recognition on engraved metal surfaces.

**Key Technique**: CLAHE (Contrast Limited Adaptive Histogram Equalization)

```python
# CLAHE Parameters
clahe_clip_limit = 2.0      # Contrast limiting
clahe_tile_size = (8, 8)    # Grid size for local enhancement
```

**Why CLAHE for Engraved Plates?**
- Engraved metal has low contrast between raised/recessed areas
- CLAHE enhances local contrast without over-amplifying noise
- Preserves character edges while improving visibility

**Processing Modes**:

| Mode | Operations | Best For |
|------|------------|----------|
| `none` | Pass-through | Clean, high-quality images |
| `fast` | Normalize | Quick processing, decent images |
| `balanced` | CLAHE | General use |
| `engraved` | CLAHE + Bilateral | Metal plates (recommended) |

### 2. PaddleOCR Engine

**Model**: PP-OCRv4 (English)

**Architecture Components**:
- **Detection**: DB++ (Differentiable Binarization)
- **Recognition**: SVTR (Scene Text Recognition Transformer)

**Optimized Configuration**:

```python
PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False,  # Disabled - VIN plates are aligned
    use_doc_unwarping=False,             # Disabled - metal plates are flat
    use_textline_orientation=False,      # Disabled - not needed for VIN
    det_db_box_thresh=0.3,               # Lower threshold catches more text
    rec_thresh=0.3,                      # Accept lower confidence results
)
```

**Why These Settings?**
- Document orientation and unwarping add 5-10x overhead
- VIN plates are typically photographed straight-on
- Lower thresholds help with faint engravings

### 3. VINPostProcessor

**Purpose**: Correct OCR errors specific to VIN format.

#### 3.1 Artifact Removal

Common artifacts from engraved plates:
- `*` - Separator characters
- `#` - Boundary markers
- `X`, `Y`, `T`, `F`, `A`, `I` - Misread plate borders

```python
# Regex patterns
ARTIFACT_PATTERNS = [
    r'^[*#XYT]+',     # Start artifacts
    r'[*#]+$',         # End artifacts
    r'^[IYTFA][*#]*',  # Common prefixes
]
```

#### 3.2 Invalid Character Correction

VIN specification prohibits I, O, Q to avoid confusion:

| Invalid | Replacement | Reason |
|---------|-------------|--------|
| I | 1 | Looks like 1 |
| O | 0 | Looks like 0 |
| Q | 0 | Looks like 0 |

#### 3.3 Position-Based Correction

VIN structure knowledge helps correct ambiguous characters:

```
Position:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
           └─ WMI ─┘  └──── VDS ────┘  │  │  │  └── SEQUENTIAL ─┘
                                      Check Year Plant
```

**Positions 12-17 (Sequential Number)**:
- Should always be digits
- Apply letter→digit corrections here

#### 3.4 Checksum Validation

VIN has built-in error detection at position 9:

```python
# Weight by position
WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

# Character values
VALUES = {'A':1, 'B':2, ..., '0':0, '1':1, ...}

# Algorithm
sum = Σ (VALUE[char[i]] × WEIGHT[i])
check_digit = sum mod 11  # 10 = 'X'
```

## Data Flow

```
Input Image
    │
    ├──[Load]──────────────▶ numpy.ndarray (BGR)
    │
    ├──[Preprocess]────────▶ numpy.ndarray (Gray, enhanced)
    │
    ├──[OCR]───────────────▶ dict: {rec_texts: [...], rec_scores: [...]}
    │
    ├──[Extract]───────────▶ (text: str, confidence: float)
    │
    ├──[Postprocess]───────▶ dict: {vin, confidence, corrections, ...}
    │
    └──[Return]────────────▶ Final Result
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Processing Time | 2-3 sec/image (CPU) |
| Detection Rate | 99.7% |
| Exact Match (baseline) | 5% |
| Exact Match (with pipeline) | 25% |
| F1 Score (baseline) | 43% |
| F1 Score (with pipeline) | 55% |

## Error Handling

```python
# Image load failure
if image is None:
    return {'error': 'Could not load image', ...}

# No text detected
if not result or len(result) == 0:
    return ('', 0.0)

# Invalid VIN length
is_valid_length = len(vin) == 17
```

## Extension Points

### Adding New Preprocessing Modes

```python
def preprocess(self, image):
    if self.mode == 'custom':
        # Your custom preprocessing
        return custom_process(image)
```

### Adding Character Corrections

```python
# In VINPostProcessor._apply_position_corrections()
custom_fixes = {
    'Z': '2',  # Additional confusion
}
```

### Custom Artifact Patterns

```python
ARTIFACT_PATTERNS.append(r'your_pattern')
```

## Thread Safety

- `VINOCRPipeline` is **not thread-safe** due to PaddleOCR internals
- Create separate instances for parallel processing
- Or use a lock around `recognize()` calls

## Memory Usage

- Model loading: ~500MB (first use)
- Per-image: ~50-100MB depending on resolution
- Models are cached after first load
