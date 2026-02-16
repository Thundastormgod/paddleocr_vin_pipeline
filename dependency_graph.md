# VIN OCR Pipeline - Complete Dependency Graph

## Overview
This document outlines the complete import dependency graph for the VIN OCR Pipeline project, with special focus on the WebUI components and their relationships.

## Project Structure & Dependencies

```
paddleocr_vin_pipeline/
├── src/vin_ocr/                    # Main package
│   ├── __init__.py                 # Core exports, lazy imports
│   ├── cli.py                      # CLI entry point
│   ├── core/                       # VIN utilities & validation
│   ├── pipeline/                   # Main recognition pipeline
│   ├── inference/                  # Paddle & ONNX inference engines
│   ├── training/                   # Training scripts & components
│   ├── evaluation/                 # Metrics & evaluation tools
│   ├── providers/                  # OCR backend providers
│   ├── preprocessing/              # Image preprocessing
│   ├── utils/                      # Data preparation & utilities
│   └── web/                        # Streamlit web UI
├── config.py                       # Centralized configuration
├── requirements.txt                # Dependencies
└── pyproject.toml                 # Python packaging
```

## Core Dependencies

### 1. **Configuration System** (`config.py`)
```python
# External dependencies
import os, json, logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# No internal dependencies - root of config system
```

### 2. **Core VIN Utilities** (`src/vin_ocr/core/vin_utils.py`)
```python
# External dependencies
import re, logging
from typing import Optional, Dict, List, Tuple, FrozenSet
from dataclasses import dataclass, field
from enum import Enum

# No internal dependencies - single source of truth for VIN logic
```

## Main Pipeline Dependencies

### 3. **VIN Recognition Pipeline** (`src/vin_ocr/pipeline/vin_pipeline.py`)
```python
# External dependencies
import cv2, numpy as np, re, logging, time
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# Internal dependencies
from config import get_config                                    # [2]
from ..preprocessing import VINPreprocessor, PreprocessConfig, PreprocessStrategy  # [8]
from ..core.vin_utils import VINConstants                       # [2]

# Optional PaddleOCR import
try:
    from paddleocr import PaddleOCR
except ImportError:
    PADDLEOCR_AVAILABLE = False
```

## Inference Layer Dependencies

### 4. **Paddle Inference** (`src/vin_ocr/inference/paddle_inference.py`)
```python
# External dependencies
import os, cv2, numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Internal dependencies
# None - standalone inference engine
```

### 5. **ONNX Inference** (`src/vin_ocr/inference/onnx_inference.py`)
```python
# External dependencies
import onnxruntime as ort
import cv2, numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Internal dependencies
# None - standalone ONNX inference
```

## Provider Layer Dependencies

### 6. **OCR Providers** (`src/vin_ocr/providers/ocr_providers.py`)
```python
# External dependencies
import os, base64, time, logging, numpy as np
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# Internal dependencies
from ..core.vin_utils import VINConstants                      # [2]
from config import get_config                                  # [2]
from ..preprocessing import VINPreprocessor, PreprocessConfig, PreprocessStrategy  # [8]

# Optional DeepSeek import
try:
    import requests
    from transformers import AutoTokenizer, AutoProcessor
except ImportError:
    DEEPSEEK_AVAILABLE = False
```

## Preprocessing Dependencies

### 7. **VIN Preprocessing** (`src/vin_ocr/preprocessing/vin_preprocessor.py`)
```python
# External dependencies
import cv2, numpy as np, logging
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Any

# Internal dependencies
from config import get_config                                  # [2]
```

## Training Layer Dependencies

### 8. **Training Components** (`src/vin_ocr/web/training_components.py`)
```python
# External dependencies
import json, os, signal, time, threading, subprocess, sys
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# Platform-specific imports
try:
    import fcntl  # Unix-only
except ImportError:
    try:
        import msvcrt  # Windows
    except ImportError:
        pass

# Internal dependencies
# None - standalone training management
```

### 9. **PaddleOCR Fine-tuning** (`src/vin_ocr/training/finetune_paddleocr.py`)
```python
# External dependencies
import os, sys, json, logging, argparse
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from paddle.vision.transforms import Compose
import cv2, numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Internal dependencies
from config import get_config                                  # [2]
from ..core.vin_utils import VINConstants, extract_vin_from_filename  # [2]
from ..preprocessing import VINPreprocessor, PreprocessStrategy  # [8]
```

## WebUI Dependencies - CRITICAL PATH

### 10. **Main Web Application** (`src/vin_ocr/web/app.py`)
```python
# External dependencies
import os, sys, json, shutil, random, tempfile, logging
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Optional external dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Internal dependencies - CORE WEBUI IMPORTS
from config import get_config                                  # [2]
from src.vin_ocr.core.vin_utils import extract_vin_from_filename, is_valid_vin  # [2]

# Optional internal dependencies with graceful fallbacks
try:
    from src.vin_ocr.providers.ocr_providers import DeepSeekOCRProvider, DeepSeekOCRConfig  # [6]
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

try:
    from src.vin_ocr.inference.paddle_inference import VINInference          # [4]
    from src.vin_ocr.inference.onnx_inference import ONNXVINRecognizer        # [5]
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False

try:
    from src.vin_ocr.training.finetune_paddleocr import VINFineTuner, TRAINING_AVAILABLE  # [9]
except ImportError:
    TRAINING_AVAILABLE = False

try:
    from src.vin_ocr.training.hyperparameter_tuning import HyperparameterTuner
    HYPERPARAMETER_TUNING_AVAILABLE = True
except ImportError:
    HYPERPARAMETER_TUNING_AVAILABLE = False

try:
    from src.vin_ocr.web.training_components import (                     # [8]
        get_global_runner,
        get_global_tracker,
        TrainingRunner,
        ProgressTracker,
    )
    TRAINING_COMPONENTS_AVAILABLE = True
except ImportError:
    TRAINING_COMPONENTS_AVAILABLE = False
```

## Evaluation Dependencies

### 11. **Multi-Model Evaluation** (`src/vin_ocr/evaluation/multi_model_evaluation.py`)
```python
# External dependencies
import os, json, logging, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Internal dependencies
from config import get_config                                  # [2]
from ..core.vin_utils import extract_vin_from_filename, validate_vin  # [2]
from ..pipeline.vin_pipeline import VINOCRPipeline              # [3]
from ..providers.ocr_providers import OCRProviderFactory        # [6]
from ..inference.paddle_inference import VINInference           # [4]
from ..inference.onnx_inference import ONNXVINRecognizer        # [5]
```

## Utility Dependencies

### 12. **Dataset Preparation** (`src/vin_ocr/utils/prepare_dataset.py`)
```python
# External dependencies
import os, json, logging, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Internal dependencies
from config import get_config                                  # [2]
from ..core.vin_utils import extract_vin_from_filename, validate_vin  # [2]
```

### 13. **Dataset Validation** (`src/vin_ocr/utils/validate_dataset.py`)
```python
# External dependencies
import os, json, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

# Internal dependencies
from config import get_config                                  # [2]
from ..core.vin_utils import extract_vin_from_filename, validate_vin  # [2]
```

## CLI Dependencies

### 14. **CLI Entry Point** (`src/vin_ocr/cli.py`)
```python
# External dependencies
import argparse, sys
from pathlib import Path

# Internal dependencies
from src.vin_ocr.inference import VINInference, ONNXVINRecognizer  # [4,5]
from src.vin_ocr.web.app import cmd_serve                         # [10]
```

## Package Entry Points

### 15. **Main Package Init** (`src/vin_ocr/__init__.py`)
```python
# External dependencies
# None

# Internal dependencies - Core exports (lightweight)
from .core import (
    VINConstants, VIN_LENGTH, VIN_VALID_CHARS, VINValidationResult,
    validate_vin, validate_vin_format, extract_vin_from_filename,
    extract_vin_from_text, calculate_check_digit, validate_checksum,
    levenshtein_distance, correct_vin,
)

# Lazy imports for inference (heavier dependencies)
def __getattr__(name: str):
    if name == "VINInference":
        from .inference.paddle_inference import VINInference      # [4]
        return VINInference
    elif name == "ONNXVINRecognizer":
        from .inference.onnx_inference import ONNXVINRecognizer    # [5]
        return ONNXVINRecognizer
```

## WebUI Critical Dependency Path

The WebUI (`src/vin_ocr/web/app.py`) has the most complex dependency tree:

```
WebUI (app.py) [10]
├── config.py [2] ✅ REQUIRED
├── core.vin_utils [2] ✅ REQUIRED  
├── providers.ocr_providers [6] ⚠️ OPTIONAL (DeepSeek)
├── inference.paddle_inference [4] ⚠️ OPTIONAL
├── inference.onnx_inference [5] ⚠️ OPTIONAL
├── training.finetune_paddleocr [9] ⚠️ OPTIONAL
├── training.hyperparameter_tuning ⚠️ OPTIONAL
├── web.training_components [8] ⚠️ OPTIONAL
└── External: streamlit, pandas, PIL, plotly
```

## Dependency Risk Analysis

### **High Risk** (Required for basic functionality)
- `config.py` - Central configuration, no alternatives
- `core.vin_utils` - VIN validation logic, no alternatives
- `streamlit` - WebUI framework, no alternatives

### **Medium Risk** (Optional but impacts functionality)
- `inference.*` - Without these, no model inference
- `providers.ocr_providers` - Without this, no DeepSeek support
- `training_components` - Without this, no training UI

### **Low Risk** (Nice-to-have features)
- `plotly` - Charts and visualizations
- `PIL` - Image preview functionality
- `training.*` - Training functionality

## Circular Dependency Prevention

The project avoids circular dependencies through:

1. **Single Source of Truth**: `core.vin_utils` has no internal dependencies
2. **Configuration Root**: `config.py` has no internal dependencies  
3. **Layered Architecture**: Clear dependency direction from bottom to top
4. **Lazy Imports**: Heavy modules loaded on-demand in `__init__.py`
5. **Optional Imports**: WebUI gracefully handles missing dependencies

## Installation Impact

### **Minimal Installation** (Basic WebUI only)
```bash
pip install streamlit pandas opencv-python numpy
```

### **Full Installation** (All features)
```bash
pip install -r requirements.txt
# Includes: paddlepaddle, paddleocr, onnxruntime, plotly, PIL, transformers, etc.
```

### **Development Installation**
```bash
pip install -r requirements.txt
pip install pytest black isort mypy pre-commit
```

## External Service Dependencies

### **Optional External Services**
- **DeepSeek API**: Requires API key and internet connection
- **GPU Acceleration**: Requires CUDA-compatible NVIDIA GPU
- **HPC Clusters**: For large-scale training (RTX 3090+)

### **No External Dependencies**
- All core functionality works offline
- No required API keys or services
- Self-contained training and inference

This dependency graph shows a well-architected system with clear separation of concerns, graceful degradation, and optional dependencies that enhance functionality without breaking the core system.
