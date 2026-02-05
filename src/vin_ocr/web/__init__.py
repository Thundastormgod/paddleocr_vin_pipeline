"""
VIN OCR Web Module
==================

Streamlit-based web interface for VIN OCR pipeline.

Features:
- Single image VIN recognition
- Batch processing
- Model evaluation and comparison
- Training management (when available)

Usage:
    streamlit run src/vin_ocr/web/app.py
    
    # Or from project root:
    ./run_streamlit.bat  (Windows)
    streamlit run src/vin_ocr/web/app.py  (macOS/Linux)
"""

# Training UI components
from .training_components import (
    TrainingUI,
    ProgressTracker,
    TrainingRunner,
    TrainingState,
    TrainingUpdate,
    get_global_tracker,
    get_global_runner,
)

# Main app is standalone Streamlit application
# Run with: streamlit run src/vin_ocr/web/app.py

__all__ = [
    "app",
    "TrainingUI",
    "ProgressTracker",
    "TrainingRunner",
    "TrainingState",
    "TrainingUpdate",
    "get_global_tracker",
    "get_global_runner",
]
