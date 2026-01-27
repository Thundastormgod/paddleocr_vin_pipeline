"""
VIN OCR Web UI Package
======================

Streamlit-based web interface for VIN recognition.

Usage:
    streamlit run web_ui/app.py
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ["app"]

WEB_UI_DIR = Path(__file__).parent
