#!/usr/bin/env python3
"""
PaddleOCR Training Runner
========================

Convenience script to run PaddleOCR fine-tuning from project root.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Run the training module
if __name__ == "__main__":
    from src.vin_ocr.training.finetune_paddleocr import main
    main()
