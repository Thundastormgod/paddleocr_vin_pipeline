"""
VIN OCR Training Module
=======================

Training and fine-tuning pipelines for VIN OCR models.

Available Scripts:
- finetune_paddleocr.py: Fine-tune PaddleOCR models on VIN dataset
- finetune_deepseek.py: Fine-tune DeepSeek VL models with LoRA
- train_from_scratch.py: Train PaddleOCR from scratch

Usage (CLI):
    # Fine-tune PaddleOCR
    python -m src.vin_ocr.training.finetune_paddleocr --config configs/vin_finetune_config.yml
    
    # Fine-tune DeepSeek
    python -m src.vin_ocr.training.finetune_deepseek --data_dir ./data/train
    
    # Train from scratch
    python -m src.vin_ocr.training.train_from_scratch --data_dir ./data/train
"""

# Training scripts are standalone - import when needed
# from .finetune_paddleocr import main as finetune_paddle
# from .finetune_deepseek import main as finetune_deepseek
# from .train_from_scratch import main as train_scratch

__all__ = [
    "finetune_paddleocr",
    "finetune_deepseek",
    "train_from_scratch",
]
