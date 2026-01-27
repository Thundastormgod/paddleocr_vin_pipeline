#!/usr/bin/env python3
"""
Unified VIN OCR Training Script
================================

Unified interface for training both PaddleOCR and DeepSeek-OCR models
for VIN recognition.

Usage:
    # Train PaddleOCR model
    python train_vin_model.py --model paddleocr --config configs/vin_finetune_config.yml
    
    # Train DeepSeek-OCR model with LoRA
    python train_vin_model.py --model deepseek --config configs/deepseek_finetune_config.yml --lora
    
    # Train both models sequentially
    python train_vin_model.py --model all
    
    # Resume training
    python train_vin_model.py --model paddleocr --resume output/vin_rec_finetune/latest

Author: JRL-VIN Project
Date: January 2026
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_paddleocr_availability() -> bool:
    """Check if PaddleOCR training is available."""
    try:
        from finetune_paddleocr import PPOCR_TRAIN_AVAILABLE, PADDLE_AVAILABLE
        return PADDLE_AVAILABLE and PPOCR_TRAIN_AVAILABLE
    except ImportError:
        return False


def check_deepseek_availability() -> bool:
    """Check if DeepSeek training is available."""
    try:
        from finetune_deepseek import TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE
        return TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE
    except ImportError:
        return False


def train_paddleocr(
    config_path: str,
    resume_from: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None
):
    """Train PaddleOCR model."""
    logger.info("=" * 60)
    logger.info("Training PaddleOCR Model for VIN Recognition")
    logger.info("=" * 60)
    
    from finetune_paddleocr import VINFineTuner, load_config
    
    # Load config
    config = load_config(config_path)
    
    # Apply overrides
    if output_dir:
        config['Global']['save_model_dir'] = output_dir
    if epochs:
        config['Global']['epoch_num'] = epochs
    if batch_size:
        config['Train']['loader']['batch_size_per_card'] = batch_size
    if learning_rate:
        config['Optimizer']['lr']['learning_rate'] = learning_rate
    
    # Create trainer and train
    trainer = VINFineTuner(
        config=config,
        output_dir=config['Global']['save_model_dir']
    )
    
    trainer.train(resume_from=resume_from)
    
    logger.info("PaddleOCR training complete!")
    return trainer.best_accuracy


def train_deepseek(
    config_path: str,
    resume_from: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    use_lora: bool = True
):
    """Train DeepSeek-OCR model."""
    logger.info("=" * 60)
    logger.info("Training DeepSeek-OCR Model for VIN Recognition")
    logger.info("=" * 60)
    
    from finetune_deepseek import DeepSeekVINTrainer, load_config, DeepSeekFineTuneConfig
    
    # Load config
    if Path(config_path).exists():
        config = load_config(config_path)
    else:
        config = DeepSeekFineTuneConfig()
    
    # Apply overrides
    config.use_lora = use_lora
    if output_dir:
        config.output_dir = output_dir
    if epochs:
        config.num_epochs = epochs
    if batch_size:
        config.batch_size = batch_size
    if learning_rate:
        config.learning_rate = learning_rate
    
    # Create trainer and train
    trainer = DeepSeekVINTrainer(config)
    trainer.train(resume_from=resume_from)
    
    logger.info("DeepSeek-OCR training complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Unified VIN OCR Training Script'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['paddleocr', 'deepseek', 'all'],
        default='paddleocr',
        help='Model to train (default: paddleocr)'
    )
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='Path to config file'
    )
    parser.add_argument(
        '--resume', '-r',
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--lora',
        action='store_true',
        help='Use LoRA for DeepSeek (default: true)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full fine-tuning for DeepSeek (no LoRA)'
    )
    
    args = parser.parse_args()
    
    # Determine default config based on model
    if args.config is None:
        if args.model == 'paddleocr':
            args.config = 'configs/vin_finetune_config.yml'
        elif args.model == 'deepseek':
            args.config = 'configs/deepseek_finetune_config.yml'
    
    # Check availability
    paddleocr_available = check_paddleocr_availability()
    deepseek_available = check_deepseek_availability()
    
    logger.info("Training Environment:")
    logger.info(f"  PaddleOCR training available: {paddleocr_available}")
    logger.info(f"  DeepSeek training available: {deepseek_available}")
    
    # Train selected model(s)
    if args.model == 'paddleocr':
        if not paddleocr_available:
            logger.error("PaddleOCR training not available!")
            logger.error("Ensure PaddleOCR is cloned: git clone https://github.com/PaddlePaddle/PaddleOCR.git")
            sys.exit(1)
        
        train_paddleocr(
            config_path=args.config,
            resume_from=args.resume,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    
    elif args.model == 'deepseek':
        if not deepseek_available:
            logger.error("DeepSeek training not available!")
            logger.error("Install requirements: pip install torch transformers peft")
            sys.exit(1)
        
        use_lora = not args.full  # Default to LoRA unless --full specified
        if args.lora:
            use_lora = True
        
        train_deepseek(
            config_path=args.config,
            resume_from=args.resume,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_lora=use_lora
        )
    
    elif args.model == 'all':
        results = {}
        
        # Train PaddleOCR
        if paddleocr_available:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 1: Training PaddleOCR")
            logger.info("=" * 60 + "\n")
            
            paddle_config = 'configs/vin_finetune_config.yml'
            paddle_output = args.output + '/paddleocr' if args.output else None
            
            results['paddleocr'] = train_paddleocr(
                config_path=paddle_config,
                output_dir=paddle_output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
        else:
            logger.warning("Skipping PaddleOCR (not available)")
        
        # Train DeepSeek
        if deepseek_available:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 2: Training DeepSeek-OCR")
            logger.info("=" * 60 + "\n")
            
            deepseek_config = 'configs/deepseek_finetune_config.yml'
            deepseek_output = args.output + '/deepseek' if args.output else None
            
            train_deepseek(
                config_path=deepseek_config,
                output_dir=deepseek_output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                use_lora=True
            )
            results['deepseek'] = True
        else:
            logger.warning("Skipping DeepSeek (not available)")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete - Summary")
        logger.info("=" * 60)
        for model, result in results.items():
            logger.info(f"  {model}: {result}")
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
