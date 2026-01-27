#!/usr/bin/env python3
"""
DeepSeek-OCR Fine-Tuning Pipeline for VIN Recognition
======================================================

Fine-tunes the DeepSeek-OCR vision-language model for VIN recognition.
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

Features:
- LoRA fine-tuning for memory-efficient training
- Support for full fine-tuning on larger GPUs
- Mixed precision training (bf16/fp16)
- Gradient checkpointing for memory efficiency
- DeepSpeed integration for distributed training
- Hugging Face Trainer with custom metrics

Requirements:
- PyTorch >= 2.0
- transformers >= 4.36
- peft >= 0.6 (for LoRA)
- bitsandbytes (for 8-bit quantization, optional)
- deepspeed (for distributed training, optional)

Usage:
    # LoRA fine-tuning (recommended)
    python finetune_deepseek.py --config configs/deepseek_finetune_config.yml --lora
    
    # Full fine-tuning (requires >40GB VRAM)
    python finetune_deepseek.py --config configs/deepseek_finetune_config.yml --full
    
    # Resume training
    python finetune_deepseek.py --config configs/deepseek_finetune_config.yml --resume output/deepseek_vin/checkpoint-1000

Author: JRL-VIN Project
Date: January 2026
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# Check for required libraries
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
PEFT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    # Provide stub class when torch not available
    class TorchDataset:
        """Stub Dataset class when torch is not installed."""
        pass
    DataLoader = None

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoProcessor,
        Trainer, TrainingArguments,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        TrainerCallback
    )
    from transformers.modeling_utils import PreTrainedModel
    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    # RuntimeError can occur due to numpy version incompatibility
    pass  # Silent - will print warning only when needed

try:
    from peft import (
        LoraConfig, get_peft_model, prepare_model_for_kbit_training,
        TaskType, PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    pass  # Silent - will print warning only when needed

# Optional imports
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DeepSeekFineTuneConfig:
    """Configuration for DeepSeek fine-tuning."""
    # Model
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    
    # Training
    output_dir: str = "./output/deepseek_vin_finetune"
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization
    use_8bit: bool = False
    use_4bit: bool = False
    
    # Data
    train_data_path: str = "./finetune_data/train_labels.txt"
    val_data_path: str = "./finetune_data/val_labels.txt"
    data_dir: str = "./finetune_data/"
    max_length: int = 32
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Seed
    seed: int = 42


def load_config(config_path: str) -> DeepSeekFineTuneConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return DeepSeekFineTuneConfig(**config_dict)


# =============================================================================
# DATASET
# =============================================================================

class VINDeepSeekDataset(TorchDataset):
    """Dataset for VIN recognition with DeepSeek-OCR."""
    
    def __init__(
        self,
        data_dir: str,
        label_file: str,
        processor: Any,
        max_length: int = 32,
        is_training: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.is_training = is_training
        
        # Load labels
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    img_path, label = line.split('\t', 1)
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        label = parts[1]
                    else:
                        continue
                
                full_path = self.data_dir / img_path
                if full_path.exists():
                    self.samples.append((str(full_path), label))
        
        logger.info(f"Loaded {len(self.samples)} samples from {label_file}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Process image
        inputs = self.processor(
            images=image,
            text=f"<image>\nOCR this VIN number:",
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].squeeze(0)
        
        # Add labels
        label_ids = self.processor.tokenizer(
            label,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )['input_ids'].squeeze(0)
        
        inputs['labels'] = label_ids
        inputs['text_labels'] = label
        
        return inputs


# =============================================================================
# TRAINER
# =============================================================================

class DeepSeekVINTrainer:
    """Fine-tuning trainer for DeepSeek-OCR on VIN recognition."""
    
    def __init__(self, config: DeepSeekFineTuneConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        self._set_seed(config.seed)
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Training state
        self.best_accuracy = 0.0
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def setup_model(self):
        """Initialize model with optional quantization and LoRA."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Quantization config
        quantization_config = None
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.use_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Load processor/tokenizer
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        
        # Apply LoRA if requested
        if self.config.use_lora and PEFT_AVAILABLE:
            logger.info("Applying LoRA adaptation")
            
            if self.config.use_8bit or self.config.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded successfully")
        return self.model
    
    def setup_datasets(self) -> Tuple[Any, Any]:
        """Setup training and validation datasets."""
        train_dataset = VINDeepSeekDataset(
            data_dir=self.config.data_dir,
            label_file=self.config.train_data_path,
            processor=self.processor,
            max_length=self.config.max_length,
            is_training=True
        )
        
        val_dataset = VINDeepSeekDataset(
            data_dir=self.config.data_dir,
            label_file=self.config.val_data_path,
            processor=self.processor,
            max_length=self.config.max_length,
            is_training=False
        )
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        pred_ids = np.argmax(predictions, axis=-1)
        
        # Decode to text
        pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate exact match accuracy
        correct = sum(1 for p, l in zip(pred_texts, label_texts) if p.strip() == l.strip())
        accuracy = correct / len(pred_texts) if pred_texts else 0.0
        
        # Calculate character error rate
        total_chars = sum(len(l) for l in label_texts)
        total_errors = sum(
            self._levenshtein(p.strip(), l.strip())
            for p, l in zip(pred_texts, label_texts)
        )
        cer = total_errors / total_chars if total_chars > 0 else 1.0
        
        return {
            'accuracy': accuracy,
            'cer': cer
        }
    
    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance."""
        if len(s1) < len(s2):
            return DeepSeekVINTrainer._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def train(self, resume_from: Optional[str] = None):
        """Run training with Hugging Face Trainer."""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("PyTorch and transformers are required for training")
        
        # Setup model
        self.setup_model()
        
        # Setup datasets
        train_dataset, val_dataset = self.setup_datasets()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            seed=self.config.seed,
        )
        
        # Data collator
        def collate_fn(batch):
            """Custom collate function for batch processing."""
            keys = batch[0].keys()
            collated = {}
            
            for key in keys:
                if key == 'text_labels':
                    collated[key] = [item[key] for item in batch]
                elif isinstance(batch[0][key], torch.Tensor):
                    collated[key] = torch.stack([item[key] for item in batch])
                else:
                    collated[key] = [item[key] for item in batch]
            
            return collated
        
        # Callbacks
        callbacks = []
        
        # Add custom progress callback for UI
        class ProgressCallback(TrainerCallback):
            """Callback to write training progress to JSON for UI monitoring."""
            def __init__(self, output_dir: Path, num_epochs: int):
                self.output_dir = output_dir
                self.num_epochs = num_epochs
                self.progress_file = output_dir / 'training_progress.json'
                self.best_accuracy = 0.0
                self.train_losses = []
                self.val_accuracies = []
                self.start_time = None
            
            def _save_progress(self, status, epoch, train_loss, val_loss, val_accuracy):
                import time
                elapsed = time.time() - self.start_time if self.start_time else 0
                progress = {
                    'status': status,
                    'current_epoch': epoch,
                    'total_epochs': self.num_epochs,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'best_accuracy': self.best_accuracy,
                    'elapsed_time': elapsed,
                    'timestamp': datetime.now().isoformat(),
                    'train_losses': self.train_losses[-20:],
                    'val_accuracies': self.val_accuracies[-20:]
                }
                try:
                    with open(self.progress_file, 'w') as f:
                        json.dump(progress, f, indent=2)
                except Exception:
                    pass
            
            def on_train_begin(self, args, state, control, **kwargs):
                import time
                self.start_time = time.time()
                self._save_progress('starting', 0, 0.0, 0.0, 0.0)
            
            def on_epoch_begin(self, args, state, control, **kwargs):
                self._save_progress('training', int(state.epoch) + 1, 0.0, 0.0, 0.0)
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    train_loss = logs.get('loss', 0.0)
                    eval_loss = logs.get('eval_loss', 0.0)
                    eval_accuracy = logs.get('eval_accuracy', 0.0)
                    if train_loss:
                        self.train_losses.append(train_loss)
                    if eval_accuracy:
                        self.val_accuracies.append(eval_accuracy)
                        if eval_accuracy > self.best_accuracy:
                            self.best_accuracy = eval_accuracy
                    self._save_progress('training', int(state.epoch) + 1, train_loss, eval_loss, eval_accuracy)
            
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics:
                    val_accuracy = metrics.get('eval_accuracy', 0.0)
                    val_loss = metrics.get('eval_loss', 0.0)
                    if val_accuracy > self.best_accuracy:
                        self.best_accuracy = val_accuracy
                    self.val_accuracies.append(val_accuracy)
                    self._save_progress('validating', int(state.epoch) + 1, 0.0, val_loss, val_accuracy)
            
            def on_train_end(self, args, state, control, **kwargs):
                self._save_progress('completed', self.num_epochs, 0.0, 0.0, self.best_accuracy)
        
        # Add progress callback
        callbacks.append(ProgressCallback(self.output_dir, self.config.num_epochs))
        
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        # Resume from checkpoint if specified
        resume_path = resume_from if resume_from else None
        
        # Train
        logger.info("=" * 60)
        logger.info("Starting DeepSeek-OCR Fine-Tuning for VIN Recognition")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  LoRA: {self.config.use_lora}")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 60)
        
        trainer.train(resume_from_checkpoint=resume_path)
        
        # Save final model
        trainer.save_model(str(self.output_dir / "final_model"))
        
        # Save LoRA weights separately if using LoRA
        if self.config.use_lora and PEFT_AVAILABLE:
            self.model.save_pretrained(str(self.output_dir / "lora_weights"))
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"  Model saved to: {self.output_dir}")
        logger.info("=" * 60)
    
    def export_for_inference(self, merge_lora: bool = True):
        """Export trained model for inference."""
        logger.info("Exporting model for inference...")
        
        inference_dir = self.output_dir / "inference"
        inference_dir.mkdir(exist_ok=True)
        
        if self.config.use_lora and merge_lora and PEFT_AVAILABLE:
            # Merge LoRA weights into base model
            logger.info("Merging LoRA weights...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(str(inference_dir))
        else:
            self.model.save_pretrained(str(inference_dir))
        
        # Save processor
        self.processor.save_pretrained(str(inference_dir))
        
        logger.info(f"Inference model saved to: {inference_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune DeepSeek-OCR for VIN Recognition'
    )
    parser.add_argument(
        '--config', '-c',
        default='configs/deepseek_finetune_config.yml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume', '-r',
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--lora',
        action='store_true',
        help='Use LoRA fine-tuning (recommended)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full fine-tuning (requires >40GB VRAM)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for DeepSeek fine-tuning.")
        print("Install with: pip install torch")
        sys.exit(1)
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers is required for DeepSeek fine-tuning.")
        print("Install with: pip install transformers")
        sys.exit(1)
    
    # Load or create config
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        logger.warning(f"Config file not found: {args.config}")
        logger.info("Using default configuration")
        config = DeepSeekFineTuneConfig()
    
    # Apply CLI overrides
    if args.lora:
        config.use_lora = True
    if args.full:
        config.use_lora = False
    if args.output:
        config.output_dir = args.output
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Create trainer and start training
    trainer = DeepSeekVINTrainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
