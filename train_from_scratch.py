#!/usr/bin/env python3
"""
Train OCR Models from Scratch for VIN Recognition
==================================================

This script trains OCR models from random initialization (not fine-tuning).
Use this when you have a large dataset (50,000+ images) and want to build
a custom model specifically for VIN recognition.

Supported Models:
- PaddleOCR (CRNN, SVTR architectures)
- DeepSeek-OCR (Vision-Language Model)

Requirements:
- Large labeled VIN dataset (50,000+ images recommended)
- Significant GPU memory (24GB+ recommended)
- Extended training time (days to weeks)

Usage:
    # Train PaddleOCR from scratch
    python train_from_scratch.py --model paddleocr --config configs/train_scratch_config.yml
    
    # Train with ONNX export
    python train_from_scratch.py --model paddleocr --export-onnx
    
    # Train DeepSeek from scratch (requires massive resources)
    python train_from_scratch.py --model deepseek --config configs/deepseek_scratch_config.yml

Author: JRL-VIN Project
Date: January 2026
"""

import os
import sys
import yaml
import json
import time
import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class PaddleOCRScratchConfig:
    """Configuration for training PaddleOCR from scratch."""
    
    # Model architecture
    architecture: str = "SVTR_LCNet"  # Options: CRNN, SVTR_LCNet, SVTR_Tiny
    backbone: str = "PPLCNetV3"  # Options: MobileNetV3, PPLCNetV3, ResNet
    
    # Input configuration
    image_height: int = 48
    image_width: int = 320
    max_text_length: int = 17  # VIN is always 17 characters
    
    # Character set (VIN characters only)
    character_dict_path: str = "./configs/vin_dict.txt"
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    lr_scheduler: str = "cosine"  # Options: cosine, step, exponential
    warmup_epochs: int = 5
    weight_decay: float = 0.0001
    
    # Optimizer
    optimizer: str = "Adam"  # Options: Adam, SGD, AdamW
    
    # Loss function
    loss_type: str = "CTCLoss"  # Options: CTCLoss, AttentionLoss, MultiLoss
    
    # Data augmentation
    use_augmentation: bool = True
    aug_prob: float = 0.5
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Hardware
    use_gpu: bool = True
    use_amp: bool = True  # Automatic Mixed Precision
    num_workers: int = 4
    
    # Checkpointing
    save_epoch_step: int = 10
    eval_batch_step: int = 500
    
    # Paths
    train_data_dir: str = "./data/train"
    train_label_file: str = "./data/train/labels.txt"
    val_data_dir: str = "./data/val"
    val_label_file: str = "./data/val/labels.txt"
    output_dir: str = "./output/vin_scratch_train"
    
    # Export
    export_onnx: bool = False
    onnx_opset: int = 14


@dataclass
class DeepSeekScratchConfig:
    """Configuration for training DeepSeek-OCR from scratch."""
    
    # Model architecture (much smaller than original for feasibility)
    model_type: str = "vision_encoder_decoder"
    vision_encoder: str = "vit_base_patch16"  # Smaller ViT
    text_decoder: str = "gpt2_small"  # Smaller decoder
    
    # Input configuration
    image_size: int = 384
    max_text_length: int = 32
    
    # Training parameters
    num_epochs: int = 50
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 0.0001
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # Hardware
    use_gpu: bool = True
    
    # Paths
    train_data_path: str = "./data/train/labels.txt"
    val_data_path: str = "./data/val/labels.txt"
    data_dir: str = "./data"
    output_dir: str = "./output/deepseek_scratch_train"
    
    # Export
    export_onnx: bool = False
    
    # Seed
    seed: int = 42


# =============================================================================
# VIN CHARACTER SET
# =============================================================================

VIN_CHARACTERS = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"  # 33 chars (no I, O, Q)


def create_vin_dict(output_path: str = "./configs/vin_dict.txt"):
    """Create VIN character dictionary file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for char in VIN_CHARACTERS:
            f.write(f"{char}\n")
    logger.info(f"Created VIN dictionary with {len(VIN_CHARACTERS)} characters at {output_path}")
    return output_path


# =============================================================================
# PADDLEOCR TRAINING FROM SCRATCH
# =============================================================================

class PaddleOCRScratchTrainer:
    """
    Train PaddleOCR recognition model from scratch.
    
    This creates a new model with random weights and trains it
    specifically for VIN recognition.
    """
    
    def __init__(self, config: PaddleOCRScratchConfig):
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
        
    def setup(self):
        """Setup training environment."""
        logger.info("=" * 60)
        logger.info("PADDLEOCR TRAINING FROM SCRATCH")
        logger.info("=" * 60)
        
        # Check PaddlePaddle
        try:
            import paddle
            import paddle.nn as nn
            from paddle.io import DataLoader, Dataset
            logger.info(f"PaddlePaddle version: {paddle.__version__}")
        except ImportError:
            raise RuntimeError("PaddlePaddle not installed. Run: pip install paddlepaddle-gpu")
        
        # Set device
        if self.config.use_gpu and paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
            logger.info("Using GPU for training")
        else:
            paddle.set_device('cpu')
            logger.info("Using CPU for training")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create VIN dictionary if needed
        if not os.path.exists(self.config.character_dict_path):
            create_vin_dict(self.config.character_dict_path)
        
        # Load character dictionary
        self.char_dict = self._load_char_dict()
        logger.info(f"Character set size: {len(self.char_dict)}")
        
        # Build model from scratch
        self._build_model()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup AMP
        if self.config.use_amp:
            from paddle.amp import GradScaler
            self.scaler = GradScaler()
            logger.info("Using Automatic Mixed Precision (AMP)")
        
        logger.info("Setup complete!")
        
    def _load_char_dict(self) -> Dict[str, int]:
        """Load character dictionary."""
        char_dict = {'<blank>': 0}  # CTC blank token
        with open(self.config.character_dict_path, 'r') as f:
            for idx, line in enumerate(f, start=1):
                char = line.strip()
                if char:
                    char_dict[char] = idx
        return char_dict
    
    def _build_model(self):
        """Build OCR model from scratch with random weights."""
        import paddle
        import paddle.nn as nn
        
        logger.info(f"Building {self.config.architecture} model from scratch...")
        
        num_classes = len(self.char_dict)
        
        if self.config.architecture == "CRNN":
            self.model = self._build_crnn(num_classes)
        elif self.config.architecture == "SVTR_LCNet":
            self.model = self._build_svtr_lcnet(num_classes)
        elif self.config.architecture == "SVTR_Tiny":
            self.model = self._build_svtr_tiny(num_classes)
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if not p.stop_gradient)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def _build_crnn(self, num_classes: int):
        """Build CRNN architecture from scratch."""
        import paddle
        import paddle.nn as nn
        
        class CRNN(nn.Layer):
            """Classic CRNN for OCR."""
            
            def __init__(self, num_classes, img_height=48, hidden_size=256):
                super().__init__()
                
                # CNN backbone
                self.cnn = nn.Sequential(
                    # Layer 1
                    nn.Conv2D(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2D(2, 2),
                    
                    # Layer 2
                    nn.Conv2D(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2D(2, 2),
                    
                    # Layer 3
                    nn.Conv2D(128, 256, 3, padding=1),
                    nn.BatchNorm2D(256),
                    nn.ReLU(),
                    
                    # Layer 4
                    nn.Conv2D(256, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2D((2, 1), (2, 1)),
                    
                    # Layer 5
                    nn.Conv2D(256, 512, 3, padding=1),
                    nn.BatchNorm2D(512),
                    nn.ReLU(),
                    
                    # Layer 6
                    nn.Conv2D(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2D((2, 1), (2, 1)),
                    
                    # Layer 7
                    nn.Conv2D(512, 512, 2),
                    nn.BatchNorm2D(512),
                    nn.ReLU(),
                )
                
                # Calculate feature map height after CNN
                self.feature_height = img_height // 16
                
                # RNN
                self.rnn = nn.LSTM(
                    input_size=512 * self.feature_height,
                    hidden_size=hidden_size,
                    num_layers=2,
                    direction='bidirectional'
                )
                
                # Output layer
                self.fc = nn.Linear(hidden_size * 2, num_classes)
                
            def forward(self, x):
                # CNN
                conv = self.cnn(x)
                
                # Reshape for RNN: (batch, channels, height, width) -> (batch, width, channels*height)
                b, c, h, w = conv.shape
                conv = conv.transpose([0, 3, 1, 2])  # (b, w, c, h)
                conv = conv.reshape([b, w, c * h])
                
                # RNN
                rnn_out, _ = self.rnn(conv)
                
                # Output
                output = self.fc(rnn_out)
                
                return output
        
        return CRNN(num_classes, self.config.image_height)
    
    def _build_svtr_lcnet(self, num_classes: int):
        """Build SVTR with LCNet backbone from scratch."""
        import paddle
        import paddle.nn as nn
        
        class SVTRLCNet(nn.Layer):
            """SVTR with PPLCNetV3 backbone for OCR."""
            
            def __init__(self, num_classes, img_height=48, img_width=320, hidden_size=120):
                super().__init__()
                
                # Simplified LCNet-like backbone
                self.backbone = nn.Sequential(
                    # Stem
                    nn.Conv2D(3, 16, 3, stride=2, padding=1),
                    nn.BatchNorm2D(16),
                    nn.Hardswish(),
                    
                    # Stage 1
                    self._make_stage(16, 32, 2),
                    
                    # Stage 2
                    self._make_stage(32, 64, 2),
                    
                    # Stage 3
                    self._make_stage(64, 128, 2),
                    
                    # Stage 4
                    self._make_stage(128, 256, 2),
                )
                
                # Global pooling on height dimension
                self.pool = nn.AdaptiveAvgPool2D((1, None))
                
                # Transformer encoder (simplified SVTR)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=256,
                        nhead=8,
                        dim_feedforward=512,
                        dropout=0.1,
                        activation='gelu'
                    ),
                    num_layers=2
                )
                
                # Output projection
                self.fc = nn.Linear(256, num_classes)
                
            def _make_stage(self, in_channels, out_channels, stride):
                return nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 3, stride=stride, padding=1),
                    nn.BatchNorm2D(out_channels),
                    nn.Hardswish(),
                    nn.Conv2D(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2D(out_channels),
                    nn.Hardswish(),
                )
                
            def forward(self, x):
                # Backbone
                features = self.backbone(x)
                
                # Pool height dimension
                features = self.pool(features)  # (b, c, 1, w)
                features = features.squeeze(2)  # (b, c, w)
                features = features.transpose([0, 2, 1])  # (b, w, c)
                
                # Transformer
                features = self.transformer(features)
                
                # Output
                output = self.fc(features)
                
                return output
        
        return SVTRLCNet(num_classes, self.config.image_height, self.config.image_width)
    
    def _build_svtr_tiny(self, num_classes: int):
        """Build SVTR-Tiny architecture from scratch."""
        import paddle
        import paddle.nn as nn
        
        class SVTRTiny(nn.Layer):
            """SVTR-Tiny: Pure transformer for OCR."""
            
            def __init__(self, num_classes, img_height=48, img_width=320, 
                         embed_dim=192, depth=6, num_heads=6):
                super().__init__()
                
                # Patch embedding
                self.patch_embed = nn.Sequential(
                    nn.Conv2D(3, embed_dim // 2, 3, stride=2, padding=1),
                    nn.BatchNorm2D(embed_dim // 2),
                    nn.GELU(),
                    nn.Conv2D(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2D(embed_dim),
                    nn.GELU(),
                )
                
                # Calculate sequence length
                h, w = img_height // 4, img_width // 4
                self.seq_len = h * w
                
                # Positional embedding
                self.pos_embed = self.create_parameter(
                    shape=[1, self.seq_len, embed_dim],
                    default_initializer=nn.initializer.TruncatedNormal(std=0.02)
                )
                
                # Transformer encoder
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=embed_dim * 4,
                        dropout=0.1,
                        activation='gelu'
                    ),
                    num_layers=depth
                )
                
                # Reshape for sequence output
                self.reshape_out = nn.AdaptiveAvgPool1D(80)  # Fixed output length
                
                # Output head
                self.fc = nn.Linear(embed_dim, num_classes)
                
            def forward(self, x):
                # Patch embedding
                x = self.patch_embed(x)  # (b, c, h, w)
                
                # Flatten spatial dimensions
                b, c, h, w = x.shape
                x = x.flatten(2).transpose([0, 2, 1])  # (b, h*w, c)
                
                # Add positional embedding
                x = x + self.pos_embed[:, :x.shape[1], :]
                
                # Transformer
                x = self.transformer(x)
                
                # Reshape for CTC
                x = x.transpose([0, 2, 1])  # (b, c, seq)
                x = self.reshape_out(x)  # (b, c, 80)
                x = x.transpose([0, 2, 1])  # (b, 80, c)
                
                # Output
                output = self.fc(x)
                
                return output
        
        return SVTRTiny(num_classes, self.config.image_height, self.config.image_width)
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        import paddle
        
        # Learning rate scheduler
        total_steps = self.config.num_epochs * 1000  # Approximate
        warmup_steps = self.config.warmup_epochs * 1000
        
        if self.config.lr_scheduler == "cosine":
            lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                learning_rate=self.config.learning_rate,
                T_max=total_steps - warmup_steps,
            )
        elif self.config.lr_scheduler == "step":
            lr_scheduler = paddle.optimizer.lr.StepDecay(
                learning_rate=self.config.learning_rate,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        else:
            lr_scheduler = self.config.learning_rate
        
        # Warmup
        if self.config.warmup_epochs > 0:
            lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=lr_scheduler,
                warmup_steps=warmup_steps,
                start_lr=self.config.learning_rate * 0.01,
                end_lr=self.config.learning_rate
            )
        
        self.scheduler = lr_scheduler
        
        # Optimizer
        if self.config.optimizer == "Adam":
            self.optimizer = paddle.optimizer.Adam(
                learning_rate=lr_scheduler,
                parameters=self.model.parameters(),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "AdamW":
            self.optimizer = paddle.optimizer.AdamW(
                learning_rate=lr_scheduler,
                parameters=self.model.parameters(),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "SGD":
            self.optimizer = paddle.optimizer.SGD(
                learning_rate=lr_scheduler,
                parameters=self.model.parameters(),
                weight_decay=self.config.weight_decay
            )
        
        logger.info(f"Optimizer: {self.config.optimizer}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"LR scheduler: {self.config.lr_scheduler}")
    
    def train(self):
        """Run training from scratch."""
        import paddle
        from paddle.amp import auto_cast
        
        logger.info("\n" + "=" * 60)
        logger.info("STARTING TRAINING FROM SCRATCH")
        logger.info("=" * 60)
        
        # Load data
        train_loader = self._create_dataloader(
            self.config.train_data_dir,
            self.config.train_label_file,
            is_training=True
        )
        
        val_loader = self._create_dataloader(
            self.config.val_data_dir,
            self.config.val_label_file,
            is_training=False
        )
        
        # Loss function
        loss_fn = paddle.nn.CTCLoss(blank=0, reduction='mean')
        
        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
                global_step += 1
                
                # Forward pass with AMP
                if self.config.use_amp:
                    with auto_cast():
                        outputs = self.model(images)
                        # Reshape for CTC: (T, N, C)
                        outputs = outputs.transpose([1, 0, 2])
                        input_lengths = paddle.full([outputs.shape[1]], outputs.shape[0], dtype='int64')
                        loss = loss_fn(outputs, labels, input_lengths, label_lengths)
                    
                    # Backward with scaler
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                    self.scaler.minimize(self.optimizer, scaled_loss)
                else:
                    outputs = self.model(images)
                    outputs = outputs.transpose([1, 0, 2])
                    input_lengths = paddle.full([outputs.shape[1]], outputs.shape[0], dtype='int64')
                    loss = loss_fn(outputs, labels, input_lengths, label_lengths)
                    
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.clear_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Log progress
                if global_step % 100 == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.optimizer.get_lr()
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                               f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
                
                # Evaluate
                if global_step % self.config.eval_batch_step == 0:
                    val_acc = self._evaluate(val_loader)
                    logger.info(f"Validation Accuracy: {val_acc:.2%}")
                    
                    if val_acc > self.best_accuracy:
                        self.best_accuracy = val_acc
                        self._save_checkpoint("best_model", epoch, val_acc)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            self.train_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_epoch_step == 0:
                self._save_checkpoint(f"epoch_{epoch+1}", epoch, self.best_accuracy)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING COMPLETE! Best accuracy: {self.best_accuracy:.2%}")
        logger.info("=" * 60)
        
        # Export ONNX if requested
        if self.config.export_onnx:
            self._export_onnx()
        
        return self.best_accuracy
    
    def _create_dataloader(self, data_dir: str, label_file: str, is_training: bool):
        """Create data loader for training/validation."""
        import paddle
        from paddle.io import DataLoader, Dataset
        
        class VINDataset(Dataset):
            def __init__(self, data_dir, label_file, char_dict, img_h, img_w, 
                         max_len, augment=False):
                self.data_dir = Path(data_dir)
                self.char_dict = char_dict
                self.img_h = img_h
                self.img_w = img_w
                self.max_len = max_len
                self.augment = augment
                
                # Load samples
                self.samples = []
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if '\t' in line:
                                img_path, label = line.split('\t', 1)
                                self.samples.append((img_path, label))
                
                logger.info(f"Loaded {len(self.samples)} samples from {label_file}")
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                
                # Load image with proper error handling
                full_path = self.data_dir / img_path
                if not full_path.exists():
                    full_path = Path(img_path)
                
                img = cv2.imread(str(full_path))
                if img is None:
                    # Fail fast: Skip corrupted samples by returning next valid sample
                    # This maintains data integrity - never train on fake data
                    logger.warning(f"Failed to load image: {full_path}, trying next sample")
                    return self.__getitem__((idx + 1) % len(self.samples))
                
                # Preprocess
                img = cv2.resize(img, (self.img_w, self.img_h))
                img = img.astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
                img = img.transpose([2, 0, 1])  # HWC -> CHW
                
                # Augmentation
                if self.augment:
                    img = self._augment(img)
                
                # Encode label
                label_encoded = []
                for char in label[:self.max_len]:
                    if char in self.char_dict:
                        label_encoded.append(self.char_dict[char])
                
                # Pad label
                label_length = len(label_encoded)
                while len(label_encoded) < self.max_len:
                    label_encoded.append(0)
                
                return (
                    paddle.to_tensor(img, dtype='float32'),
                    paddle.to_tensor(label_encoded, dtype='int64'),
                    paddle.to_tensor([label_length], dtype='int64')
                )
            
            def _augment(self, img):
                """Simple augmentation."""
                if np.random.random() < 0.5:
                    # Random brightness
                    img = img + np.random.uniform(-0.1, 0.1)
                if np.random.random() < 0.3:
                    # Add noise
                    noise = np.random.normal(0, 0.02, img.shape)
                    img = img + noise
                return np.clip(img, -1, 1).astype(np.float32)
        
        dataset = VINDataset(
            data_dir, label_file, self.char_dict,
            self.config.image_height, self.config.image_width,
            self.config.max_text_length,
            augment=is_training and self.config.use_augmentation
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=self.config.num_workers,
            drop_last=is_training
        )
    
    def _evaluate(self, dataloader) -> float:
        """Evaluate model on validation set."""
        import paddle
        
        self.model.eval()
        correct = 0
        total = 0
        
        # Reverse char dict for decoding
        idx_to_char = {v: k for k, v in self.char_dict.items()}
        
        with paddle.no_grad():
            for images, labels, label_lengths in dataloader:
                outputs = self.model(images)
                
                # CTC decode
                preds = outputs.argmax(axis=2)  # (batch, seq)
                
                for i in range(preds.shape[0]):
                    pred_seq = preds[i].numpy()
                    label_len = label_lengths[i].item()
                    label_seq = labels[i][:label_len].numpy()
                    
                    # CTC decode (remove blanks and duplicates)
                    decoded = []
                    prev = -1
                    for idx in pred_seq:
                        if idx != 0 and idx != prev:  # 0 is blank
                            decoded.append(idx)
                        prev = idx
                    
                    # Compare
                    pred_text = ''.join([idx_to_char.get(idx, '') for idx in decoded])
                    label_text = ''.join([idx_to_char.get(idx, '') for idx in label_seq])
                    
                    if pred_text == label_text:
                        correct += 1
                    total += 1
        
        self.model.train()
        return correct / total if total > 0 else 0.0
    
    def _save_checkpoint(self, name: str, epoch: int, accuracy: float):
        """Save model checkpoint."""
        import paddle
        
        save_path = Path(self.config.output_dir) / name
        os.makedirs(save_path, exist_ok=True)
        
        paddle.save(self.model.state_dict(), str(save_path / "model.pdparams"))
        paddle.save(self.optimizer.state_dict(), str(save_path / "optimizer.pdopt"))
        
        # Save config
        config_dict = {
            'epoch': epoch,
            'accuracy': accuracy,
            'architecture': self.config.architecture,
            'num_classes': len(self.char_dict)
        }
        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved checkpoint to {save_path}")
    
    def _export_onnx(self):
        """Export model to ONNX format."""
        import paddle
        
        logger.info("Exporting model to ONNX...")
        
        # Create dummy input
        dummy_input = paddle.randn([1, 3, self.config.image_height, self.config.image_width])
        
        # Export
        onnx_path = Path(self.config.output_dir) / "model.onnx"
        
        try:
            paddle.onnx.export(
                self.model,
                str(onnx_path),
                input_spec=[paddle.static.InputSpec(
                    shape=[None, 3, self.config.image_height, self.config.image_width],
                    dtype='float32'
                )],
                opset_version=self.config.onnx_opset
            )
            logger.info(f"ONNX model exported to {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")


# =============================================================================
# DEEPSEEK TRAINING FROM SCRATCH
# =============================================================================

class DeepSeekScratchTrainer:
    """
    Train a Vision-Language model from scratch for VIN recognition.
    
    Note: This creates a smaller model than the full DeepSeek-OCR,
    as training the full model from scratch requires massive resources.
    """
    
    def __init__(self, config: DeepSeekScratchConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.best_accuracy = 0.0
        
    def setup(self):
        """Setup training environment."""
        logger.info("=" * 60)
        logger.info("VISION-LANGUAGE MODEL TRAINING FROM SCRATCH")
        logger.info("=" * 60)
        
        # Check PyTorch
        try:
            import torch
            import torch.nn as nn
            logger.info(f"PyTorch version: {torch.__version__}")
            
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif self.config.use_gpu and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                logger.info("Using Apple MPS")
            else:
                self.device = torch.device('cpu')
                logger.info("Using CPU")
        except ImportError:
            raise RuntimeError("PyTorch not installed. Run: pip install torch")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Build model
        self._build_model()
        
        logger.info("Setup complete!")
    
    def _build_model(self):
        """Build Vision-Encoder-Decoder model from scratch."""
        import torch
        import torch.nn as nn
        
        class VisionEncoderDecoder(nn.Module):
            """Simple Vision-Encoder-Decoder for OCR."""
            
            def __init__(self, img_size=384, vocab_size=34, hidden_dim=512, 
                         num_encoder_layers=6, num_decoder_layers=6):
                super().__init__()
                
                self.vocab_size = vocab_size
                self.hidden_dim = hidden_dim
                
                # Vision Encoder (simplified ViT)
                patch_size = 16
                num_patches = (img_size // patch_size) ** 2
                
                self.patch_embed = nn.Sequential(
                    nn.Conv2d(3, hidden_dim, patch_size, stride=patch_size),
                    nn.Flatten(2),
                )
                
                self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
                
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=8,
                        dim_feedforward=hidden_dim * 4,
                        dropout=0.1,
                        batch_first=True
                    ),
                    num_layers=num_encoder_layers
                )
                
                # Text Decoder
                self.token_embed = nn.Embedding(vocab_size, hidden_dim)
                self.decoder = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=hidden_dim,
                        nhead=8,
                        dim_feedforward=hidden_dim * 4,
                        dropout=0.1,
                        batch_first=True
                    ),
                    num_layers=num_decoder_layers
                )
                
                self.output_proj = nn.Linear(hidden_dim, vocab_size)
                
            def forward(self, images, target_ids=None, max_len=17):
                # Encode images
                x = self.patch_embed(images)  # (B, C, N)
                x = x.transpose(1, 2)  # (B, N, C)
                x = x + self.pos_embed[:, :x.size(1), :]
                memory = self.encoder(x)
                
                if target_ids is not None:
                    # Teacher forcing
                    tgt_embed = self.token_embed(target_ids)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        target_ids.size(1)
                    ).to(images.device)
                    
                    decoded = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                    logits = self.output_proj(decoded)
                    return logits
                else:
                    # Autoregressive generation
                    batch_size = images.size(0)
                    generated = torch.zeros(batch_size, 1, dtype=torch.long, device=images.device)
                    
                    for _ in range(max_len):
                        tgt_embed = self.token_embed(generated)
                        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                            generated.size(1)
                        ).to(images.device)
                        
                        decoded = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                        logits = self.output_proj(decoded[:, -1:, :])
                        next_token = logits.argmax(dim=-1)
                        generated = torch.cat([generated, next_token], dim=1)
                    
                    return generated[:, 1:]  # Remove start token
        
        # VIN vocab: 33 chars + special tokens
        vocab_size = len(VIN_CHARACTERS) + 3  # + PAD, SOS, EOS
        
        self.model = VisionEncoderDecoder(
            img_size=self.config.image_size,
            vocab_size=vocab_size,
            hidden_dim=512,
            num_encoder_layers=6,
            num_decoder_layers=6
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        
    def train(self):
        """Run training from scratch."""
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        logger.info("\n" + "=" * 60)
        logger.info("STARTING TRAINING FROM SCRATCH")
        logger.info("=" * 60)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs
        )
        
        # Loss function
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 0 is PAD
        
        # Create dataloaders
        train_loader = self._create_dataloader(is_training=True)
        val_loader = self._create_dataloader(is_training=False)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images, labels[:, :-1])  # Shift for teacher forcing
                
                # Calculate loss
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1)  # Shift targets
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            # End of epoch
            scheduler.step()
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
            
            # Evaluate
            val_acc = self._evaluate(val_loader)
            logger.info(f"Validation Accuracy: {val_acc:.2%}")
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self._save_checkpoint("best_model", epoch)
        
        logger.info(f"\nTraining complete! Best accuracy: {self.best_accuracy:.2%}")
        
        # Export ONNX if requested
        if self.config.export_onnx:
            self._export_onnx()
        
        return self.best_accuracy
    
    def _create_dataloader(self, is_training: bool):
        """Create data loader."""
        import torch
        from torch.utils.data import Dataset, DataLoader
        
        class VINVLMDataset(Dataset):
            def __init__(self, data_path, data_dir, img_size, max_len, is_train):
                self.data_dir = Path(data_dir)
                self.img_size = img_size
                self.max_len = max_len
                
                # Create vocab
                self.char_to_idx = {c: i + 3 for i, c in enumerate(VIN_CHARACTERS)}
                self.char_to_idx['<PAD>'] = 0
                self.char_to_idx['<SOS>'] = 1
                self.char_to_idx['<EOS>'] = 2
                
                # Load samples
                self.samples = []
                if os.path.exists(data_path):
                    with open(data_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if '\t' in line:
                                img_path, label = line.split('\t', 1)
                                self.samples.append((img_path, label))
            
            def __len__(self):
                if not self.samples:
                    raise ValueError(
                        f"No training samples found. Ensure data exists at {self.data_dir} "
                        f"with a valid label file. Training cannot proceed with empty dataset."
                    )
                return len(self.samples)
            
            def __getitem__(self, idx):
                if not self.samples:
                    raise ValueError("Dataset is empty. Cannot retrieve samples from empty dataset.")
                
                img_path, label_text = self.samples[idx]
                
                # Load and preprocess image with proper error handling
                full_path = self.data_dir / img_path
                if not full_path.exists():
                    full_path = Path(img_path)
                
                img = cv2.imread(str(full_path))
                if img is None:
                    # Fail fast: Skip corrupted samples by returning next valid sample
                    logger.warning(f"Failed to load image: {full_path}, trying next sample")
                    return self.__getitem__((idx + 1) % len(self.samples))
                
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)
                
                # Encode label: <SOS> + chars + <EOS> + <PAD>
                label = [self.char_to_idx['<SOS>']]
                for c in label_text[:self.max_len]:
                    if c in self.char_to_idx:
                        label.append(self.char_to_idx[c])
                label.append(self.char_to_idx['<EOS>'])
                
                # Pad
                while len(label) < self.max_len + 2:
                    label.append(self.char_to_idx['<PAD>'])
                
                return img, torch.tensor(label[:self.max_len + 2], dtype=torch.long)
        
        data_path = self.config.train_data_path if is_training else self.config.val_data_path
        dataset = VINVLMDataset(
            data_path, self.config.data_dir,
            self.config.image_size, self.config.max_text_length,
            is_training
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=2
        )
    
    def _evaluate(self, dataloader) -> float:
        """Evaluate model."""
        import torch
        
        self.model.eval()
        correct = 0
        total = 0
        
        idx_to_char = {i + 3: c for i, c in enumerate(VIN_CHARACTERS)}
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                
                # Generate predictions
                preds = self.model(images, max_len=17)
                
                for i in range(preds.size(0)):
                    pred_text = ''.join([
                        idx_to_char.get(idx.item(), '') 
                        for idx in preds[i] 
                        if idx.item() in idx_to_char
                    ])
                    
                    label_text = ''.join([
                        idx_to_char.get(idx.item(), '')
                        for idx in labels[i]
                        if idx.item() in idx_to_char
                    ])
                    
                    if pred_text == label_text:
                        correct += 1
                    total += 1
        
        self.model.train()
        return correct / total if total > 0 else 0.0
    
    def _save_checkpoint(self, name: str, epoch: int):
        """Save model checkpoint."""
        import torch
        
        save_path = Path(self.config.output_dir) / name
        os.makedirs(save_path, exist_ok=True)
        
        torch.save(self.model.state_dict(), save_path / "model.pt")
        torch.save(self.optimizer.state_dict(), save_path / "optimizer.pt")
        
        logger.info(f"Saved checkpoint to {save_path}")
    
    def _export_onnx(self):
        """Export model to ONNX."""
        import torch
        
        logger.info("Exporting to ONNX...")
        
        onnx_path = Path(self.config.output_dir) / "model.onnx"
        dummy_input = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(self.device)
        
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                input_names=['image'],
                output_names=['logits'],
                dynamic_axes={
                    'image': {0: 'batch'},
                    'logits': {0: 'batch'}
                },
                opset_version=14
            )
            logger.info(f"ONNX exported to {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train OCR models from scratch")
    parser.add_argument("--model", choices=["paddleocr", "deepseek"], required=True,
                       help="Model type to train")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX after training")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    if args.model == "paddleocr":
        config = PaddleOCRScratchConfig()
        
        # Override with args
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.lr:
            config.learning_rate = args.lr
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.export_onnx:
            config.export_onnx = True
        
        trainer = PaddleOCRScratchTrainer(config)
        trainer.setup()
        trainer.train()
        
    elif args.model == "deepseek":
        config = DeepSeekScratchConfig()
        
        # Override with args
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.lr:
            config.learning_rate = args.lr
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.export_onnx:
            config.export_onnx = True
        
        trainer = DeepSeekScratchTrainer(config)
        trainer.setup()
        trainer.train()


if __name__ == "__main__":
    main()
