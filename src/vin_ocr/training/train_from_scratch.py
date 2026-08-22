#!/usr/bin/env python3
"""
Train OCR Models from Scratch for VIN Recognition
==================================================

This script trains OCR models from random initialization (not fine-tuning).
Use this when you have a large dataset (50,000+ images) and want to build
a custom model specifically for VIN recognition.

Supported Models:
- PaddleOCR PP-OCRv5 (State-of-the-art, 2024 - RECOMMENDED)
- PaddleOCR SVTR_LCNet (Good accuracy/speed balance)
- PaddleOCR SVTR_Tiny (Pure transformer, faster)
- PaddleOCR CRNN (Classic architecture, fastest training)
- DeepSeek-OCR (Vision-Language Model)

Model Sources:
- Training uses PaddlePaddle framework (local computation)
- All models trained from scratch with random initialization
- No pretrained weights required for from-scratch training
- Models saved locally as .pdparams, exported to inference format

Requirements:
- Large labeled VIN dataset (50,000+ images recommended)
- Significant GPU memory (24GB+ for RTX 3090)
- Extended training time (days to weeks)

Usage:
    # Train PP-OCRv5 (recommended)
    python -m src.vin_ocr.training.train_from_scratch --model paddleocr --arch PP-OCRv5
    
    # Train SVTR_LCNet
    python -m src.vin_ocr.training.train_from_scratch --model paddleocr --arch SVTR_LCNet
    
    # Train with ONNX export
    python -m src.vin_ocr.training.train_from_scratch --model paddleocr --export-onnx
    
    # Train DeepSeek from scratch (requires massive resources)
    python -m src.vin_ocr.training.train_from_scratch --model deepseek --config configs/deepseek_scratch_config.yml

Author: JRL-VIN Project
Date: January 2026
"""

import os
import sys
import yaml
import json
import time
import random
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

# Import unified VIN preprocessing module
from ..preprocessing import VINPreprocessor, PreprocessConfig, PreprocessStrategy
from .metrics import require_finite_loss

# Import hardware detection
from ..utils.hardware_utils import HardwareDetector

# Setup logging with immediate flush for GPU training visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# Force unbuffered output for real-time training visibility
for handler in logger.handlers:
    handler.flush = sys.stdout.flush

# NOTE: a module-level `PROJECT_ROOT = Path(__file__).parent` was removed here.
# It had zero references anywhere in the codebase, and it was also wrong: this
# file sits at <root>/src/vin_ocr/training/, so `.parent` is that training
# directory, not the repository root (four levels up). Had anything ever read
# it, every derived path would have been wrong by three levels — the same
# defect that made multi_model_evaluation.py resolve zero dataset images.
# Anything needing the root should use Path(__file__).resolve().parents[3].


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class PaddleOCRScratchConfig:
    """Configuration for training PaddleOCR from scratch."""
    
    # Model architecture
    # Options: PP-OCRv5, CRNN, SVTR_LCNet, SVTR_Tiny
    architecture: str = "PP-OCRv5"  # Default to latest v5
    backbone: str = "PPLCNetV3"  # Only PPLCNetV3-style is implemented here; no other backbone option exists in this legacy trainer
    
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
    num_workers: int = 0  # Use 0 for macOS compatibility
    
    # Reproducibility: seeds python's random, numpy and paddle (M11)
    seed: int = 42
    
    # Checkpointing
    save_epoch_step: int = 10
    eval_batch_step: int = 500
    
    # Paths - default to finetune_data for consistency
    train_data_dir: str = "./finetune_data"
    train_label_file: str = "./finetune_data/train_labels.txt"
    val_data_dir: str = "./finetune_data"
    val_label_file: str = "./finetune_data/val_labels.txt"
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

# Special token ids shared by the DeepSeek-style trainer's dataset encoding
# and the model's autoregressive generation (M8): teacher forcing feeds
# target sequences that START with <SOS>, so generation must be seeded with
# the SAME token. Seeding with 0 fed <PAD> as the first decoder input - a
# token the decoder never saw in that position during training.
PAD_TOKEN_ID = 0
SOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


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
        # Whether a best_model checkpoint has been written yet. The first
        # validation always writes one: early CTC training sits at 0.0
        # exact-match for many epochs, and a strict val_acc > 0.0 gate
        # would leave the whole run without any loadable checkpoint (M10).
        self._best_checkpoint_written = False
        
    def setup(self):
        """Setup training environment."""
        logger.info("=" * 60)
        logger.info("PADDLEOCR TRAINING FROM SCRATCH")
        logger.info("=" * 60)
        
        # Hardware detection
        hardware_detector = HardwareDetector()
        hw_config = hardware_detector.get_training_config("paddleocr")
        
        logger.info("Hardware Detection:")
        logger.info(f"  Device: {hw_config.get('device_name', 'Unknown')}")
        logger.info(f"  GPU Memory: {hw_config.get('total_memory_gb', 'N/A')} GB")
        logger.info(f"  Use GPU: {hw_config.get('use_gpu', False)}")
        logger.info(f"  Use AMP: {hw_config.get('use_amp', False)}")
        logger.info(f"  Recommended batch size: {hw_config.get('recommended_batch_size', 8)}")
        logger.info(f"  Num workers: {hw_config.get('num_workers', 0)}")
        
        # Log data paths for debugging
        logger.info(f"Train data dir: {self.config.train_data_dir}")
        logger.info(f"Train label file: {self.config.train_label_file}")
        logger.info(f"Val data dir: {self.config.val_data_dir}")
        logger.info(f"Val label file: {self.config.val_label_file}")
        logger.info(f"Output dir: {self.config.output_dir}")
        
        # Check PaddlePaddle
        try:
            import paddle
            import paddle.nn as nn
            from paddle.io import DataLoader, Dataset
            logger.info(f"PaddlePaddle version: {paddle.__version__}")
        except ImportError:
            raise RuntimeError("PaddlePaddle not installed. Run: pip install paddlepaddle-gpu")
        
        # Set device based on hardware detection
        use_gpu = self.config.use_gpu and hw_config.get('use_gpu', False)
        if use_gpu:
            paddle.set_device('gpu')
            logger.info("Using GPU for training")
        else:
            paddle.set_device('cpu')
            logger.info("Using CPU for training (PaddlePaddle does not support MPS)")
        
        # Seed every RNG the training path draws from (M11): python's
        # random, numpy (augmentation noise/probabilities) and paddle
        # (weight init, data shuffling). The `seed` config field was
        # previously parsed and never used.
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        paddle.seed(self.config.seed)
        logger.info(f"Random seed: {self.config.seed}")
        
        # Override config with hardware-detected optimal settings
        if self.config.batch_size == 64 and hw_config.get('recommended_batch_size'):
            logger.info(f"Note: Recommended batch size for your hardware: {hw_config.get('recommended_batch_size')}")
        
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
        
        # NOTE: the optimizer and LR schedule are built in train(), where the
        # real len(train_loader) is known. Building them here forced the
        # schedule horizon onto a guessed steps-per-epoch (C4).
        
        # Setup AMP
        if self.config.use_amp:
            from paddle.amp import GradScaler
            self.scaler = GradScaler()
            logger.info("Using Automatic Mixed Precision (AMP)")
        
        logger.info("Setup complete!")
        
    def _load_char_dict(self) -> Dict[str, int]:
        """
        Load the character dictionary.

        Delegates to src.vin_ocr.core.charset.load_char_dict so that this
        trainer, finetune_paddleocr.py and both inference backends share ONE
        char<->index mapping.

        This method previously reimplemented the mapping as::

            char_dict = {'<blank>': 0}
            for idx, line in enumerate(f, start=1):
                char_dict[line.strip()] = idx

        which is the mirror image of the bug core/charset.py was written to
        prevent. configs/vin_dict.txt ships '<blank>' as its FIRST line, so
        enumerate(..., start=1) re-mapped '<blank>' to 1 and shifted every
        character up by one. Three failures followed:

          * num_classes = len(char_dict) = 34, so the final Linear emits
            indices 0..33 - but 'Z' encoded to 34. Every label containing 'Z'
            was an out-of-alphabet CTC target.
          * paddle.nn.CTCLoss(blank=0) and the greedy decoder both treat 0 as
            blank, while the dict said class 0 was unused and class 1 was
            '<blank>'. The decoder could splice the literal 7-character string
            "<blank>" into a predicted VIN.
          * Inference (which does delegate) mapped '0'->1, 'A'->11, 'Z'->33
            against training's 2/12/34, so a perfectly-trained model decoded to
            garbage - exactly the failure documented in core/charset.py.

        It also depended on whether the dict file already existed: create_vin_dict()
        writes the 33 characters WITHOUT a blank line, which loaded correctly,
        so identical code produced two different charsets. load_char_dict()
        normalises both conventions.
        """
        from src.vin_ocr.core.charset import load_char_dict

        char_to_idx, idx_to_char = load_char_dict(self.config.character_dict_path)
        # Cache the canonical reverse map so decoding cannot re-derive a
        # different one.
        self.idx_to_char = idx_to_char
        return char_to_idx
    
    def _build_model(self):
        """Build OCR model from scratch with random weights."""
        import paddle
        import paddle.nn as nn
        
        logger.info(f"Building {self.config.architecture} model from scratch...")
        
        # The `backbone` knob predates this trainer's architecture dispatch:
        # every architecture below hard-codes its own backbone and there is
        # no alternative implementation to select. Reject non-default values
        # loudly instead of silently ignoring them (L22).
        if self.config.backbone != "PPLCNetV3":
            raise ValueError(
                "backbone config not supported by this trainer: got "
                f"{self.config.backbone!r}. Each architecture "
                "(PP-OCRv5/CRNN/SVTR_LCNet/SVTR_Tiny) hard-codes its own "
                "backbone; select the model via `architecture` instead."
            )
        
        num_classes = len(self.char_dict)
        
        if self.config.architecture == "CRNN":
            self.model = self._build_crnn(num_classes)
        elif self.config.architecture == "SVTR_LCNet":
            self.model = self._build_svtr_lcnet(num_classes)
        elif self.config.architecture == "SVTR_Tiny":
            self.model = self._build_svtr_tiny(num_classes)
        elif self.config.architecture == "PP-OCRv5":
            self.model = self._build_pp_ocrv5(num_classes)
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
            
            def __init__(self, num_classes, img_height=48, img_width=320, hidden_size=256):
                super().__init__()
                
                self.img_height = img_height
                self.img_width = img_width
                
                # CNN backbone
                self.cnn = nn.Sequential(
                    # Layer 1
                    nn.Conv2D(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2D(2, 2),  # h/2, w/2
                    
                    # Layer 2
                    nn.Conv2D(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2D(2, 2),  # h/4, w/4
                    
                    # Layer 3
                    nn.Conv2D(128, 256, 3, padding=1),
                    nn.BatchNorm2D(256),
                    nn.ReLU(),
                    
                    # Layer 4
                    nn.Conv2D(256, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2D((2, 1), (2, 1)),  # h/8, w/4
                    
                    # Layer 5
                    nn.Conv2D(256, 512, 3, padding=1),
                    nn.BatchNorm2D(512),
                    nn.ReLU(),
                    
                    # Layer 6
                    nn.Conv2D(512, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2D((2, 1), (2, 1)),  # h/16, w/4
                    
                    # Layer 7 - use padding to maintain dims
                    nn.Conv2D(512, 512, 3, padding=1),
                    nn.BatchNorm2D(512),
                    nn.ReLU(),
                )
                
                # Adaptive pooling to ensure consistent height of 1
                self.adaptive_pool = nn.AdaptiveAvgPool2D((1, None))
                
                # RNN - input is 512 (channels) since height is pooled to 1
                self.rnn = nn.LSTM(
                    input_size=512,
                    hidden_size=hidden_size,
                    num_layers=2,
                    direction='bidirectional'
                )
                
                # Output layer
                self.fc = nn.Linear(hidden_size * 2, num_classes)
                
            def forward(self, x):
                # CNN
                conv = self.cnn(x)  # (b, 512, h', w')
                
                # Adaptive pool height to 1
                conv = self.adaptive_pool(conv)  # (b, 512, 1, w')
                
                # Reshape for RNN: (batch, channels, 1, width) -> (batch, width, channels)
                b, c, h, w = conv.shape
                conv = conv.squeeze(2)  # (b, 512, w')
                conv = conv.transpose([0, 2, 1])  # (b, w', 512)
                
                # RNN
                rnn_out, _ = self.rnn(conv)
                
                # Output
                output = self.fc(rnn_out)
                
                return output
        
        return CRNN(num_classes, self.config.image_height, self.config.image_width)
    
    def _build_svtr_lcnet(self, num_classes: int):
        """Build SVTR with LCNet backbone from scratch."""
        import paddle
        import paddle.nn as nn
        
        class SVTRLCNet(nn.Layer):
            """SVTR with PPLCNetV3 backbone for OCR."""
            
            def __init__(self, num_classes, img_height=48, img_width=320, hidden_size=120):
                super().__init__()
                
                # Simplified LCNet-like backbone.
                #
                # CTC GEOMETRY: the output timestep count T is the width of
                # the final feature map, and CTC needs T >= label length
                # (17 for a VIN). The stem and stage 1 downsample both axes
                # (width 320 -> 80); stages 2-4 use stride (2, 1) so height
                # keeps compressing while WIDTH IS PRESERVED, giving T = 80.
                #
                # Every stage previously used isotropic stride 2: width
                # 320/2^5 = 10 < 17, so no CTC alignment existed, the loss
                # was inf/nan from the first batch, and the epoch averages
                # were poisoned silently. PPHGNet below already used (2, 1)
                # for exactly this reason; this backbone had been left
                # behind.
                self.backbone = nn.Sequential(
                    # Stem
                    nn.Conv2D(3, 16, 3, stride=2, padding=1),
                    nn.BatchNorm2D(16),
                    nn.Hardswish(),
                    
                    # Stage 1
                    self._make_stage(16, 32, 2),
                    
                    # Stage 2 (height only)
                    self._make_stage(32, 64, (2, 1)),
                    
                    # Stage 3 (height only)
                    self._make_stage(64, 128, (2, 1)),
                    
                    # Stage 4 (height only)
                    self._make_stage(128, 256, (2, 1)),
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
                
                # Pools the WIDTH axis (image columns) to the fixed CTC
                # timestep count. Applied only AFTER the height axis has
                # been collapsed, so every timestep is a contiguous span of
                # image columns (M9).
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
                
                # Reshape for CTC (M9). flatten(2) built an h-major sequence
                # (index = row * w + col), so pooling it straight to 80 bins
                # mixed pixels from DIFFERENT ROWS into single timesteps and
                # broke CTC's premise that timesteps advance monotonically
                # along the writing direction. Restore the 2-D grid, collapse
                # HEIGHT first so each position is one image COLUMN, then
                # pool along width only.
                x = x.reshape([-1, h, w, c])  # (b, h, w, c)
                x = x.mean(axis=1)            # (b, w, c) - per-column features
                x = x.transpose([0, 2, 1])    # (b, c, w)
                x = self.reshape_out(x)       # (b, c, 80) - columns -> timesteps
                x = x.transpose([0, 2, 1])    # (b, 80, c)
                
                # Output
                output = self.fc(x)
                
                return output
        
        return SVTRTiny(num_classes, self.config.image_height, self.config.image_width)
    
    def _build_pp_ocrv5(self, num_classes: int):
        """Build PP-OCRv5 architecture from scratch.
        
        PP-OCRv5 is the state-of-the-art PaddleOCR architecture (2024) with:
        - PPHGNetV2 backbone with SE attention
        - SVTR-like transformer encoder
        - CTCHead decoder
        - Improved training strategies
        """
        import paddle
        import paddle.nn as nn
        
        class PPOCRv5(nn.Layer):
            """PP-OCRv5: State-of-the-art PaddleOCR model architecture.
            
            Features:
            - PPHGNetV2-inspired backbone with SE attention blocks
            - Multi-scale feature fusion
            - SVTR-style transformer encoder with local and global mixing
            - CTCHead for sequence recognition
            """
            
            def __init__(self, num_classes, img_height=48, img_width=320, 
                         embed_dim=256, depth=4, num_heads=8):
                super().__init__()
                
                self.embed_dim = embed_dim
                
                # PPHGNetV2-inspired backbone with SE attention
                self.backbone = self._build_pphgnetv2_backbone(embed_dim)
                
                # Multi-scale feature fusion
                self.neck = nn.Sequential(
                    nn.Conv2D(embed_dim, embed_dim, 1),
                    nn.BatchNorm2D(embed_dim),
                    nn.Hardswish(),
                )
                
                # Global pooling on height dimension
                self.pool = nn.AdaptiveAvgPool2D((1, None))
                
                # SVTR-style transformer encoder with both local and global attention
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    activation='gelu'
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
                
                # Layer norm before output
                self.norm = nn.LayerNorm(embed_dim)
                
                # CTCHead output projection
                self.fc = nn.Linear(embed_dim, num_classes)
                
            def _build_pphgnetv2_backbone(self, out_channels):
                """Build PPHGNetV2-inspired backbone with SE attention."""
                
                class SEBlock(nn.Layer):
                    """Squeeze-and-Excitation block."""
                    def __init__(self, channels, reduction=4):
                        super().__init__()
                        self.pool = nn.AdaptiveAvgPool2D(1)
                        self.fc1 = nn.Conv2D(channels, channels // reduction, 1)
                        self.fc2 = nn.Conv2D(channels // reduction, channels, 1)
                        
                    def forward(self, x):
                        w = self.pool(x)
                        w = nn.functional.relu(self.fc1(w))
                        w = nn.functional.sigmoid(self.fc2(w))
                        return x * w
                
                class HGBlock(nn.Layer):
                    """HGNet-style block with SE attention."""
                    def __init__(self, in_channels, out_channels, stride=1):
                        super().__init__()
                        mid_channels = out_channels // 2
                        
                        self.conv1 = nn.Sequential(
                            nn.Conv2D(in_channels, mid_channels, 1),
                            nn.BatchNorm2D(mid_channels),
                            nn.Hardswish(),
                        )
                        self.conv2 = nn.Sequential(
                            nn.Conv2D(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels),
                            nn.BatchNorm2D(mid_channels),
                            nn.Hardswish(),
                        )
                        self.conv3 = nn.Sequential(
                            nn.Conv2D(mid_channels, out_channels, 1),
                            nn.BatchNorm2D(out_channels),
                        )
                        self.se = SEBlock(out_channels)
                        
                        # Shortcut
                        if stride != 1 or in_channels != out_channels:
                            self.shortcut = nn.Sequential(
                                nn.Conv2D(in_channels, out_channels, 1, stride=stride),
                                nn.BatchNorm2D(out_channels),
                            )
                        else:
                            self.shortcut = nn.Identity()
                        
                        self.act = nn.Hardswish()
                        
                    def forward(self, x):
                        identity = self.shortcut(x)
                        out = self.conv1(x)
                        out = self.conv2(out)
                        out = self.conv3(out)
                        out = self.se(out)
                        out = out + identity
                        return self.act(out)
                
                return nn.Sequential(
                    # Stem
                    nn.Conv2D(3, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2D(32),
                    nn.Hardswish(),
                    
                    # Stage 1
                    HGBlock(32, 64, stride=2),
                    HGBlock(64, 64),
                    
                    # Stage 2
                    HGBlock(64, 128, stride=2),
                    HGBlock(128, 128),
                    
                    # Stage 3
                    HGBlock(128, 192, stride=(2, 1)),
                    HGBlock(192, 192),
                    
                    # Stage 4
                    HGBlock(192, out_channels, stride=(2, 1)),
                    HGBlock(out_channels, out_channels),
                )
                
            def forward(self, x):
                # Backbone feature extraction
                features = self.backbone(x)  # (b, c, h', w')
                
                # Neck for feature refinement
                features = self.neck(features)
                
                # Pool height dimension
                features = self.pool(features)  # (b, c, 1, w)
                features = features.squeeze(2)  # (b, c, w)
                features = features.transpose([0, 2, 1])  # (b, w, c)
                
                # Transformer encoder
                features = self.transformer(features)
                
                # Layer normalization
                features = self.norm(features)
                
                # Output projection
                output = self.fc(features)
                
                return output
        
        return PPOCRv5(num_classes, self.config.image_height, self.config.image_width)
    
    def _setup_optimizer(self, steps_per_epoch: int):
        """Setup optimizer and learning rate scheduler.
        
        Args:
            steps_per_epoch: REAL number of optimizer steps per epoch,
                i.e. len(train_loader). The schedule horizon is built in
                STEP units and train() calls scheduler.step() once per
                batch, so `warmup_epochs` and `lr_scheduler` take effect
                exactly as configured. Previously the horizon came from a
                guessed max(100, 1000 // batch_size) and the scheduler was
                never stepped at all (C4), so every warmup run trained at
                start_lr (1% of the configured LR) forever.
        """
        import paddle
        
        if steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be positive, got {steps_per_epoch}"
            )
        
        # Schedule horizon in real optimizer steps.
        total_steps = max(self.config.num_epochs * steps_per_epoch, 1)
        warmup_steps = min(self.config.warmup_epochs * steps_per_epoch, total_steps // 2)
        
        # Ensure T_max is at least 1
        t_max = max(total_steps - warmup_steps, 1)
        
        if self.config.lr_scheduler == "cosine":
            lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                learning_rate=self.config.learning_rate,
                T_max=t_max,
            )
        elif self.config.lr_scheduler == "step":
            # Decay every ~third of the run. step_size is in scheduler
            # steps, which advance once per batch.
            step_size = max(self.config.num_epochs // 3, 1) * steps_per_epoch
            lr_scheduler = paddle.optimizer.lr.StepDecay(
                learning_rate=self.config.learning_rate,
                step_size=step_size,
                gamma=0.1
            )
        else:
            lr_scheduler = self.config.learning_rate
        
        # Warmup (only if we have enough steps)
        if self.config.warmup_epochs > 0 and warmup_steps > 0:
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
        logger.info(
            f"Schedule horizon: {total_steps} steps "
            f"({steps_per_epoch} steps/epoch), warmup {warmup_steps} steps"
        )
    
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
        
        # Guard the epoch averages and the schedule horizon (L22): with
        # drop_last=True a dataset smaller than batch_size yields ZERO
        # batches, which previously surfaced as ZeroDivisionError at the
        # end of the first epoch.
        steps_per_epoch = len(train_loader)
        if steps_per_epoch == 0:
            raise RuntimeError(
                f"Training loader is empty: every epoch would run 0 batches "
                f"(batch_size={self.config.batch_size} with drop_last=True "
                f"drops the final partial batch). Reduce batch_size or add "
                f"training data."
            )
        
        # Build optimizer + LR schedule from the REAL steps-per-epoch (C4).
        self._setup_optimizer(steps_per_epoch)
        # Constant-LR configs store a plain float in self.scheduler; only
        # real schedulers are stepped.
        scheduler_is_steppable = isinstance(
            self.scheduler, paddle.optimizer.lr.LRScheduler
        )
        
        # Loss function
        loss_fn = paddle.nn.CTCLoss(blank=0, reduction='mean')
        
        label_smoothing = float(self.config.label_smoothing)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        
        def _compute_loss(images, labels, label_lengths):
            """CTC loss with optional uniform label smoothing (L22).
            
            Standard label smoothing rewrites one-hot CE targets as
            (1-eps)*one_hot + eps*uniform, which decomposes into
            (1-eps)*CE + eps*CE(uniform). The CTC analogue blends the CTC
            loss with the cross-entropy of every timestep's distribution
            against the uniform distribution. At eps=0 this is exactly the
            plain CTC loss. The `label_smoothing` knob was previously
            parsed and never used.
            """
            outputs = self.model(images)
            # Reshape for CTC: (T, N, C)
            outputs = outputs.transpose([1, 0, 2])
            input_lengths = paddle.full([outputs.shape[1]], outputs.shape[0], dtype='int64')
            loss = loss_fn(outputs, labels, input_lengths, label_lengths)
            if label_smoothing > 0.0:
                uniform_ce = -paddle.nn.functional.log_softmax(outputs, axis=2).mean()
                loss = (1.0 - label_smoothing) * loss + label_smoothing * uniform_ce
            return loss
        
        # Progress file path for UI monitoring
        progress_file = Path(self.config.output_dir) / "training_progress.json"
        
        def _write_progress(epoch, batch, total_batches, loss, accuracy=0.0, message=""):
            """Write progress to JSON file for UI monitoring."""
            progress = {
                "epoch": epoch,
                "total_epochs": self.config.num_epochs,
                "batch": batch,
                "total_batches": total_batches,
                "loss": float(loss),
                "accuracy": float(accuracy),
                "best_accuracy": float(self.best_accuracy),
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
            try:
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write progress file: {e}")
        
        def _validate_and_save_best(epoch, batch, avg_loss):
            """Validate, log, persist progress and save best_model (M10).
            
            Runs the FULL best-model logic on every validation, wherever it
            was triggered from (periodic step boundary or end of epoch).
            """
            val_acc, val_metrics = self._evaluate(val_loader)
            logger.info(f"Validation Accuracy: {val_acc:.2%}")
            
            # Print comprehensive metrics
            if val_metrics:
                m = val_metrics
                logger.info(f"  📊 Image-Level: {m.get('correct_images', 0)}/{m.get('total_images', 0)} correct ({m.get('image_accuracy', 0)*100:.2f}%)")
                logger.info(f"  📝 Char-Level: Acc={m.get('char_accuracy', 0)*100:.2f}%, F1-micro={m.get('f1_micro', 0):.4f}, F1-macro={m.get('f1_macro', 0):.4f}")
                logger.info(f"  🏭 Industry: CER={m.get('cer', 1)*100:.2f}%, ValidVIN={m.get('valid_vin_rate', 0)*100:.2f}%")
            
            # Write progress with accuracy
            _write_progress(epoch+1, batch, total_batches, avg_loss, val_acc, f"Validation: {val_acc:.2%}")
            
            # First validation writes the baseline best_model; afterwards
            # only strict improvement overwrites it.
            if val_acc > self.best_accuracy or not self._best_checkpoint_written:
                self.best_accuracy = max(self.best_accuracy, val_acc)
                self._save_checkpoint("best_model", epoch, val_acc)
                self._best_checkpoint_written = True
            return val_acc
        
        # Training loop
        global_step = 0
        total_batches = steps_per_epoch
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
                global_step += 1
                
                # Forward pass with AMP
                if self.config.use_amp:
                    with auto_cast():
                        loss = _compute_loss(images, labels, label_lengths)
                    
                    # Backward with scaler
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                    self.scaler.minimize(self.optimizer, scaled_loss)
                else:
                    loss = _compute_loss(images, labels, label_lengths)
                    
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.clear_grad()
                
                # Advance the LR schedule once per optimizer step - the
                # horizon in _setup_optimizer is built in step units.
                # Without this call the schedule never moved and the whole
                # run trained at the warmup start LR (C4).
                if scheduler_is_steppable:
                    self.scheduler.step()
                
                epoch_loss += require_finite_loss(
                    loss.item(),
                    context=f"scratch training epoch {epoch + 1} "
                            f"batch {batch_idx}",
                )
                num_batches += 1
                
                # Log progress and write to progress file
                if global_step % 100 == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.optimizer.get_lr()
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                               f"Batch {batch_idx+1}/{total_batches} | Step {global_step} | "
                               f"Loss: {avg_loss:.4f} | LR: {lr:.6f}")
                    _write_progress(epoch+1, batch_idx+1, total_batches, avg_loss)
                
                # Periodic evaluation (kept); the end-of-epoch evaluation
                # below runs regardless, so short runs still validate (M10).
                if self.config.eval_batch_step > 0 and global_step % self.config.eval_batch_step == 0:
                    _validate_and_save_best(epoch, batch_idx + 1, epoch_loss / num_batches)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            self.train_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f}")
            
            # End-of-epoch validation - ALWAYS runs, so runs shorter than
            # eval_batch_step global steps still validate and still write
            # best_model (M10).
            _validate_and_save_best(epoch, total_batches, avg_epoch_loss)
            
            _write_progress(epoch+1, total_batches, total_batches, avg_epoch_loss, self.best_accuracy, f"Epoch {epoch+1} complete")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_epoch_step == 0:
                self._save_checkpoint(f"epoch_{epoch+1}", epoch, self.best_accuracy)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING COMPLETE! Best accuracy: {self.best_accuracy:.2%}")
        logger.info("=" * 60)
        _write_progress(self.config.num_epochs, total_batches, total_batches, avg_epoch_loss, self.best_accuracy, "Training complete!")
        
        # Export ONNX if requested
        if self.config.export_onnx:
            self._export_onnx()
        
        return self.best_accuracy
    
    def _create_dataloader(self, data_dir: str, label_file: str, is_training: bool):
        """Create data loader for training/validation."""
        import paddle
        from paddle.io import DataLoader, Dataset
        
        # Validate paths exist before creating dataset
        if not os.path.exists(label_file):
            raise FileNotFoundError(
                f"Label file not found: {label_file}\n"
                f"Please check the path or create the labels file.\n"
                f"Expected format: 'image_filename  VIN_LABEL' (space or tab separated)"
            )
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Please check the path exists and contains training images."
            )
        
        # Create VIN preprocessor for consistent preprocessing during training
        vin_preprocess_config = PreprocessConfig(
            strategy=PreprocessStrategy.ENGRAVED,
            target_width=self.config.image_width * 3,
            min_height=self.config.image_height,
        )
        vin_preprocessor = VINPreprocessor(config=vin_preprocess_config)
        
        class VINDataset(Dataset):
            #: Fraction of distinct unreadable images above which the run
            #: aborts: past that point the epoch is substantially built from
            #: duplicated neighbour samples and no longer measures the
            #: dataset. Mirrors finetune_paddleocr.VINRecognitionDataset.
            MAX_CORRUPT_FRACTION = 0.05
            
            def __init__(self, data_dir, label_file, char_dict, img_h, img_w, 
                         max_len, augment=False, aug_prob=0.5, preprocessor=None):
                self.data_dir = Path(data_dir)
                self.char_dict = char_dict
                self.img_h = img_h
                self.img_w = img_w
                self.max_len = max_len
                self.augment = augment
                self.aug_prob = float(aug_prob)
                if not 0.0 <= self.aug_prob <= 1.0:
                    raise ValueError(f"aug_prob must be in [0, 1], got {aug_prob}")
                self.preprocessor = preprocessor
                # Unreadable paths already warned about, so each is logged
                # once rather than once per epoch (M12).
                self._corrupt_paths = set()
                
                # Load samples
                self.samples = []
                rejected_unknown = 0
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            # Handle both tab and space-separated formats
                            if '\t' in line:
                                parts = line.split('\t', 1)
                            else:
                                # Split on multiple spaces
                                parts = line.split(None, 1)  # Split on any whitespace
                            
                            if len(parts) == 2:
                                img_path, label = parts
                                img_path, label = img_path.strip(), label.strip()
                                # Reject labels with characters outside the
                                # training charset (L22): silently dropping
                                # the characters shortened the CTC target and
                                # supervised the model on text that is NOT
                                # the image's text. Policy matches the M13
                                # fix direction in finetune_paddleocr:
                                # reject the sample with a warning.
                                unknown = sorted({c for c in label if c not in self.char_dict})
                                if unknown:
                                    rejected_unknown += 1
                                    logger.warning(
                                        f"Rejecting sample {img_path}: label {label!r} "
                                        f"contains characters outside the charset: {unknown}"
                                    )
                                    continue
                                self.samples.append((img_path, label))
                
                if rejected_unknown:
                    logger.warning(
                        f"Rejected {rejected_unknown} sample(s) with "
                        f"out-of-charset labels from {label_file}"
                    )
                
                # Validate we have samples
                if len(self.samples) == 0:
                    raise ValueError(
                        f"No valid samples found in {label_file}!\n"
                        f"Expected format: 'image_filename  VIN_LABEL' (space or tab separated)\n"
                        f"Example: 'img001.jpg  1HGBH41JXMN109186'"
                    )
                
                logger.info(f"Loaded {len(self.samples)} samples from {label_file}")
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                # Skip unreadable images by advancing to the next sample
                # with a bounded LOOP, not recursion (M12): recursing on
                # (idx + 1) % len is a RecursionError on a run of bad files
                # and silently duplicates neighbours otherwise. Every skipped
                # path is logged (once); an all-corrupt dataset raises.
                img = None
                for offset in range(len(self.samples)):
                    img_path, label = self.samples[(idx + offset) % len(self.samples)]
                    full_path = self.data_dir / img_path
                    if not full_path.exists():
                        full_path = Path(img_path)
                    img = cv2.imread(str(full_path))
                    if img is not None:
                        break
                    if img_path not in self._corrupt_paths:
                        self._corrupt_paths.add(img_path)
                        logger.warning(f"Unreadable image skipped: {full_path}")
                if img is None:
                    raise RuntimeError(
                        f"No readable image in the entire dataset "
                        f"({len(self.samples)} samples); first path: "
                        f"{self.samples[idx][0]}"
                    )
                max_corrupt = max(1, int(len(self.samples) * self.MAX_CORRUPT_FRACTION))
                if len(self._corrupt_paths) > max_corrupt:
                    raise RuntimeError(
                        f"{len(self._corrupt_paths)} of {len(self.samples)} "
                        f"images are unreadable "
                        f"(>{self.MAX_CORRUPT_FRACTION:.0%}); aborting - a "
                        f"run padded with duplicated neighbours would measure "
                        f"the loader, not the dataset. Corrupt paths logged "
                        f"above."
                    )
                
                # Apply VIN-optimized preprocessing (CLAHE, morphology, etc.)
                if self.preprocessor is not None:
                    img = self.preprocessor.process(img)
                
                # Resize to target dimensions
                img = cv2.resize(img, (self.img_w, self.img_h))
                img = img.astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
                img = img.transpose([2, 0, 1])  # HWC -> CHW
                
                # Augmentation, gated by the configured probability (L22:
                # aug_prob was previously parsed and ignored).
                if self.augment and np.random.random() < self.aug_prob:
                    img = self._augment(img)
                
                # Encode label. Labels were validated against the charset at
                # load time, so every character maps.
                label_encoded = [self.char_dict[char] for char in label[:self.max_len]]
                
                # Pad label
                label_length = len(label_encoded)
                while len(label_encoded) < self.max_len:
                    label_encoded.append(0)
                
                return (
                    paddle.to_tensor(img, dtype='float32'),
                    paddle.to_tensor(label_encoded, dtype='int32'),  # CTC requires int32
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
            augment=is_training and self.config.use_augmentation,
            aug_prob=self.config.aug_prob,
            preprocessor=vin_preprocessor
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=self.config.num_workers,
            drop_last=is_training
        )
    
    def _evaluate(self, dataloader) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate model on validation set with comprehensive metrics.
        
        Returns:
            Tuple of (accuracy, metrics_dict)
        """
        import paddle
        
        self.model.eval()
        
        # Canonical reverse map, cached by _load_char_dict(). BLANK_TOKEN is
        # excluded so a stray blank prediction can never splice the literal
        # string "<blank>" into a decoded VIN.
        from src.vin_ocr.core.charset import (
            BLANK_INDEX,
            BLANK_TOKEN,
            ctc_greedy_decode,
        )

        idx_to_char = {
            idx: ch
            for idx, ch in getattr(self, 'idx_to_char', {}).items()
            if idx != BLANK_INDEX and ch != BLANK_TOKEN
        }
        
        all_predictions = []
        all_targets = []
        
        with paddle.no_grad():
            for images, labels, label_lengths in dataloader:
                outputs = self.model(images)
                
                # CTC decode
                preds = outputs.argmax(axis=2)  # (batch, seq)
                
                for i in range(preds.shape[0]):
                    pred_seq = preds[i].numpy()
                    label_len = label_lengths[i].item()
                    label_seq = labels[i][:label_len].numpy()
                    
                    # Single canonical decode (collapse-then-strip); a local
                    # copy of this loop is how the wrong-blank decoder bug
                    # shipped elsewhere in this repo.
                    pred_text, _ = ctc_greedy_decode(pred_seq, idx_to_char)
                    label_text = ''.join([idx_to_char.get(idx, '') for idx in label_seq])
                    
                    all_predictions.append(pred_text)
                    all_targets.append(label_text)
        
        self.model.train()
        
        # Calculate comprehensive metrics
        try:
            from ..evaluation.metrics import EvaluationMetricsCalculator
            calc = EvaluationMetricsCalculator()
            calc.add_batch(all_predictions, all_targets)
            full_metrics = calc.compute()
            
            metrics = {
                # Image-level
                'correct_images': full_metrics.image_level.correct_images,
                'failed_images': full_metrics.image_level.failed_images,
                'total_images': full_metrics.image_level.total_images,
                'image_accuracy': full_metrics.image_level.accuracy,
                
                # Character-level
                'total_characters': full_metrics.character_level.total_characters,
                'correct_characters': full_metrics.character_level.correct_characters,
                'char_accuracy': full_metrics.character_level.char_accuracy,
                'f1_micro': full_metrics.character_level.f1_micro,
                'f1_macro': full_metrics.character_level.f1_macro,
                'precision': full_metrics.character_level.precision,
                'recall': full_metrics.character_level.recall,
                
                # Industry
                'cer': full_metrics.character_level.char_error_rate,
                'ned': full_metrics.character_level.normalized_edit_distance,
                
                # Full metrics object
                '_full_metrics': full_metrics,
            }
            return metrics['image_accuracy'], metrics
            
        except ImportError:
            # Fallback to basic accuracy
            correct = sum(1 for p, t in zip(all_predictions, all_targets, strict=True) if p == t)
            total = len(all_predictions)
            accuracy = correct / total if total > 0 else 0.0
            return accuracy, {'image_accuracy': accuracy, 'correct_images': correct, 'total_images': total}
    
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
        
        # Export (the input contract is carried by input_spec below; a
        # dead `dummy_input` tensor used to be created here and never used)
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
        
        # Seed every RNG this trainer draws from (M11): python's random,
        # numpy (preprocessing) and torch (weight init, DataLoader
        # shuffling; manual_seed also seeds the CUDA/MPS generators). The
        # `seed` config field was previously parsed and never used.
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        logger.info(f"Random seed: {self.config.seed}")
        
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
                    # Autoregressive generation, seeded with <SOS> exactly
                    # like the teacher-forced training sequences (M8).
                    # Seeding with zeros fed <PAD> as the first decoder
                    # input - a token the decoder never saw in that position
                    # during training.
                    batch_size = images.size(0)
                    generated = torch.full(
                        (batch_size, 1), SOS_TOKEN_ID,
                        dtype=torch.long, device=images.device,
                    )
                    
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
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
        
        # Create dataloaders
        train_loader = self._create_dataloader(is_training=True)
        val_loader = self._create_dataloader(is_training=False)
        
        accumulation_steps = int(self.config.gradient_accumulation_steps)
        if accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {accumulation_steps}"
            )
        
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
                
                # Backward pass. Dividing by the accumulation window makes
                # the accumulated gradient the gradient of the MEAN loss
                # over the window (M7); without it every accumulated step
                # was accumulation_steps times larger than configured.
                (loss / accumulation_steps).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_loss += require_finite_loss(
                    loss.item(),
                    context=f"DeepSeek scratch training epoch {epoch + 1} "
                            f"batch {batch_idx}",
                )
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            if num_batches == 0:
                raise RuntimeError(
                    "Training loader produced zero batches; cannot compute "
                    "an epoch average. Check the dataset and batch size."
                )
            
            # Flush the leftover partial accumulation window (M7): stepping
            # here keeps its gradients from silently leaking into the first
            # update of the NEXT epoch.
            if num_batches % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # End of epoch
            scheduler.step()
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
            
            # Evaluate
            val_acc, val_metrics = self._evaluate(val_loader)
            logger.info(f"Validation Accuracy: {val_acc:.2%}")
            
            # Print comprehensive metrics
            if val_metrics:
                m = val_metrics
                logger.info(f"  📊 Image-Level: {m.get('correct_images', 0)}/{m.get('total_images', 0)} correct ({m.get('image_accuracy', 0)*100:.2f}%)")
                logger.info(f"  📝 Char-Level: Acc={m.get('char_accuracy', 0)*100:.2f}%, F1-micro={m.get('f1_micro', 0):.4f}, F1-macro={m.get('f1_macro', 0):.4f}")
                logger.info(f"  🏭 Industry: CER={m.get('cer', 1)*100:.2f}%, ValidVIN={m.get('valid_vin_rate', 0)*100:.2f}%")
            
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
        
        # Create VIN preprocessor for consistent preprocessing
        vin_preprocess_config = PreprocessConfig(
            strategy=PreprocessStrategy.ENGRAVED,
            target_width=self.config.image_size * 2,
            min_height=self.config.image_size // 2,
        )
        vin_preprocessor = VINPreprocessor(config=vin_preprocess_config)
        
        class VINVLMDataset(Dataset):
            #: Fraction of distinct unreadable images above which the run
            #: aborts (M12) - mirrors the paddle dataset above and
            #: finetune_paddleocr.VINRecognitionDataset.
            MAX_CORRUPT_FRACTION = 0.05
            
            def __init__(self, data_path, data_dir, img_size, max_len, is_train, preprocessor=None):
                self.data_dir = Path(data_dir)
                self.img_size = img_size
                self.max_len = max_len
                self.preprocessor = preprocessor
                # Unreadable paths already warned about (logged once each).
                self._corrupt_paths = set()
                
                # Create vocab. The special ids are the module-level
                # constants so the model's generation seed (SOS_TOKEN_ID)
                # can never drift from this encoding (M8).
                self.char_to_idx = {c: i + 3 for i, c in enumerate(VIN_CHARACTERS)}
                self.char_to_idx['<PAD>'] = PAD_TOKEN_ID
                self.char_to_idx['<SOS>'] = SOS_TOKEN_ID
                self.char_to_idx['<EOS>'] = EOS_TOKEN_ID
                
                # Load samples
                self.samples = []
                if os.path.exists(data_path):
                    with open(data_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if '\t' in line:
                                img_path, label = line.split('\t', 1)
                                img_path, label = img_path.strip(), label.strip()
                                # Reject labels with characters outside the
                                # charset (L22): silently dropping them
                                # supervised the model on text that is NOT
                                # the image's text.
                                unknown = sorted({c for c in label if c not in self.char_to_idx})
                                if unknown:
                                    logger.warning(
                                        f"Rejecting sample {img_path}: label {label!r} "
                                        f"contains characters outside the charset: {unknown}"
                                    )
                                    continue
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
                
                # Skip unreadable images by advancing to the next sample
                # with a bounded LOOP, not recursion (M12): recursing on
                # (idx + 1) % len is a RecursionError on a run of bad files
                # and silently duplicates neighbours otherwise.
                img = None
                for offset in range(len(self.samples)):
                    img_path, label_text = self.samples[(idx + offset) % len(self.samples)]
                    full_path = self.data_dir / img_path
                    if not full_path.exists():
                        full_path = Path(img_path)
                    img = cv2.imread(str(full_path))
                    if img is not None:
                        break
                    if img_path not in self._corrupt_paths:
                        self._corrupt_paths.add(img_path)
                        logger.warning(f"Unreadable image skipped: {full_path}")
                if img is None:
                    raise RuntimeError(
                        f"No readable image in the entire dataset "
                        f"({len(self.samples)} samples); first path: "
                        f"{self.samples[idx][0]}"
                    )
                max_corrupt = max(1, int(len(self.samples) * self.MAX_CORRUPT_FRACTION))
                if len(self._corrupt_paths) > max_corrupt:
                    raise RuntimeError(
                        f"{len(self._corrupt_paths)} of {len(self.samples)} "
                        f"images are unreadable "
                        f"(>{self.MAX_CORRUPT_FRACTION:.0%}); aborting - a "
                        f"run padded with duplicated neighbours would measure "
                        f"the loader, not the dataset. Corrupt paths logged "
                        f"above."
                    )
                
                # Apply VIN-optimized preprocessing (CLAHE, morphology, etc.)
                if self.preprocessor is not None:
                    img = self.preprocessor.process(img)
                
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)
                
                # Encode label: <SOS> + chars + <EOS> + <PAD>. Labels were
                # validated against the charset at load time.
                label = [self.char_to_idx['<SOS>']]
                for c in label_text[:self.max_len]:
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
            is_training,
            preprocessor=vin_preprocessor
        )
        
        # Use num_workers=0 on macOS to avoid pickling issues
        import platform
        num_workers = 0 if platform.system() == 'Darwin' else 2
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=num_workers
        )
    
    def _evaluate(self, dataloader) -> Tuple[float, Dict[str, Any]]:
        """Evaluate model with comprehensive metrics."""
        import torch
        
        self.model.eval()
        
        idx_to_char = {i + 3: c for i, c in enumerate(VIN_CHARACTERS)}
        
        all_predictions = []
        all_targets = []
        
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
                    
                    all_predictions.append(pred_text)
                    all_targets.append(label_text)
        
        self.model.train()
        
        # Calculate comprehensive metrics
        try:
            from ..evaluation.metrics import EvaluationMetricsCalculator
            calc = EvaluationMetricsCalculator()
            calc.add_batch(all_predictions, all_targets)
            full_metrics = calc.compute()
            
            metrics = {
                'correct_images': full_metrics.image_level.correct_images,
                'failed_images': full_metrics.image_level.failed_images,
                'total_images': full_metrics.image_level.total_images,
                'image_accuracy': full_metrics.image_level.accuracy,
                'char_accuracy': full_metrics.character_level.char_accuracy,
                'f1_micro': full_metrics.character_level.f1_micro,
                'f1_macro': full_metrics.character_level.f1_macro,
                'cer': full_metrics.character_level.char_error_rate,
                'ned': full_metrics.character_level.normalized_edit_distance,
            }
            return metrics['image_accuracy'], metrics
        except ImportError:
            correct = sum(1 for p, t in zip(all_predictions, all_targets, strict=True) if p == t)
            total = len(all_predictions)
            accuracy = correct / total if total > 0 else 0.0
            return accuracy, {'image_accuracy': accuracy}
    
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
    
    # Data paths
    parser.add_argument("--train-data-dir", type=str, help="Training data directory")
    parser.add_argument("--train-labels", type=str, help="Training labels file")
    parser.add_argument("--val-data-dir", type=str, help="Validation data directory")
    parser.add_argument("--val-labels", type=str, help="Validation labels file")
    
    # Architecture (for PaddleOCR)
    parser.add_argument("--architecture", type=str, 
                       choices=["PP-OCRv5", "CRNN", "SVTR_LCNet", "SVTR_Tiny"],
                       help="Model architecture (PP-OCRv5 recommended)")
    
    # Device
    parser.add_argument("--device", type=str, help="Device to use (gpu/cpu/auto)")
    
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
        
        # Data paths
        if args.train_data_dir:
            config.train_data_dir = args.train_data_dir
        if args.train_labels:
            config.train_label_file = args.train_labels
        if args.val_data_dir:
            config.val_data_dir = args.val_data_dir
        if args.val_labels:
            config.val_label_file = args.val_labels
        
        # Architecture
        if args.architecture:
            config.architecture = args.architecture
        
        # Device
        if args.device:
            config.use_gpu = args.device.lower() in ("gpu", "cuda", "auto")
        
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
        
        # Data paths (DeepSeek uses different field names)
        if args.train_data_dir:
            config.data_dir = args.train_data_dir
        if args.train_labels:
            config.train_data_path = args.train_labels
        if args.val_labels:
            config.val_data_path = args.val_labels
        
        # Device
        if args.device:
            config.use_gpu = args.device.lower() in ("gpu", "cuda", "mps", "auto")
        
        trainer = DeepSeekScratchTrainer(config)
        trainer.setup()
        trainer.train()


if __name__ == "__main__":
    main()
