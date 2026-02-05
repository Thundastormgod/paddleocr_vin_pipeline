#!/usr/bin/env python3
"""
Direct Paddle Inference for Fine-tuned VIN Models
==================================================

This module provides direct inference using PaddlePaddle's inference API,
bypassing the PaddleOCR wrapper which has compatibility issues with custom models.

Usage:
    from src.vin_ocr.inference.paddle_inference import VINInference
    
    inference = VINInference("output/final_test/inference")
    result = inference.recognize("image.jpg")
    print(f"VIN: {result['vin']}, Confidence: {result['confidence']}")
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


# VIN charset (no I, O, Q per ISO 3779)
VIN_CHARSET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"


class VINInference:
    """
    Direct PaddlePaddle inference for fine-tuned VIN recognition models.
    
    This bypasses PaddleOCR's wrapper to work with custom models that don't
    match PaddleOCR v5's expected format.
    """
    
    def __init__(
        self,
        model_dir: str,
        char_dict_path: Optional[str] = None,
        use_gpu: bool = False,
        input_shape: tuple = (3, 48, 320),
    ):
        """
        Initialize the VIN inference engine.
        
        Args:
            model_dir: Path to inference model directory containing:
                       - inference.json (or inference.pdmodel)
                       - inference.pdiparams
            char_dict_path: Path to character dictionary (optional)
            use_gpu: Whether to use GPU
            input_shape: Model input shape (C, H, W)
        """
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu
        self.input_channels, self.input_height, self.input_width = input_shape
        
        # Load character dictionary
        self.char_dict = self._load_char_dict(char_dict_path)
        self.idx_to_char = {v: k for k, v in self.char_dict.items()}
        
        # Load model
        self.predictor = self._load_model()
        
    def _load_char_dict(self, dict_path: Optional[str] = None) -> Dict[str, int]:
        """Load character dictionary."""
        char_dict = {'<blank>': 0}
        
        # Try to find dict in model directory
        if dict_path is None:
            possible_dicts = [
                self.model_dir / "vin_dict.txt",
                self.model_dir.parent / "vin_dict.txt",
                Path("configs/vin_dict.txt"),
            ]
            for p in possible_dicts:
                if p.exists():
                    dict_path = str(p)
                    break
        
        if dict_path and Path(dict_path).exists():
            with open(dict_path, 'r') as f:
                for idx, line in enumerate(f, start=1):
                    char = line.strip()
                    if char:
                        char_dict[char] = idx
        else:
            # Use default VIN charset
            for idx, char in enumerate(VIN_CHARSET, start=1):
                char_dict[char] = idx
                
        return char_dict
    
    def _load_model(self):
        """Load the Paddle inference model."""
        from paddle import inference
        
        # Find model files
        model_file = None
        params_file = self.model_dir / "inference.pdiparams"
        
        # Check for .json (newer format) or .pdmodel (older format)
        if (self.model_dir / "inference.json").exists():
            model_file = self.model_dir / "inference.json"
        elif (self.model_dir / "inference.pdmodel").exists():
            model_file = self.model_dir / "inference.pdmodel"
        else:
            raise FileNotFoundError(f"No model file found in {self.model_dir}")
        
        if not params_file.exists():
            raise FileNotFoundError(f"Params file not found: {params_file}")
        
        # Create config
        config = inference.Config(str(model_file), str(params_file))
        
        if self.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(4)
        
        config.disable_glog_info()
        # Note: Don't use enable_memory_optim() - causes issues with Paddle 3.x
        
        # Create predictor
        predictor = inference.create_predictor(config)
        
        return predictor
    
    def preprocess(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Path to image or numpy array (BGR)
            
        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        else:
            img = image.copy()
        
        # Ensure 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        ratio = self.input_height / h
        new_w = min(int(w * ratio), self.input_width)
        
        img = cv2.resize(img, (new_w, self.input_height))
        
        # Pad to target width
        if new_w < self.input_width:
            padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
            padded[:, :new_w] = img
            img = padded
        
        # Convert BGR to RGB (if training used RGB)
        # Note: Most PaddleOCR models use BGR, so we keep BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, 0)
        
        return img
    
    def postprocess(self, output: np.ndarray) -> tuple:
        """
        CTC decode the model output.
        
        Args:
            output: Model output logits [batch, time, classes]
            
        Returns:
            Tuple of (decoded_text, confidence, raw_indices)
        """
        # Get predictions
        preds = np.argmax(output[0], axis=1)
        
        # Calculate confidences
        probs = self._softmax(output[0])
        
        # CTC decode (collapse repeats and remove blanks)
        prev_idx = -1
        decoded = []
        confidences = []
        
        for t in range(len(preds)):
            idx = preds[t]
            if idx != 0 and idx != prev_idx:  # Not blank and not repeat
                if idx in self.idx_to_char:
                    decoded.append(self.idx_to_char[idx])
                    confidences.append(float(probs[t, idx]))
            prev_idx = idx
        
        text = ''.join(decoded)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return text, avg_confidence, preds
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def recognize(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Recognize VIN from image.
        
        Args:
            image: Path to image or numpy array
            
        Returns:
            Dict with 'vin', 'confidence', 'raw_text', 'processing_time'
        """
        import time
        start_time = time.time()
        
        result = {
            'vin': '',
            'confidence': 0.0,
            'raw_text': '',
            'processing_time': 0.0,
            'error': None
        }
        
        try:
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Get input handle
            input_names = self.predictor.get_input_names()
            input_handle = self.predictor.get_input_handle(input_names[0])
            input_handle.copy_from_cpu(input_tensor)
            
            # Run inference
            self.predictor.run()
            
            # Get output
            output_names = self.predictor.get_output_names()
            output_handle = self.predictor.get_output_handle(output_names[0])
            output = output_handle.copy_to_cpu()
            
            # Postprocess
            text, confidence, _ = self.postprocess(output)
            
            result['raw_text'] = text
            result['confidence'] = confidence
            
            # Extract VIN (should be 17 chars)
            # Filter to valid VIN characters only
            vin_chars = ''.join(c for c in text.upper() if c in VIN_CHARSET)
            result['vin'] = vin_chars[:17] if len(vin_chars) >= 17 else vin_chars
            
        except Exception as e:
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def recognize_batch(self, images: List[Union[str, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Recognize VINs from multiple images.
        
        Args:
            images: List of image paths or numpy arrays
            
        Returns:
            List of recognition results
        """
        return [self.recognize(img) for img in images]


def test_inference():
    """Test the inference with a sample image."""
    import sys
    
    # Find a test image and model
    test_images = list(Path("dataset/test").glob("*.jpg"))[:1]
    models = list(Path("output").glob("*/inference"))
    
    if not test_images:
        print("No test images found in dataset/test/")
        return
    
    if not models:
        print("No inference models found in output/*/inference/")
        return
    
    model_dir = models[0]
    test_image = test_images[0]
    
    print(f"Model: {model_dir}")
    print(f"Image: {test_image}")
    
    # Initialize and run
    inference = VINInference(str(model_dir))
    result = inference.recognize(str(test_image))
    
    print(f"\nResult:")
    print(f"  VIN: {result['vin']}")
    print(f"  Raw: {result['raw_text']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Time: {result['processing_time']:.3f}s")
    
    if result['error']:
        print(f"  Error: {result['error']}")


if __name__ == "__main__":
    test_inference()
