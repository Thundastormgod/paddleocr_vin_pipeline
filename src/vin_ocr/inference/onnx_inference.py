#!/usr/bin/env python3
"""
ONNX Inference for VIN OCR Models
==================================

Provides high-performance inference using ONNX Runtime for:
- Fine-tuned PaddleOCR models (exported to ONNX)
- Train-from-scratch models (PP-OCRv5, SVTR, CRNN)
- Cross-platform deployment (CPU, GPU, CoreML)

Features:
- Automatic hardware detection (CUDA, CoreML, CPU)
- Batch inference support
- CTC decoding for text recognition
- VIN-specific character set handling

Usage:
    from src.vin_ocr.inference.onnx_inference import ONNXVINRecognizer
    
    # Load model
    recognizer = ONNXVINRecognizer("output/onnx/model.onnx")
    
    # Single image
    result = recognizer.recognize("image.jpg")
    print(f"VIN: {result['vin']}, Confidence: {result['confidence']:.2f}")
    
    # Batch inference
    results = recognizer.recognize_batch(["img1.jpg", "img2.jpg"])

Author: VIN OCR Pipeline
License: MIT
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

# VIN valid characters (excludes I, O, Q as per ISO 3779)
VIN_CHARSET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
VIN_CHARSET_WITH_BLANK = VIN_CHARSET + " "  # Blank token for CTC


@dataclass
class ONNXInferenceConfig:
    """Configuration for ONNX inference."""
    
    # Input preprocessing
    input_height: int = 48
    input_width: int = 320
    input_channels: int = 3  # RGB or grayscale (1)
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    
    # CTC decoding
    blank_idx: int = 0  # CTC blank token index
    charset: str = VIN_CHARSET
    
    # Hardware
    use_gpu: bool = True
    gpu_id: int = 0
    use_coreml: bool = True  # For Apple Silicon
    
    # Inference
    batch_size: int = 1
    

class ONNXVINRecognizer:
    """
    ONNX-based VIN text recognizer.
    
    Supports models exported from:
    - PaddleOCR fine-tuned models
    - Train-from-scratch models (PP-OCRv5, SVTR, CRNN)
    
    Example:
        recognizer = ONNXVINRecognizer("model.onnx")
        result = recognizer.recognize("vin_image.jpg")
        print(f"VIN: {result['vin']}")
    """
    
    def __init__(
        self, 
        model_path: str,
        config: Optional[ONNXInferenceConfig] = None,
        char_dict_path: Optional[str] = None
    ):
        """
        Initialize ONNX recognizer.
        
        Args:
            model_path: Path to the ONNX model file
            config: Optional inference configuration
            char_dict_path: Optional path to character dictionary file
        """
        self.model_path = Path(model_path)
        self.config = config or ONNXInferenceConfig()
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        
        # Load character dictionary
        self.char_dict = self._load_char_dict(char_dict_path)
        self.idx_to_char = {v: k for k, v in self.char_dict.items()}
        
        # Load model
        self._load_model()
        
    def _load_char_dict(self, dict_path: Optional[str] = None) -> Dict[str, int]:
        """Load character dictionary."""
        char_dict = {'<blank>': 0}  # CTC blank token
        
        if dict_path and Path(dict_path).exists():
            with open(dict_path, 'r') as f:
                for idx, line in enumerate(f, start=1):
                    char = line.strip()
                    if char:
                        char_dict[char] = idx
        else:
            # Use default VIN charset
            for idx, char in enumerate(self.config.charset, start=1):
                char_dict[char] = idx
                
        return char_dict
    
    def _load_model(self):
        """Load ONNX model with appropriate execution provider."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX Runtime not installed. Install with:\n"
                "  pip install onnxruntime       # CPU only\n"
                "  pip install onnxruntime-gpu   # With CUDA support"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        
        # Determine execution providers
        providers = []
        
        if self.config.use_gpu:
            # Try CUDA first
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': self.config.gpu_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }))
                logger.info(f"Using CUDA GPU {self.config.gpu_id} for ONNX inference")
            
            # Try CoreML for Apple Silicon
            elif self.config.use_coreml and 'CoreMLExecutionProvider' in ort.get_available_providers():
                providers.append('CoreMLExecutionProvider')
                logger.info("Using CoreML (Apple Silicon) for ONNX inference")
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output info
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Update config from model if shape is fixed
        if len(self.input_shape) == 4:
            if isinstance(self.input_shape[2], int):
                self.config.input_height = self.input_shape[2]
            if isinstance(self.input_shape[3], int):
                self.config.input_width = self.input_shape[3]
            if isinstance(self.input_shape[1], int):
                self.config.input_channels = self.input_shape[1]
        
        actual_provider = self.session.get_providers()[0]
        logger.info(f"ONNX model loaded: {self.model_path.name}")
        logger.info(f"  Provider: {actual_provider}")
        logger.info(f"  Input: {self.input_name} {self.input_shape}")
        logger.info(f"  Outputs: {self.output_names}")
    
    def preprocess(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Path to image file or numpy array (BGR or grayscale)
            
        Returns:
            Preprocessed image tensor [1, C, H, W]
        """
        import cv2
        
        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        else:
            img = image.copy()
        
        # Convert to target channels
        if self.config.input_channels == 1:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.config.input_channels == 3:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        target_h, target_w = self.config.input_height, self.config.input_width
        
        ratio = target_h / h
        new_w = min(int(w * ratio), target_w)
        
        img = cv2.resize(img, (new_w, target_h))
        
        # Pad to target width
        if new_w < target_w:
            if self.config.input_channels == 1:
                padded = np.zeros((target_h, target_w), dtype=np.uint8)
                padded[:, :new_w] = img
            else:
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                padded[:, :new_w] = img
            img = padded
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.config.normalize_mean) / self.config.normalize_std
        
        # Reshape to [1, C, H, W]
        if self.config.input_channels == 1:
            img = np.expand_dims(img, axis=0)  # [H, W] -> [1, H, W]
        else:
            img = np.transpose(img, (2, 0, 1))  # [H, W, C] -> [C, H, W]
        
        img = np.expand_dims(img, axis=0)  # [C, H, W] -> [1, C, H, W]
        
        return img.astype(np.float32)
    
    def decode_ctc(self, output: np.ndarray) -> Tuple[str, float]:
        """
        Decode CTC output to text.
        
        Args:
            output: Model output logits [B, T, num_classes] or [T, num_classes]
            
        Returns:
            Tuple of (decoded_text, confidence_score)
        """
        # Handle batch dimension
        if output.ndim == 3:
            output = output[0]  # [T, num_classes]
        
        # Get predicted indices and probabilities
        pred_probs = self._softmax(output)
        pred_indices = np.argmax(pred_probs, axis=1)
        pred_max_probs = np.max(pred_probs, axis=1)
        
        # CTC greedy decoding (collapse repeated + remove blanks)
        decoded_chars = []
        decoded_probs = []
        prev_idx = self.config.blank_idx
        
        for t, (idx, prob) in enumerate(zip(pred_indices, pred_max_probs)):
            if idx != self.config.blank_idx and idx != prev_idx:
                char = self.idx_to_char.get(idx, '')
                if char and char != '<blank>':
                    decoded_chars.append(char)
                    decoded_probs.append(prob)
            prev_idx = idx
        
        text = ''.join(decoded_chars)
        confidence = float(np.mean(decoded_probs)) if decoded_probs else 0.0
        
        return text, confidence
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax along the last axis."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def recognize(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Recognize VIN from a single image.
        
        Args:
            image: Path to image file or numpy array
            
        Returns:
            Dict with 'vin', 'raw_text', 'confidence', 'is_valid'
        """
        try:
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Decode
            raw_text, confidence = self.decode_ctc(outputs[0])
            
            # Extract VIN (17 characters)
            vin = raw_text[:17] if len(raw_text) >= 17 else raw_text
            is_valid = len(vin) == 17 and all(c in VIN_CHARSET for c in vin)
            
            return {
                'vin': vin,
                'raw_text': raw_text,
                'confidence': confidence,
                'is_valid': is_valid,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return {
                'vin': '',
                'raw_text': '',
                'confidence': 0.0,
                'is_valid': False,
                'error': str(e)
            }
    
    def recognize_batch(
        self, 
        images: List[Union[str, np.ndarray]],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Recognize VINs from multiple images.
        
        Args:
            images: List of image paths or numpy arrays
            batch_size: Batch size for inference (default: config.batch_size)
            
        Returns:
            List of recognition results
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                try:
                    tensor = self.preprocess(img)
                    batch_tensors.append(tensor[0])  # Remove batch dim
                except Exception as e:
                    logger.warning(f"Failed to preprocess image: {e}")
                    results.append({
                        'vin': '', 'raw_text': '', 'confidence': 0.0,
                        'is_valid': False, 'error': str(e)
                    })
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack into batch
            batch_input = np.stack(batch_tensors, axis=0)
            
            # Run inference
            try:
                outputs = self.session.run(self.output_names, {self.input_name: batch_input})
                
                # Decode each result
                for j in range(len(batch_tensors)):
                    raw_text, confidence = self.decode_ctc(outputs[0][j:j+1])
                    vin = raw_text[:17] if len(raw_text) >= 17 else raw_text
                    is_valid = len(vin) == 17 and all(c in VIN_CHARSET for c in vin)
                    
                    results.append({
                        'vin': vin,
                        'raw_text': raw_text,
                        'confidence': confidence,
                        'is_valid': is_valid,
                        'error': None
                    })
                    
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                for _ in batch_tensors:
                    results.append({
                        'vin': '', 'raw_text': '', 'confidence': 0.0,
                        'is_valid': False, 'error': str(e)
                    })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            'model_path': str(self.model_path),
            'input_name': self.input_name,
            'input_shape': self.input_shape,
            'output_names': self.output_names,
            'provider': self.session.get_providers()[0] if self.session else None,
            'charset_size': len(self.char_dict),
            'config': {
                'input_height': self.config.input_height,
                'input_width': self.config.input_width,
                'input_channels': self.config.input_channels,
            }
        }


def load_onnx_model(model_path: str, **kwargs) -> ONNXVINRecognizer:
    """
    Convenience function to load an ONNX model.
    
    Args:
        model_path: Path to the ONNX model file
        **kwargs: Additional config options
        
    Returns:
        ONNXVINRecognizer instance
    """
    config = ONNXInferenceConfig(**kwargs) if kwargs else None
    return ONNXVINRecognizer(model_path, config=config)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for ONNX inference."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run VIN recognition with ONNX model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recognize single image
  python -m src.vin_ocr.inference.onnx_inference --model output/onnx/model.onnx --image test.jpg
  
  # Recognize multiple images
  python -m src.vin_ocr.inference.onnx_inference --model model.onnx --images img1.jpg img2.jpg img3.jpg
  
  # Recognize from directory
  python -m src.vin_ocr.inference.onnx_inference --model model.onnx --dir test_images/
        """
    )
    
    parser.add_argument('--model', '-m', required=True, help='Path to ONNX model')
    parser.add_argument('--image', '-i', help='Single image path')
    parser.add_argument('--images', nargs='+', help='Multiple image paths')
    parser.add_argument('--dir', '-d', help='Directory containing images')
    parser.add_argument('--char-dict', help='Character dictionary file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    # Collect images
    images = []
    if args.image:
        images.append(args.image)
    if args.images:
        images.extend(args.images)
    if args.dir:
        import glob
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(glob.glob(os.path.join(args.dir, ext)))
            images.extend(glob.glob(os.path.join(args.dir, ext.upper())))
    
    if not images:
        print("No images provided. Use --image, --images, or --dir")
        sys.exit(1)
    
    # Load model
    config = ONNXInferenceConfig(
        use_gpu=args.gpu,
        batch_size=args.batch_size
    )
    recognizer = ONNXVINRecognizer(args.model, config=config, char_dict_path=args.char_dict)
    
    # Print model info
    info = recognizer.get_model_info()
    print(f"\nModel: {info['model_path']}")
    print(f"Provider: {info['provider']}")
    print(f"Input: {info['input_shape']}")
    print()
    
    # Run recognition
    if len(images) == 1:
        result = recognizer.recognize(images[0])
        print(f"Image: {images[0]}")
        print(f"  VIN: {result['vin']}")
        print(f"  Raw: {result['raw_text']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Valid: {result['is_valid']}")
        if result['error']:
            print(f"  Error: {result['error']}")
    else:
        results = recognizer.recognize_batch(images)
        print(f"Processed {len(images)} images:\n")
        
        valid_count = sum(1 for r in results if r['is_valid'])
        error_count = sum(1 for r in results if r['error'])
        
        for img, result in zip(images, results):
            status = "✓" if result['is_valid'] else "✗"
            print(f"  {status} {Path(img).name}: {result['vin']} ({result['confidence']:.2f})")
        
        print(f"\nSummary: {valid_count}/{len(images)} valid VINs, {error_count} errors")


if __name__ == '__main__':
    main()
