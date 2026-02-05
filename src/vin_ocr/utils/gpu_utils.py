#!/usr/bin/env python3
"""
GPU Utilities for VIN OCR Pipeline
===================================

Centralized GPU detection, configuration, and management for all models.

Supports:
- NVIDIA CUDA GPUs
- Apple Silicon MPS (Metal Performance Shaders)
- AMD ROCm GPUs
- Fallback to CPU

Usage:
    from src.vin_ocr.utils.gpu_utils import GPUManager, get_gpu_manager
    
    gpu = get_gpu_manager()
    device = gpu.get_best_device()
    
    # Or check specific backends
    if gpu.cuda_available:
        print(f"CUDA: {gpu.cuda_device_name}")

Author: JRL-VIN Project
Date: February 2026
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    ROCM = "rocm"


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_type: DeviceType
    device_id: int = 0
    name: str = "Unknown"
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    is_available: bool = False
    error_message: Optional[str] = None


@dataclass
class GPUStatus:
    """Overall GPU status for the system."""
    # Availability flags
    cuda_available: bool = False
    mps_available: bool = False
    rocm_available: bool = False
    
    # Device info
    cuda_devices: List[GPUInfo] = field(default_factory=list)
    mps_info: Optional[GPUInfo] = None
    
    # Best device recommendation
    best_device: DeviceType = DeviceType.CPU
    best_device_name: str = "CPU"
    
    # Dependencies
    torch_available: bool = False
    torch_version: Optional[str] = None
    paddle_available: bool = False
    paddle_version: Optional[str] = None
    paddle_gpu: bool = False
    
    # Error messages
    errors: List[str] = field(default_factory=list)


class GPUManager:
    """
    Centralized GPU management for all OCR models.
    
    Detects available GPU backends and provides unified device selection.
    """
    
    _instance: Optional['GPUManager'] = None
    
    def __new__(cls):
        """Singleton pattern to avoid repeated GPU detection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._status: Optional[GPUStatus] = None
        self._use_gpu: bool = True  # Default to attempting GPU use
        self._preferred_device: Optional[DeviceType] = None
        self._initialized = True
    
    def detect_all(self, force_refresh: bool = False) -> GPUStatus:
        """
        Detect all available GPU backends.
        
        Args:
            force_refresh: Re-detect even if already cached
            
        Returns:
            GPUStatus with all detection results
        """
        if self._status is not None and not force_refresh:
            return self._status
        
        status = GPUStatus()
        
        # Detect PyTorch backends
        self._detect_torch(status)
        
        # Detect PaddlePaddle GPU
        self._detect_paddle(status)
        
        # Determine best device
        self._determine_best_device(status)
        
        self._status = status
        return status
    
    def _detect_torch(self, status: GPUStatus):
        """Detect PyTorch GPU backends."""
        try:
            import torch
            status.torch_available = True
            status.torch_version = torch.__version__
            
            # CUDA detection
            if torch.cuda.is_available():
                status.cuda_available = True
                for i in range(torch.cuda.device_count()):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        total_mem = props.total_memory / (1024**3)
                        
                        # Get available memory
                        try:
                            torch.cuda.set_device(i)
                            free_mem = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3)
                        except:
                            free_mem = total_mem
                        
                        gpu_info = GPUInfo(
                            device_type=DeviceType.CUDA,
                            device_id=i,
                            name=props.name,
                            memory_total_gb=round(total_mem, 2),
                            memory_available_gb=round(free_mem, 2),
                            compute_capability=f"{props.major}.{props.minor}",
                            is_available=True
                        )
                        status.cuda_devices.append(gpu_info)
                    except Exception as e:
                        status.errors.append(f"CUDA device {i} error: {e}")
            
            # MPS detection (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                status.mps_available = True
                status.mps_info = GPUInfo(
                    device_type=DeviceType.MPS,
                    device_id=0,
                    name="Apple Silicon GPU",
                    is_available=True
                )
                
                # Try to get more info
                try:
                    if hasattr(torch.backends.mps, 'is_built') and torch.backends.mps.is_built():
                        status.mps_info.name = "Apple Silicon GPU (MPS Built)"
                except:
                    pass
            
            # ROCm detection
            if hasattr(torch, 'hip') or 'rocm' in torch.__version__.lower():
                try:
                    if torch.cuda.is_available():  # ROCm uses CUDA interface
                        status.rocm_available = True
                except:
                    pass
                    
        except ImportError:
            status.errors.append("PyTorch not installed")
        except Exception as e:
            status.errors.append(f"PyTorch detection error: {e}")
    
    def _detect_paddle(self, status: GPUStatus):
        """Detect PaddlePaddle GPU support."""
        try:
            import paddle
            status.paddle_available = True
            status.paddle_version = paddle.__version__
            
            # Check if PaddlePaddle has GPU support
            try:
                status.paddle_gpu = paddle.device.is_compiled_with_cuda()
                if status.paddle_gpu:
                    # Verify GPU is actually usable
                    gpu_count = paddle.device.cuda.device_count()
                    if gpu_count > 0:
                        logger.info(f"PaddlePaddle GPU available: {gpu_count} device(s)")
                    else:
                        status.paddle_gpu = False
            except Exception as e:
                status.errors.append(f"PaddlePaddle GPU check error: {e}")
                status.paddle_gpu = False
                
        except ImportError:
            status.errors.append("PaddlePaddle not installed")
        except Exception as e:
            status.errors.append(f"PaddlePaddle detection error: {e}")
    
    def _determine_best_device(self, status: GPUStatus):
        """Determine the best available device."""
        if status.cuda_available and status.cuda_devices:
            # Pick CUDA device with most memory
            best_cuda = max(status.cuda_devices, key=lambda x: x.memory_total_gb)
            status.best_device = DeviceType.CUDA
            status.best_device_name = f"CUDA: {best_cuda.name}"
        elif status.mps_available:
            status.best_device = DeviceType.MPS
            status.best_device_name = "Apple Silicon (MPS)"
        elif status.rocm_available:
            status.best_device = DeviceType.ROCM
            status.best_device_name = "AMD ROCm"
        else:
            status.best_device = DeviceType.CPU
            status.best_device_name = "CPU"
    
    @property
    def status(self) -> GPUStatus:
        """Get cached GPU status, detecting if needed."""
        if self._status is None:
            self.detect_all()
        return self._status
    
    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self.status.cuda_available
    
    @property
    def mps_available(self) -> bool:
        """Check if MPS (Apple Silicon) is available."""
        return self.status.mps_available
    
    @property
    def any_gpu_available(self) -> bool:
        """Check if any GPU is available."""
        return self.status.cuda_available or self.status.mps_available or self.status.rocm_available
    
    @property
    def use_gpu(self) -> bool:
        """Get current GPU usage preference."""
        return self._use_gpu and self.any_gpu_available
    
    @use_gpu.setter
    def use_gpu(self, value: bool):
        """Set GPU usage preference."""
        self._use_gpu = value
    
    def get_device_string(self, for_torch: bool = True) -> str:
        """
        Get device string for model loading.
        
        Args:
            for_torch: If True, return PyTorch device string.
                      If False, return PaddlePaddle device string.
        
        Returns:
            Device string like 'cuda:0', 'mps', 'cpu', or 'gpu:0'
        """
        if not self._use_gpu:
            return "cpu"
        
        status = self.status
        
        if for_torch:
            if status.cuda_available:
                return "cuda:0"
            elif status.mps_available:
                return "mps"
            else:
                return "cpu"
        else:
            # PaddlePaddle format
            if status.paddle_gpu:
                return "gpu:0"
            else:
                return "cpu"
    
    def get_torch_device(self):
        """Get PyTorch device object."""
        try:
            import torch
            return torch.device(self.get_device_string(for_torch=True))
        except ImportError:
            return None
    
    def get_paddle_device(self) -> str:
        """Get PaddlePaddle device string."""
        return self.get_device_string(for_torch=False)
    
    def set_paddle_device(self):
        """Set PaddlePaddle to use the appropriate device."""
        try:
            import paddle
            device = self.get_paddle_device()
            paddle.set_device(device)
            logger.info(f"PaddlePaddle device set to: {device}")
        except Exception as e:
            logger.warning(f"Failed to set PaddlePaddle device: {e}")
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get GPU status as a dictionary for UI display."""
        status = self.status
        
        return {
            "gpu_available": self.any_gpu_available,
            "using_gpu": self.use_gpu,
            "best_device": status.best_device_name,
            "cuda": {
                "available": status.cuda_available,
                "devices": [
                    {
                        "id": d.device_id,
                        "name": d.name,
                        "memory_total_gb": d.memory_total_gb,
                        "memory_available_gb": d.memory_available_gb,
                        "compute_capability": d.compute_capability,
                    }
                    for d in status.cuda_devices
                ] if status.cuda_available else []
            },
            "mps": {
                "available": status.mps_available,
                "name": status.mps_info.name if status.mps_info else None
            },
            "rocm": {
                "available": status.rocm_available
            },
            "torch": {
                "available": status.torch_available,
                "version": status.torch_version
            },
            "paddle": {
                "available": status.paddle_available,
                "version": status.paddle_version,
                "gpu_support": status.paddle_gpu
            },
            "errors": status.errors
        }
    
    def get_summary_string(self) -> str:
        """Get a human-readable summary of GPU status."""
        status = self.status
        lines = []
        
        if self.any_gpu_available:
            lines.append(f"✅ GPU Available: {status.best_device_name}")
            
            if status.cuda_available:
                for d in status.cuda_devices:
                    lines.append(f"   CUDA {d.device_id}: {d.name} ({d.memory_total_gb}GB)")
            
            if status.mps_available:
                lines.append(f"   MPS: {status.mps_info.name}")
        else:
            lines.append("❌ No GPU available - using CPU")
        
        if status.errors:
            lines.append("\nWarnings:")
            for err in status.errors[:3]:  # Limit to 3 errors
                lines.append(f"   ⚠️ {err}")
        
        return "\n".join(lines)
    
    def get_training_recommendations(self, model_type: str = "deepseek") -> Dict[str, Any]:
        """
        Get recommended training settings based on available GPU.
        
        Optimized for common GPU configurations including RTX 3090 (24GB).
        
        Args:
            model_type: Type of model ('deepseek', 'paddleocr')
            
        Returns:
            Dictionary with recommended training settings
        """
        status = self.status
        recommendations = {
            "gpu_available": self.any_gpu_available,
            "device": self.get_device_string(),
            "warnings": [],
        }
        
        if not self.any_gpu_available:
            recommendations["warnings"].append("No GPU available - training will be slow")
            recommendations["batch_size"] = 1
            recommendations["gradient_accumulation"] = 16
            recommendations["use_fp16"] = False
            recommendations["use_bf16"] = False
            recommendations["gradient_checkpointing"] = True
            recommendations["use_8bit"] = False
            return recommendations
        
        # Get GPU memory
        gpu_memory_gb = 0
        gpu_name = "Unknown"
        if status.cuda_available and status.cuda_devices:
            best_gpu = max(status.cuda_devices, key=lambda x: x.memory_total_gb)
            gpu_memory_gb = best_gpu.memory_total_gb
            gpu_name = best_gpu.name
        
        recommendations["gpu_name"] = gpu_name
        recommendations["gpu_memory_gb"] = gpu_memory_gb
        
        if model_type == "deepseek":
            # DeepSeek-OCR VLM settings
            if gpu_memory_gb >= 48:
                # A100/A6000 class - can do full fine-tuning
                recommendations.update({
                    "batch_size": 8,
                    "gradient_accumulation": 4,
                    "use_lora": False,  # Can do full fine-tuning
                    "use_bf16": True,
                    "use_fp16": False,
                    "gradient_checkpointing": False,
                    "use_8bit": False,
                })
            elif gpu_memory_gb >= 20:
                # RTX 3090/4090 class (24GB) - LoRA with bf16
                recommendations.update({
                    "batch_size": 4,
                    "gradient_accumulation": 8,
                    "use_lora": True,
                    "use_bf16": True,
                    "use_fp16": False,
                    "gradient_checkpointing": True,
                    "use_8bit": False,
                })
                if "3090" in gpu_name or "4090" in gpu_name:
                    recommendations["notes"] = "RTX 3090/4090 detected - optimal settings applied"
            elif gpu_memory_gb >= 14:
                # RTX 3080/4080 class (16GB) - LoRA with 8-bit
                recommendations.update({
                    "batch_size": 2,
                    "gradient_accumulation": 16,
                    "use_lora": True,
                    "use_bf16": False,
                    "use_fp16": True,
                    "gradient_checkpointing": True,
                    "use_8bit": True,
                })
                recommendations["warnings"].append("Limited VRAM - using 8-bit quantization")
            else:
                # Smaller GPUs - may struggle
                recommendations.update({
                    "batch_size": 1,
                    "gradient_accumulation": 32,
                    "use_lora": True,
                    "use_bf16": False,
                    "use_fp16": True,
                    "gradient_checkpointing": True,
                    "use_8bit": True,
                })
                recommendations["warnings"].append(
                    f"GPU has only {gpu_memory_gb}GB VRAM - DeepSeek training may be slow or fail"
                )
        else:
            # PaddleOCR settings (less memory intensive)
            if gpu_memory_gb >= 8:
                recommendations.update({
                    "batch_size": 64,
                    "use_fp16": True,
                })
            elif gpu_memory_gb >= 4:
                recommendations.update({
                    "batch_size": 32,
                    "use_fp16": True,
                })
            else:
                recommendations.update({
                    "batch_size": 16,
                    "use_fp16": False,
                })
        
        return recommendations


# Global instance accessor
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def check_gpu_available() -> bool:
    """Quick check if any GPU is available."""
    return get_gpu_manager().any_gpu_available


def get_device(for_torch: bool = True) -> str:
    """Get the best device string."""
    return get_gpu_manager().get_device_string(for_torch=for_torch)


def set_use_gpu(enabled: bool):
    """Set whether to use GPU."""
    get_gpu_manager().use_gpu = enabled


# Environment variable support
if os.environ.get("VIN_OCR_USE_CPU", "").lower() in ("1", "true", "yes"):
    get_gpu_manager().use_gpu = False
    logger.info("GPU disabled via VIN_OCR_USE_CPU environment variable")
