"""
Hardware Detection and Configuration Utilities
===============================================

Detects available hardware (CPU, CUDA GPU, Apple Silicon MPS) and provides
configuration recommendations for training.

Features:
- Automatic device detection (CUDA, MPS, CPU)
- Memory estimation for model loading
- Quantization availability check
- Training configuration recommendations

Usage:
    from src.vin_ocr.utils.hardware_utils import HardwareDetector, get_hardware_info
    
    # Quick check
    info = get_hardware_info()
    print(info)
    
    # Detailed detector
    detector = HardwareDetector()
    config = detector.get_training_config("deepseek")

Author: VIN OCR Pipeline
Date: February 2026
"""

import os
import platform
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


def _paddle_cuda_device_present(paddle_module) -> bool:
    """
    True only when paddle is compiled with CUDA AND at least one CUDA
    device is actually present at runtime.

    is_compiled_with_cuda() is a build flag: it is True for GPU wheels
    running on GPU-less machines. Consumers use this value as `use_gpu`,
    so it must reflect runtime reality - selecting the 'gpu' device on a
    machine without one makes trainer init fail.
    """
    try:
        if not paddle_module.is_compiled_with_cuda():
            return False
        return paddle_module.device.cuda.device_count() > 0
    except AttributeError as exc:
        # Paddle build without the device-count API: report no usable GPU
        # rather than guessing from the compile flag.
        logger.warning("Paddle CUDA device-count API unavailable: %s", exc)
        return False
    except Exception:
        logger.exception("Paddle GPU probe failed; reporting CPU-only")
        return False


class DeviceType(str, Enum):
    """Available device types."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    index: int
    name: str
    total_memory_gb: float
    device_type: DeviceType
    compute_capability: Optional[Tuple[int, int]] = None  # CUDA only
    
    def __str__(self) -> str:
        return f"{self.name} ({self.total_memory_gb:.1f} GB)"


@dataclass
class HardwareInfo:
    """Complete hardware information."""
    # System
    platform: str
    python_version: str
    
    # CPU
    cpu_count: int
    cpu_name: Optional[str] = None
    
    # GPU
    device_type: DeviceType = DeviceType.CPU
    gpus: List[GPUInfo] = field(default_factory=list)
    total_gpu_memory_gb: float = 0.0
    
    # Libraries
    torch_available: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    mps_available: bool = False
    paddle_available: bool = False
    paddle_version: Optional[str] = None
    paddle_gpu: bool = False
    
    # Quantization
    bitsandbytes_available: bool = False
    quantization_supported: bool = False
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "HARDWARE INFORMATION",
            "=" * 60,
            f"Platform: {self.platform}",
            f"Python: {self.python_version}",
            f"CPU Cores: {self.cpu_count}",
            "",
            "--- PyTorch ---",
            f"PyTorch: {'✅ ' + self.torch_version if self.torch_available else '❌ Not installed'}",
            f"CUDA: {'✅ ' + self.cuda_version if self.cuda_available else '❌ Not available'}",
            f"MPS (Apple Silicon): {'✅ Available' if self.mps_available else '❌ Not available'}",
            "",
            "--- PaddlePaddle ---",
            f"PaddlePaddle: {'✅ ' + self.paddle_version if self.paddle_available else '❌ Not installed'}",
            f"Paddle GPU: {'✅ Available' if self.paddle_gpu else '❌ CPU only'}",
            "",
            "--- Device ---",
            f"Best Device: {self.device_type.value.upper()}",
        ]
        
        if self.gpus:
            lines.append(f"GPUs ({len(self.gpus)}):")
            for gpu in self.gpus:
                lines.append(f"  [{gpu.index}] {gpu}")
            lines.append(f"Total GPU Memory: {self.total_gpu_memory_gb:.1f} GB")
        
        lines.extend([
            "",
            "--- Quantization ---",
            f"BitsAndBytes: {'✅ Available' if self.bitsandbytes_available else '❌ Not installed'}",
            f"4-bit/8-bit Quantization: {'✅ Supported' if self.quantization_supported else '❌ Not supported (requires CUDA + BitsAndBytes)'}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class HardwareDetector:
    """
    Detects and reports hardware capabilities.
    
    Example:
        detector = HardwareDetector()
        info = detector.detect()
        print(info)
        
        # Get recommended config for a model type
        config = detector.get_training_config("deepseek")
    """
    
    def __init__(self):
        self._info: Optional[HardwareInfo] = None
    
    def detect(self, force_refresh: bool = False) -> HardwareInfo:
        """
        Detect hardware capabilities.
        
        Args:
            force_refresh: Re-detect even if already cached
            
        Returns:
            HardwareInfo object with all detected capabilities
        """
        if self._info is not None and not force_refresh:
            return self._info
        
        info = HardwareInfo(
            platform=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 1,
        )
        
        # Try to get the CPU name (macOS); failure only leaves cpu_name None.
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.cpu_name = result.stdout.strip()
                else:
                    logger.debug(
                        "sysctl machdep.cpu.brand_string failed (rc=%s): %s",
                        result.returncode, result.stderr.strip()
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.debug("Could not query CPU name: %s", exc)
        
        # Check PyTorch
        try:
            try:
                import torch
            except (ImportError, OSError) as torch_error:
                # ImportError: torch not installed. OSError: installed but
                # its native libraries fail to load (e.g. CUDA DLLs).
                # Anything else is a real bug and must propagate - the
                # previous `except Exception` silently converted arbitrary
                # torch failures into "torch unavailable".
                logger.warning("PyTorch unavailable: %s", torch_error)
                info.torch_available = False
                torch = None
            if torch is not None:
                info.torch_available = True
                info.torch_version = torch.__version__
            
            # Check CUDA
            if torch is not None and torch.cuda.is_available():
                info.cuda_available = True
                info.cuda_version = torch.version.cuda
                info.device_type = DeviceType.CUDA
                
                # Get GPU info
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu = GPUInfo(
                        index=i,
                        name=props.name,
                        total_memory_gb=props.total_memory / (1024**3),
                        device_type=DeviceType.CUDA,
                        compute_capability=(props.major, props.minor)
                    )
                    info.gpus.append(gpu)
                    info.total_gpu_memory_gb += gpu.total_memory_gb
            
            # Check MPS (Apple Silicon)
            elif torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info.mps_available = True
                info.device_type = DeviceType.MPS
                
                # Estimate Apple Silicon memory (unified memory). If the
                # sysctl probe fails - by exception OR nonzero exit - fall
                # back to a conservative estimate rather than reporting no
                # GPU memory at all.
                gpu_mem_gb = 8.0  # conservative fallback
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        total_mem_bytes = int(result.stdout.strip())
                        # Apple Silicon shares memory, estimate ~70% available for GPU
                        gpu_mem_gb = (total_mem_bytes / (1024**3)) * 0.7
                    else:
                        logger.debug(
                            "sysctl hw.memsize failed (rc=%s): %s",
                            result.returncode, result.stderr.strip()
                        )
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    logger.debug("Could not query hw.memsize: %s", exc)
                info.gpus.append(GPUInfo(
                    index=0,
                    name="Apple Silicon (MPS)",
                    total_memory_gb=gpu_mem_gb,
                    device_type=DeviceType.MPS
                ))
                info.total_gpu_memory_gb = gpu_mem_gb
        except ImportError as torch_error:
            # A torch submodule imported lazily during the probe is missing:
            # an availability signal, handled as unavailable (and said so).
            logger.warning("PyTorch unavailable: %s", torch_error)
            info.torch_available = False
        except Exception:
            # Real bug or driver fault inside the capability probe. Never
            # swallow it silently: log the full traceback, then explicitly
            # degrade to CPU-safe values so detect() keeps its contract of
            # always returning a usable HardwareInfo.
            logger.exception(
                "Unexpected error while probing PyTorch capabilities; "
                "treating PyTorch as unavailable"
            )
            info.torch_available = False
            info.torch_version = None
            info.cuda_available = False
            info.cuda_version = None
            info.mps_available = False
            info.gpus = []
            info.total_gpu_memory_gb = 0.0
            info.device_type = DeviceType.CPU
        
        # Check PaddlePaddle
        try:
            import paddle
            info.paddle_available = True
            info.paddle_version = paddle.__version__
            # Runtime device presence, not just the compile flag (compiled
            # with CUDA != a CUDA device exists on this machine).
            info.paddle_gpu = _paddle_cuda_device_present(paddle)
        except ImportError:
            pass
        
        # Check BitsAndBytes (for quantization)
        try:
            import bitsandbytes
            info.bitsandbytes_available = True
            # Quantization requires CUDA + BitsAndBytes
            info.quantization_supported = info.cuda_available
        except ImportError:
            info.bitsandbytes_available = False
            info.quantization_supported = False
        
        self._info = info
        return info
    
    def get_best_device(self) -> str:
        """Get the best available device string."""
        info = self.detect()
        return info.device_type.value
    
    def get_torch_device(self):
        """
        Get a torch.device object for the best available device.

        Returns None when torch is unavailable - the same two failure
        modes as the probe in detect(): not installed (ImportError) or
        installed with broken native libraries (OSError).
        """
        try:
            import torch
            return torch.device(self.get_best_device())
        except (ImportError, OSError) as torch_error:
            logger.warning("PyTorch unavailable: %s", torch_error)
            return None
    
    def can_use_quantization(self) -> bool:
        """Check if 4-bit/8-bit quantization is available."""
        info = self.detect()
        return info.quantization_supported
    
    def get_training_config(self, model_type: str = "paddleocr") -> Dict[str, Any]:
        """
        Get recommended training configuration based on hardware.
        
        Args:
            model_type: "paddleocr" or "deepseek"
            
        Returns:
            Dict with recommended configuration
        """
        info = self.detect()
        
        config = {
            "device": info.device_type.value,
            "device_name": info.gpus[0].name if info.gpus else "CPU",
            "total_memory_gb": info.total_gpu_memory_gb,
        }
        
        if model_type.lower() == "paddleocr":
            # PaddleOCR configuration
            config.update({
                "use_gpu": info.paddle_gpu,
                "use_amp": info.cuda_available,  # AMP only works well with CUDA
                "recommended_batch_size": self._recommend_batch_size_paddle(info),
                "num_workers": 0 if platform.system() == "Darwin" else min(4, info.cpu_count),
            })
            
        elif model_type.lower() == "deepseek":
            # DeepSeek configuration
            config.update({
                "use_cuda": info.cuda_available,
                "use_mps": info.mps_available and not info.cuda_available,
                "use_4bit": info.quantization_supported and info.total_gpu_memory_gb < 16,
                "use_8bit": info.quantization_supported and info.total_gpu_memory_gb >= 16,
                "use_lora": True,  # Always recommend LoRA for fine-tuning
                "bf16": info.cuda_available,  # bfloat16 works best on CUDA
                "fp16": info.mps_available and not info.cuda_available,
                "recommended_batch_size": self._recommend_batch_size_deepseek(info),
                "gradient_accumulation_steps": self._recommend_grad_accum(info),
            })
        
        return config
    
    def _recommend_batch_size_paddle(self, info: HardwareInfo) -> int:
        """Recommend batch size for PaddleOCR based on memory."""
        if not info.paddle_gpu:
            return 8  # CPU - conservative
        
        mem = info.total_gpu_memory_gb
        if mem >= 24:
            return 64
        elif mem >= 16:
            return 32
        elif mem >= 8:
            return 16
        else:
            return 8
    
    def _recommend_batch_size_deepseek(self, info: HardwareInfo) -> int:
        """Recommend batch size for DeepSeek based on memory."""
        mem = info.total_gpu_memory_gb
        
        if info.quantization_supported:
            # With quantization, can use larger batches
            if mem >= 24:
                return 8
            elif mem >= 16:
                return 4
            else:
                return 2
        else:
            # Without quantization, need more memory
            if mem >= 24:
                return 4
            elif mem >= 16:
                return 2
            else:
                return 1
    
    def _recommend_grad_accum(self, info: HardwareInfo) -> int:
        """Recommend gradient accumulation steps."""
        batch_size = self._recommend_batch_size_deepseek(info)
        # Target effective batch size of 16
        target_effective = 16
        return max(1, target_effective // batch_size)
    
    def print_info(self):
        """Print hardware information to console."""
        info = self.detect()
        print(info)
    
    def get_summary(self) -> str:
        """Get a short summary string."""
        info = self.detect()
        
        if info.cuda_available:
            gpu_info = f"CUDA ({info.gpus[0].name}, {info.total_gpu_memory_gb:.0f}GB)"
        elif info.mps_available:
            gpu_info = f"MPS (Apple Silicon, ~{info.total_gpu_memory_gb:.0f}GB)"
        else:
            gpu_info = "CPU only"
        
        quant = "4/8-bit OK" if info.quantization_supported else "No quantization"
        
        return f"{gpu_info} | {quant}"


# Singleton instance
_detector: Optional[HardwareDetector] = None


def get_hardware_detector() -> HardwareDetector:
    """Get the global hardware detector instance."""
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    return _detector


def get_hardware_info() -> HardwareInfo:
    """Quick function to get hardware info."""
    return get_hardware_detector().detect()


def get_best_device() -> str:
    """Quick function to get best device string."""
    return get_hardware_detector().get_best_device()


def get_training_config(model_type: str = "paddleocr") -> Dict[str, Any]:
    """Quick function to get training config."""
    return get_hardware_detector().get_training_config(model_type)


def print_hardware_info():
    """Print hardware info to console."""
    get_hardware_detector().print_info()


# CLI interface
if __name__ == "__main__":
    print_hardware_info()
    
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIGURATIONS")
    print("=" * 60)
    
    print("\n📦 PaddleOCR:")
    paddle_config = get_training_config("paddleocr")
    for k, v in paddle_config.items():
        print(f"  {k}: {v}")
    
    print("\n🤖 DeepSeek-OCR:")
    deepseek_config = get_training_config("deepseek")
    for k, v in deepseek_config.items():
        print(f"  {k}: {v}")
