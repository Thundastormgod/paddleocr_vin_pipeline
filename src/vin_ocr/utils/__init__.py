"""
VIN OCR Utils Module
====================

Data preparation, validation, and GPU management utilities.

Available Scripts:
- prepare_dataset.py: Prepare VIN dataset for training
- prepare_finetune_data.py: Prepare data specifically for fine-tuning
- validate_dataset.py: Validate dataset integrity and labels

GPU Utilities:
- gpu_utils.py: Centralized GPU detection and management

Usage (CLI):
    # Prepare dataset
    python -m src.vin_ocr.utils.prepare_dataset --input_dir ./raw_data --output_dir ./data
    
    # Validate dataset
    python -m src.vin_ocr.utils.validate_dataset --data_dir ./data/train

Usage (Python):
    from src.vin_ocr.utils.gpu_utils import get_gpu_manager, check_gpu_available
    
    gpu = get_gpu_manager()
    if gpu.any_gpu_available:
        device = gpu.get_device_string()
"""

# GPU utilities
from .gpu_utils import (
    GPUManager,
    GPUInfo,
    GPUStatus,
    DeviceType,
    get_gpu_manager,
    check_gpu_available,
    get_device,
    set_use_gpu,
)

__all__ = [
    # Data utils
    "prepare_dataset",
    "prepare_finetune_data",
    "validate_dataset",
    # GPU utils
    "GPUManager",
    "GPUInfo", 
    "GPUStatus",
    "DeviceType",
    "get_gpu_manager",
    "check_gpu_available",
    "get_device",
    "set_use_gpu",
]
