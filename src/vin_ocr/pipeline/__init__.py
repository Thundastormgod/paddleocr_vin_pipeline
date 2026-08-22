"""
VIN OCR Pipeline - Main Pipeline Module
=======================================

Contains the core VIN recognition pipeline.
"""

from .vin_pipeline import VINOCRPipeline, MultiProviderVINPipeline

__all__ = ["VINOCRPipeline", "MultiProviderVINPipeline"]
