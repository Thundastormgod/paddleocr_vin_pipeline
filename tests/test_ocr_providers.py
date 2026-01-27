"""
Tests for OCR Providers Module
==============================

Unit tests for the multi-provider OCR abstraction layer.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from ocr_providers import (
    OCRProviderType,
    OCRResult,
    ProviderConfig,
    PaddleOCRConfig,
    DeepSeekOCRConfig,
    OCRProvider,
    PaddleOCRProvider,
    DeepSeekOCRProvider,
    OCRProviderFactory,
    EnsembleOCRProvider,
    OCRProviderError,
    get_default_provider,
    recognize_vin,
)


# =============================================================================
# OCRResult Tests
# =============================================================================

class TestOCRResult:
    """Tests for OCRResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic result creation."""
        result = OCRResult(
            text="SAL1A2A40SA606662",
            confidence=0.95,
            provider="TestProvider"
        )
        assert result.text == "SAL1A2A40SA606662"
        assert result.confidence == 0.95
        assert result.provider == "TestProvider"
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = OCRResult(
            text="SAL1A2A40SA606662",
            confidence=0.95,
            provider="TestProvider",
            metadata={"key": "value"}
        )
        d = result.to_dict()
        assert d["text"] == "SAL1A2A40SA606662"
        assert d["confidence"] == 0.95
        assert d["provider"] == "TestProvider"
        assert d["metadata"]["key"] == "value"
    
    def test_default_values(self):
        """Test default values are initialized."""
        result = OCRResult(text="", confidence=0.0)
        assert result.bounding_boxes == []
        assert result.metadata == {}
        assert result.provider == ""
        assert result.raw_response is None


# =============================================================================
# Config Tests
# =============================================================================

class TestPaddleOCRConfig:
    """Tests for PaddleOCR configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PaddleOCRConfig()
        assert config.lang == "en"
        assert config.use_gpu == True
        assert config.det_db_box_thresh == 0.3
        assert config.rec_thresh == 0.3
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = PaddleOCRConfig(
            lang="ch",
            use_gpu=False,
            det_db_box_thresh=0.5
        )
        assert config.lang == "ch"
        assert config.use_gpu == False
        assert config.det_db_box_thresh == 0.5


class TestDeepSeekOCRConfig:
    """Tests for DeepSeek OCR configuration (local model)."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DeepSeekOCRConfig()
        assert config.model_name == "deepseek-ai/DeepSeek-OCR"
        assert config.use_gpu == True
        assert config.use_flash_attention == True
        assert config.base_size == 1024
        assert config.image_size == 640
        assert config.crop_mode == True  # Gundam preset for VIN
        assert config.max_tokens == 8192  # DeepSeek-OCR supports up to 8192
        assert config.prompt == "<image>\nFree OCR."
        assert config.use_vllm == False
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DeepSeekOCRConfig(
            model_name="custom-model",
            use_gpu=False,
            use_flash_attention=False,
            base_size=512,
            image_size=320,
            max_tokens=256,
            crop_mode=False,
            use_vllm=True
        )
        assert config.model_name == "custom-model"
        assert config.use_gpu == False
        assert config.use_flash_attention == False
        assert config.base_size == 512
        assert config.image_size == 320
        assert config.max_tokens == 256
        assert config.crop_mode == False
        assert config.use_vllm == True
    
    def test_device_auto_detection(self):
        """Test device auto-detection in __post_init__."""
        config = DeepSeekOCRConfig(device=None)
        # Device should be auto-detected to cuda, mps, or cpu
        assert config.device in ["cuda", "mps", "cpu"]


# =============================================================================
# Provider Type Enum Tests
# =============================================================================

class TestOCRProviderType:
    """Tests for OCR provider type enum."""
    
    def test_paddleocr_value(self):
        """Test PaddleOCR enum value."""
        assert OCRProviderType.PADDLEOCR.value == "paddleocr"
    
    def test_deepseek_value(self):
        """Test DeepSeek enum value."""
        assert OCRProviderType.DEEPSEEK.value == "deepseek"
    
    def test_from_string(self):
        """Test creating enum from string."""
        assert OCRProviderType("paddleocr") == OCRProviderType.PADDLEOCR
        assert OCRProviderType("deepseek") == OCRProviderType.DEEPSEEK


# =============================================================================
# PaddleOCR Provider Tests
# =============================================================================

class TestPaddleOCRProvider:
    """Tests for PaddleOCR provider."""
    
    def test_name_property(self):
        """Test provider name."""
        provider = PaddleOCRProvider()
        assert provider.name == "PaddleOCR"
    
    def test_is_available(self):
        """Test availability check."""
        provider = PaddleOCRProvider()
        # Should be True since PaddleOCR is installed in the environment
        assert provider.is_available == True
    
    def test_load_image_from_array(self):
        """Test loading image from numpy array."""
        provider = PaddleOCRProvider()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider._load_image(img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)
    
    def test_load_image_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        provider = PaddleOCRProvider()
        with pytest.raises(ValueError, match="not found"):
            provider._load_image("/nonexistent/path.jpg")
    
    def test_image_to_base64(self):
        """Test base64 encoding."""
        provider = PaddleOCRProvider()
        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        b64 = provider._image_to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 0


# =============================================================================
# DeepSeek Provider Tests
# =============================================================================

class TestDeepSeekOCRProvider:
    """Tests for DeepSeek local model provider."""
    
    def test_name_property(self):
        """Test provider name."""
        provider = DeepSeekOCRProvider()
        assert provider.name == "DeepSeek-OCR"
    
    def test_is_available_checks_dependencies(self):
        """Test availability check depends on torch/transformers."""
        provider = DeepSeekOCRProvider()
        # is_available should check if torch and transformers are installed
        # In test environment, it may be True or False depending on deps
        assert isinstance(provider.is_available, bool)
    
    def test_config_uses_deepseek_ocr_config(self):
        """Test provider uses DeepSeekOCRConfig."""
        config = DeepSeekOCRConfig(model_name="test-model")
        provider = DeepSeekOCRProvider(config=config)
        assert provider.config.model_name == "test-model"
    
    def test_clean_response_extracts_vin(self):
        """Test response cleaning extracts VIN."""
        provider = DeepSeekOCRProvider()
        
        # Test various response formats
        assert provider._clean_response("SAL1A2A40SA606662") == "SAL1A2A40SA606662"
        assert provider._clean_response("  SAL1A2A40SA606662  ") == "SAL1A2A40SA606662"
        assert provider._clean_response("VIN: SAL1A2A40SA606662") == "SAL1A2A40SA606662"
        assert provider._clean_response("`SAL1A2A40SA606662`") == "SAL1A2A40SA606662"
    
    def test_clean_response_extracts_vin_pattern(self):
        """Test response cleaning extracts VIN pattern from text."""
        provider = DeepSeekOCRProvider()
        # The clean_response extracts 17-char alphanumeric patterns
        result = provider._clean_response("Some text SAL1A2A40SA606662 more text")
        assert result == "SAL1A2A40SA606662"
    
    def test_estimate_confidence_valid_vin(self):
        """Test confidence estimation for valid VIN."""
        provider = DeepSeekOCRProvider()
        confidence = provider._estimate_confidence("SAL1A2A40SA606662")
        # Valid 17-char VIN with valid checksum should have high confidence
        assert confidence >= 0.90, f"Expected >= 0.90, got {confidence}"
    
    def test_estimate_confidence_invalid_length(self):
        """Test confidence estimation for invalid length."""
        provider = DeepSeekOCRProvider()
        confidence = provider._estimate_confidence("SHORT")
        # Short text should have low confidence (proportional to length)
        assert confidence < 0.5, f"Expected < 0.5, got {confidence}"
    
    def test_estimate_confidence_empty(self):
        """Test confidence estimation for empty text."""
        provider = DeepSeekOCRProvider()
        assert provider._estimate_confidence("") == 0.0


# =============================================================================
# Provider Factory Tests
# =============================================================================

class TestOCRProviderFactory:
    """Tests for OCR provider factory."""
    
    def test_create_paddleocr(self):
        """Test creating PaddleOCR provider."""
        provider = OCRProviderFactory.create("paddleocr", auto_initialize=False)
        assert isinstance(provider, PaddleOCRProvider)
        assert provider.name == "PaddleOCR"
    
    def test_create_deepseek(self):
        """Test creating DeepSeek provider."""
        provider = OCRProviderFactory.create(
            "deepseek",
            auto_initialize=False,
            model_name="deepseek-ai/DeepSeek-OCR"
        )
        assert isinstance(provider, DeepSeekOCRProvider)
        assert provider.name == "DeepSeek-OCR"
    
    def test_create_from_enum(self):
        """Test creating provider from enum."""
        provider = OCRProviderFactory.create(
            OCRProviderType.PADDLEOCR,
            auto_initialize=False
        )
        assert isinstance(provider, PaddleOCRProvider)
    
    def test_create_unknown_provider_raises(self):
        """Test creating unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            OCRProviderFactory.create("unknown_provider")
    
    def test_list_available(self):
        """Test listing available providers."""
        providers = OCRProviderFactory.list_available()
        assert "paddleocr" in providers
        assert "deepseek" in providers
    
    def test_custom_config_passed(self):
        """Test custom configuration is passed to provider."""
        provider = OCRProviderFactory.create(
            "paddleocr",
            auto_initialize=False,
            lang="ch",
            use_gpu=False
        )
        assert provider.config.lang == "ch"
        assert provider.config.use_gpu == False


# =============================================================================
# Ensemble Provider Tests
# =============================================================================

class TestEnsembleOCRProvider:
    """Tests for ensemble OCR provider."""
    
    def test_creation_with_providers(self):
        """Test creating ensemble with providers."""
        p1 = PaddleOCRProvider()
        ensemble = EnsembleOCRProvider(providers=[p1], strategy="best")
        assert "PaddleOCR" in ensemble.name
    
    def test_empty_providers_raises(self):
        """Test empty providers list raises error."""
        with pytest.raises(ValueError, match="At least one provider"):
            EnsembleOCRProvider(providers=[], strategy="best")
    
    def test_best_strategy(self):
        """Test best confidence strategy."""
        ensemble = EnsembleOCRProvider(
            providers=[PaddleOCRProvider()],
            strategy="best"
        )
        
        # Create mock results
        results = [
            OCRResult(text="VIN1", confidence=0.7, provider="A"),
            OCRResult(text="VIN2", confidence=0.9, provider="B"),
            OCRResult(text="VIN3", confidence=0.5, provider="C"),
        ]
        
        best = ensemble._best_strategy(results)
        assert best.text == "VIN2"
        assert best.confidence == 0.9
    
    def test_vote_strategy(self):
        """Test majority voting strategy."""
        ensemble = EnsembleOCRProvider(
            providers=[PaddleOCRProvider()],
            strategy="vote"
        )
        
        # Create mock results with majority
        results = [
            OCRResult(text="VIN1", confidence=0.7, provider="A"),
            OCRResult(text="VIN1", confidence=0.9, provider="B"),
            OCRResult(text="VIN2", confidence=0.95, provider="C"),
        ]
        
        winner = ensemble._vote_strategy(results)
        assert winner.text == "VIN1"  # Majority wins
        assert winner.confidence == 0.9  # Highest confidence among winners
    
    def test_cascade_strategy_valid_first(self):
        """Test cascade strategy returns first valid."""
        ensemble = EnsembleOCRProvider(
            providers=[PaddleOCRProvider()],
            strategy="cascade"
        )
        
        # First result is valid 17-char VIN
        results = [
            OCRResult(text="SAL1A2A40SA606662", confidence=0.7, provider="A"),
            OCRResult(text="INVALID", confidence=0.9, provider="B"),
        ]
        
        winner = ensemble._cascade_strategy(results)
        assert winner.text == "SAL1A2A40SA606662"
    
    def test_cascade_strategy_fallback_to_best(self):
        """Test cascade strategy falls back to best when no valid VIN."""
        ensemble = EnsembleOCRProvider(
            providers=[PaddleOCRProvider()],
            strategy="cascade"
        )
        
        # No valid VINs
        results = [
            OCRResult(text="SHORT", confidence=0.7, provider="A"),
            OCRResult(text="ALSOSHORT", confidence=0.9, provider="B"),
        ]
        
        winner = ensemble._cascade_strategy(results)
        assert winner.confidence == 0.9  # Falls back to best
    
    def test_is_available_any_provider(self):
        """Test availability check."""
        p1 = PaddleOCRProvider()
        ensemble = EnsembleOCRProvider(providers=[p1])
        assert ensemble.is_available == True


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_default_provider(self):
        """Test getting default provider."""
        provider = get_default_provider()
        assert isinstance(provider, PaddleOCRProvider)
    
    @patch.object(PaddleOCRProvider, 'recognize')
    def test_recognize_vin_default_provider(self, mock_recognize):
        """Test recognize_vin with default provider."""
        mock_recognize.return_value = OCRResult(
            text="SAL1A2A40SA606662",
            confidence=0.9,
            provider="PaddleOCR"
        )
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognize_vin(img)
        
        assert result.text == "SAL1A2A40SA606662"
        mock_recognize.assert_called_once()
    
    @patch.object(DeepSeekOCRProvider, 'recognize')
    @patch.object(DeepSeekOCRProvider, 'initialize')
    def test_recognize_vin_with_provider_string(self, mock_init, mock_recognize):
        """Test recognize_vin with provider string."""
        mock_recognize.return_value = OCRResult(
            text="SAL1A2A40SA606662",
            confidence=0.9,
            provider="DeepSeek Vision"
        )
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognize_vin(img, provider="deepseek", api_key="test-key")
        
        assert result.text == "SAL1A2A40SA606662"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_ocr_provider_error_creation(self):
        """Test OCR provider error creation."""
        error = OCRProviderError(
            message="Test error",
            provider="TestProvider",
            details={"key": "value"}
        )
        assert "TestProvider" in str(error)
        assert "Test error" in str(error)
        assert error.details["key"] == "value"
    
    def test_provider_error_without_details(self):
        """Test error creation without details."""
        error = OCRProviderError(
            message="Test error",
            provider="TestProvider"
        )
        assert error.details == {}


# =============================================================================
# Integration-style Tests (with mocks)
# =============================================================================

class TestProviderIntegration:
    """Integration tests with mocking."""
    
    @patch('paddleocr.PaddleOCR')
    def test_paddleocr_full_flow(self, mock_paddleocr_class):
        """Test full PaddleOCR flow with mock."""
        # Setup mock
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [{
            'rec_texts': ['SAL1A2A40SA606662'],
            'rec_scores': [0.95],
            'dt_polys': [[[0, 0], [100, 0], [100, 30], [0, 30]]]
        }]
        mock_paddleocr_class.return_value = mock_ocr
        
        # Create and initialize provider
        provider = PaddleOCRProvider()
        provider._ocr = mock_ocr
        provider._initialized = True
        
        # Run recognition
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.recognize(img)
        
        assert result.text == "SAL1A2A40SA606662"
        assert result.confidence == 0.95
        assert result.provider == "PaddleOCR"
