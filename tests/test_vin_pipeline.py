"""
Test Suite for VIN OCR Pipeline
================================

Comprehensive tests covering:
- Unit tests for each component
- Integration tests for the full pipeline
- Edge case and adversarial testing
- Property-based validation testing

Run with: pytest tests/test_vin_pipeline.py -v
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vin_pipeline import (
    VINOCRPipeline,
    VINImagePreprocessor,
    VINPostProcessor,
    VINResult,
    validate_vin,
    decode_vin,
    calculate_check_digit,
    VIN_LENGTH,
    VIN_VALID_CHARS,
    PreprocessMode,
    ConfigurationError,
    ImageLoadError,
    CLAHEConfig,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def preprocessor():
    """Create a default preprocessor instance."""
    return VINImagePreprocessor(mode='engraved')


@pytest.fixture
def postprocessor():
    """Create a default postprocessor instance."""
    return VINPostProcessor(verbose=False)


@pytest.fixture
def sample_image():
    """Create a sample grayscale image for testing."""
    # Create a 100x400 grayscale image (typical VIN plate aspect ratio)
    img = np.random.randint(100, 200, (100, 400), dtype=np.uint8)
    return img


@pytest.fixture
def sample_bgr_image():
    """Create a sample BGR color image for testing."""
    img = np.random.randint(100, 200, (100, 400, 3), dtype=np.uint8)
    return img


# =============================================================================
# VIN VALIDATION TESTS
# =============================================================================

class TestValidateVIN:
    """Tests for validate_vin function."""
    
    def test_valid_vin(self):
        """Test validation of a structurally valid VIN."""
        # Note: This VIN may not have valid checksum
        result = validate_vin("WVWZZZ3CZWE123456")
        assert result['is_valid_length'] is True
        assert result['has_valid_chars'] is True
        assert result['invalid_chars'] == []
    
    def test_invalid_length_short(self):
        """Test VIN that is too short."""
        result = validate_vin("ABC123")
        assert result['is_valid_length'] is False
        assert result['is_fully_valid'] is False
    
    def test_invalid_length_long(self):
        """Test VIN that is too long."""
        result = validate_vin("WVWZZZ3CZWE123456789")
        assert result['is_valid_length'] is False
        assert result['is_fully_valid'] is False
    
    def test_invalid_characters(self):
        """Test VIN with invalid characters (I, O, Q)."""
        result = validate_vin("WVWZZZ3CZWI123456")  # Contains 'I'
        assert 'I' in result['invalid_chars']
        assert result['has_valid_chars'] is False
    
    def test_lowercase_normalized(self):
        """Test that lowercase VINs are normalized to uppercase."""
        result = validate_vin("wvwzzz3czwe123456")
        assert result['vin'] == "WVWZZZ3CZWE123456"
    
    def test_whitespace_trimmed(self):
        """Test that whitespace is trimmed."""
        result = validate_vin("  WVWZZZ3CZWE123456  ")
        assert result['vin'] == "WVWZZZ3CZWE123456"
    
    def test_non_string_input(self):
        """Test handling of non-string input."""
        result = validate_vin(12345)  # type: ignore
        assert 'error' in result
        assert result['is_fully_valid'] is False
    
    def test_empty_string(self):
        """Test handling of empty string."""
        result = validate_vin("")
        assert result['is_valid_length'] is False
        assert result['is_fully_valid'] is False


class TestDecodeVIN:
    """Tests for decode_vin function."""
    
    def test_decode_valid_vin(self):
        """Test decoding of a valid VIN."""
        result = decode_vin("SAL1P9EU2SA606664")
        assert result['wmi'] == "SAL"
        assert result['vds'] == "1P9EU2"
        assert result['check_digit'] == "2"
        assert result['sequential'] == "606664"
        assert result['plant_code'] == "A"
    
    def test_decode_invalid_length(self):
        """Test decoding of invalid length VIN."""
        result = decode_vin("SHORT")
        assert 'error' in result
    
    def test_model_year_decoding(self):
        """Test model year code interpretation."""
        result = decode_vin("WVWZZZ3CZSE123456")  # S = 1995 or 2025
        assert result['model_year_code'] == "S"
        # Model year should be decoded (modern interpretation: 2025)
        assert result['model_year'] == 2025
        # Display should show ambiguity
        assert 'model_year_display' in result
        assert "2025" in result['model_year_display']
        assert "1995" in result['model_year_display']  # Shows both options
    
    def test_model_year_digit_unambiguous(self):
        """Test that digit year codes (2001-2009) are unambiguous."""
        result = decode_vin("WVWZZZ3CZ5E123456")  # 5 = 2005
        assert result['model_year_code'] == "5"
        assert result['model_year'] == 2005
        assert result['model_year_display'] == "2005"  # No ambiguity


class TestCalculateCheckDigit:
    """Tests for check digit calculation."""
    
    def test_check_digit_calculation(self):
        """Test that check digit calculation is correct."""
        # For a known valid VIN, verify the check digit
        vin = "11111111111111111"  # All 1s
        check = calculate_check_digit(vin)
        assert check in '0123456789X'
    
    def test_invalid_character_raises(self):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError):
            calculate_check_digit("VINWITHI1234567")  # Contains I
    
    def test_short_vin_raises(self):
        """Test that short VIN raises ValueError."""
        with pytest.raises(ValueError):
            calculate_check_digit("SHORT")


# =============================================================================
# PREPROCESSOR TESTS
# =============================================================================

class TestVINImagePreprocessor:
    """Tests for VINImagePreprocessor class."""
    
    def test_mode_none_passthrough(self, sample_image):
        """Test that 'none' mode returns image unchanged."""
        preprocessor = VINImagePreprocessor(mode='none')
        result = preprocessor.preprocess(sample_image)
        np.testing.assert_array_equal(result, sample_image)
    
    def test_mode_fast_returns_grayscale(self, sample_bgr_image):
        """Test that 'fast' mode returns normalized grayscale."""
        preprocessor = VINImagePreprocessor(mode='fast')
        result = preprocessor.preprocess(sample_bgr_image)
        assert len(result.shape) == 2  # Should be grayscale
    
    def test_mode_balanced_applies_clahe(self, sample_image):
        """Test that 'balanced' mode applies CLAHE."""
        preprocessor = VINImagePreprocessor(mode='balanced')
        result = preprocessor.preprocess(sample_image)
        # Result should be different due to CLAHE
        assert not np.array_equal(result, sample_image)
    
    def test_mode_engraved_applies_bilateral(self, sample_image):
        """Test that 'engraved' mode applies bilateral filter."""
        preprocessor = VINImagePreprocessor(mode='engraved')
        result = preprocessor.preprocess(sample_image)
        assert result.shape == sample_image.shape
    
    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            VINImagePreprocessor(mode='invalid_mode')
    
    def test_enum_mode_accepted(self):
        """Test that PreprocessMode enum is accepted."""
        preprocessor = VINImagePreprocessor(mode=PreprocessMode.ENGRAVED)
        assert preprocessor.mode == PreprocessMode.ENGRAVED
    
    def test_large_image_resized(self):
        """Test that very large images are resized to prevent OOM."""
        large_image = np.zeros((5000, 5000), dtype=np.uint8)
        preprocessor = VINImagePreprocessor(mode='fast')
        result = preprocessor.preprocess(large_image)
        assert max(result.shape) <= preprocessor.MAX_DIMENSION
    
    def test_empty_image_raises(self):
        """Test that empty image raises ValueError."""
        preprocessor = VINImagePreprocessor(mode='fast')
        with pytest.raises(ValueError):
            preprocessor.preprocess(np.array([]))
    
    def test_none_image_raises(self):
        """Test that None image raises ValueError."""
        preprocessor = VINImagePreprocessor(mode='fast')
        with pytest.raises(ValueError):
            preprocessor.preprocess(None)
    
    def test_custom_clahe_config(self, sample_image):
        """Test that custom CLAHE config is applied."""
        config = CLAHEConfig(clip_limit=4.0, tile_size=(16, 16))
        preprocessor = VINImagePreprocessor(mode='balanced', clahe_config=config)
        result = preprocessor.preprocess(sample_image)
        assert result is not None


# =============================================================================
# POSTPROCESSOR TESTS
# =============================================================================

class TestVINPostProcessor:
    """Tests for VINPostProcessor class."""
    
    def test_artifact_removal_prefix(self, postprocessor):
        """Test removal of prefix artifacts."""
        result = postprocessor.process("*SAL1P9EU2SA606664", 0.9)
        assert not result['vin'].startswith('*')
    
    def test_artifact_removal_suffix(self, postprocessor):
        """Test removal of suffix artifacts."""
        result = postprocessor.process("SAL1P9EU2SA606664#*", 0.9)
        assert not result['vin'].endswith('#')
        assert not result['vin'].endswith('*')
    
    def test_invalid_char_i_to_1(self, postprocessor):
        """Test that I is converted to 1."""
        result = postprocessor.process("SAL1P9EU2SA60666I", 0.9)
        assert 'I' not in result['vin']
    
    def test_invalid_char_o_to_0(self, postprocessor):
        """Test that O is converted to 0."""
        result = postprocessor.process("SAL1P9EU2SA6O6664", 0.9)
        assert 'O' not in result['vin']
    
    def test_invalid_char_q_to_0(self, postprocessor):
        """Test that Q is converted to 0."""
        result = postprocessor.process("SAL1P9EU2SA6Q6664", 0.9)
        assert 'Q' not in result['vin']
    
    def test_position_based_correction_sequential(self, postprocessor):
        """Test that letters in sequential section are converted to digits."""
        # Position 12-17 should prefer digits
        result = postprocessor.process("SAL1P9EU2SA60S664", 0.9)
        # S in position 14 should become 5
        assert result['vin'][13] == '5'  # 0-indexed position 13
    
    def test_lowercase_normalized(self, postprocessor):
        """Test that lowercase is normalized to uppercase."""
        result = postprocessor.process("sal1p9eu2sa606664", 0.9)
        assert result['vin'] == result['vin'].upper()
    
    def test_checksum_validation(self, postprocessor):
        """Test that checksum validation is performed."""
        result = postprocessor.process("SAL1P9EU2SA606664", 0.9)
        assert 'checksum_valid' in result
        assert isinstance(result['checksum_valid'], bool)
    
    def test_corrections_tracked(self, postprocessor):
        """Test that corrections are tracked in output."""
        result = postprocessor.process("*ISALOPEQ2SA606664", 0.9)
        assert 'corrections' in result
        assert len(result['corrections']) > 0
    
    def test_empty_input(self, postprocessor):
        """Test handling of empty input."""
        result = postprocessor.process("", 0.0)
        assert result['vin'] == ""
        assert result['is_valid_length'] is False


# =============================================================================
# PIPELINE INTEGRATION TESTS
# =============================================================================

class TestVINOCRPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath('vin_pipeline.py').exists(),
        reason="Pipeline module not found"
    )
    def test_pipeline_initialization(self):
        """Test that pipeline initializes without error."""
        try:
            pipeline = VINOCRPipeline(preprocess_mode='none')
            assert pipeline is not None
        except ConfigurationError:
            pytest.skip("PaddleOCR not installed")
    
    def test_invalid_image_path_error(self):
        """Test that invalid image path returns error."""
        try:
            pipeline = VINOCRPipeline(preprocess_mode='none')
            result = pipeline.recognize("/nonexistent/path/image.jpg")
            assert result.get('error') is not None
        except ConfigurationError:
            pytest.skip("PaddleOCR not installed")
    
    def test_numpy_array_input(self):
        """Test that numpy array input is accepted."""
        try:
            pipeline = VINOCRPipeline(preprocess_mode='none')
            image = np.random.randint(0, 255, (100, 400, 3), dtype=np.uint8)
            result = pipeline.recognize(image)
            assert 'vin' in result
        except ConfigurationError:
            pytest.skip("PaddleOCR not installed")
    
    def test_batch_processing_continue_on_error(self):
        """Test batch processing continues on individual errors."""
        try:
            pipeline = VINOCRPipeline(preprocess_mode='none')
            paths = ["/nonexistent1.jpg", "/nonexistent2.jpg"]
            results = pipeline.recognize_batch(
                paths,
                show_progress=False,
                continue_on_error=True
            )
            assert len(results) == 2
            assert all('error' in r or r.get('error') for r in results)
        except ConfigurationError:
            pytest.skip("PaddleOCR not installed")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case and adversarial tests."""
    
    def test_all_valid_characters(self):
        """Test VIN with all valid character types."""
        # Contains digits and all valid letters
        result = validate_vin("ABCDEFGH123JKLMNP")
        assert result['has_valid_chars'] is True
    
    def test_special_characters_rejected(self):
        """Test that special characters are not in valid set."""
        for char in "!@#$%^&*()-_=+[]{}|;':\",./<>?`~":
            assert char not in VIN_VALID_CHARS
    
    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        result = validate_vin("ЅΑL1P9ЕU2ЅА606664")  # Mixed Cyrillic
        # Should detect invalid characters
        assert result['has_valid_chars'] is False or len(result['invalid_chars']) > 0
    
    def test_whitespace_only(self):
        """Test handling of whitespace-only input."""
        result = validate_vin("   \t\n   ")
        assert result['vin'] == ""
        assert result['is_valid_length'] is False
    
    def test_vin_with_newlines(self, postprocessor):
        """Test that newlines are removed."""
        result = postprocessor.process("SAL1P9EU\n2SA606664", 0.9)
        assert '\n' not in result['vin']
    
    def test_check_digit_x(self):
        """Test that X is valid as check digit."""
        # Position 9 can be X when remainder is 10
        result = validate_vin("WVWZZZ3CZXE123456")
        # X should be valid in position 9
        assert 'X' not in result['invalid_chars']


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================

class TestPropertyBased:
    """Property-based tests for invariants."""
    
    def test_validate_always_returns_dict(self):
        """Test that validate_vin always returns a dict."""
        test_inputs = [
            "", "A", "AB", "ABC", "ABCD", "ABCDE" * 10,
            "12345678901234567", "WVWZZZ3CZWE123456",
            None, 123, [], {}
        ]
        for inp in test_inputs:
            result = validate_vin(inp)  # type: ignore
            assert isinstance(result, dict)
    
    def test_decode_always_returns_dict(self):
        """Test that decode_vin always returns a dict."""
        test_inputs = ["", "SHORT", "12345678901234567", "WVWZZZ3CZWE123456"]
        for inp in test_inputs:
            result = decode_vin(inp)
            assert isinstance(result, dict)
    
    def test_valid_vin_has_17_chars(self):
        """Test that fully valid VINs always have 17 characters."""
        # Generate some VIN-like strings
        for _ in range(10):
            chars = list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
            vin = ''.join(np.random.choice(chars) for _ in range(17))
            result = validate_vin(vin)
            if result['is_fully_valid']:
                assert len(result['vin']) == 17


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Tests for configuration classes."""
    
    def test_clahe_config_defaults(self):
        """Test CLAHEConfig default values."""
        config = CLAHEConfig()
        assert config.clip_limit == 2.0
        assert config.tile_size == (8, 8)
    
    def test_clahe_config_custom(self):
        """Test CLAHEConfig with custom values."""
        config = CLAHEConfig(clip_limit=4.0, tile_size=(16, 16))
        assert config.clip_limit == 4.0
        assert config.tile_size == (16, 16)
    
    def test_preprocess_mode_enum_values(self):
        """Test that all PreprocessMode values are strings."""
        for mode in PreprocessMode:
            assert isinstance(mode.value, str)
    
    def test_vin_length_constant(self):
        """Test that VIN_LENGTH is 17."""
        assert VIN_LENGTH == 17


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
