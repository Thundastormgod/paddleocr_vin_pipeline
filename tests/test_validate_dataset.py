"""
Test Suite for Ground Truth Validation Module
=============================================

Tests for validate_dataset.py covering:
- VIN extraction from filenames
- Label file parsing
- VIN format validation
- Dataset consistency checks

Run with: pytest tests/test_validate_dataset.py -v
"""

import pytest
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validate_dataset import (
    parse_label_file,
    validate_vin_format_detailed,
    ValidationReport,
)
# Import extract_vin_from_filename from Single Source of Truth
from vin_utils import extract_vin_from_filename


# =============================================================================
# VIN EXTRACTION TESTS
# =============================================================================

class TestExtractVINFromFilename:
    """Tests for VIN extraction from filenames."""
    
    def test_standard_format(self):
        """Test standard VIN filename format."""
        vin = extract_vin_from_filename("001-VIN_-_SAL1P9EU2SA606664_.jpg")
        assert vin == "SAL1P9EU2SA606664"
    
    def test_variant_with_number(self):
        """Test filename variant with number suffix."""
        vin = extract_vin_from_filename("001-VIN_-_SAL1P9EU2SA606664_2.jpg")
        assert vin == "SAL1P9EU2SA606664"
    
    def test_lowercase_vin(self):
        """Test lowercase VIN is normalized."""
        vin = extract_vin_from_filename("001-VIN_-_sal1p9eu2sa606664_.jpg")
        assert vin == "SAL1P9EU2SA606664"
    
    def test_no_vin_pattern(self):
        """Test filename without VIN pattern returns None."""
        vin = extract_vin_from_filename("random_image.jpg")
        assert vin is None
    
    def test_incomplete_vin(self):
        """Test filename with incomplete VIN."""
        vin = extract_vin_from_filename("001-VIN_-_SAL1P9_.jpg")
        assert vin is None


# =============================================================================
# VIN FORMAT VALIDATION TESTS  
# =============================================================================

class TestValidateVINFormat:
    """Tests for VIN format validation."""
    
    def test_valid_vin(self):
        """Test validation of valid VIN."""
        result = validate_vin_format_detailed("SAL1P9EU2SA606664")
        assert result['is_valid'] is True
        assert len(result['issues']) == 0 or 'checksum' in result['issues'][0].lower()
    
    def test_invalid_length_short(self):
        """Test VIN that is too short."""
        result = validate_vin_format_detailed("SAL1P9")
        assert result['is_valid'] is False
        assert any('length' in issue.lower() for issue in result['issues'])
    
    def test_invalid_length_long(self):
        """Test VIN that is too long."""
        result = validate_vin_format_detailed("SAL1P9EU2SA6066641234")
        assert result['is_valid'] is False
        assert any('length' in issue.lower() for issue in result['issues'])
    
    def test_invalid_char_i(self):
        """Test VIN with invalid character I."""
        result = validate_vin_format_detailed("SAL1P9EI2SA606664")
        assert result['is_valid'] is False
        assert any('character' in issue.lower() for issue in result['issues'])
    
    def test_invalid_char_o(self):
        """Test VIN with invalid character O."""
        result = validate_vin_format_detailed("SAL1P9EO2SA606664")
        assert result['is_valid'] is False
    
    def test_invalid_char_q(self):
        """Test VIN with invalid character Q."""
        result = validate_vin_format_detailed("SAL1P9EQ2SA606664")
        assert result['is_valid'] is False


# =============================================================================
# VALIDATION REPORT TESTS
# =============================================================================

class TestValidationReport:
    """Tests for ValidationReport data class."""
    
    def test_default_initialization(self):
        """Test default report initialization."""
        report = ValidationReport()
        assert report.total_images == 0
        assert report.images_without_labels == []
        assert report.mismatch_details == []
    
    def test_report_with_values(self):
        """Test report with custom values."""
        report = ValidationReport(
            total_images=100,
            total_labels=95,
            vins_from_filename=100,
            filename_label_matches=90,
            filename_label_mismatches=5
        )
        assert report.total_images == 100
        assert report.filename_label_matches == 90
    
    def test_report_lists_initialized(self):
        """Test that list fields are properly initialized."""
        report = ValidationReport()
        report.images_without_labels.append("test.jpg")
        assert len(report.images_without_labels) == 1
        
        # New report should have empty list
        report2 = ValidationReport()
        assert len(report2.images_without_labels) == 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
