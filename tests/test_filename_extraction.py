#!/usr/bin/env python3
"""
Test script to verify VIN extraction from filenames.

This module tests the extract_vin_from_filename function from vin_utils.py.

Usage as pytest:
    pytest tests/test_filename_extraction.py -v

Usage as CLI (for manual testing):
    python tests/test_filename_extraction.py
    python tests/test_filename_extraction.py --data-dir /path/to/images
    python tests/test_filename_extraction.py --filename "42-VIN -SAL1A2A40SA606645.jpg"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory for imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from vin_utils import extract_vin_from_filename


# =============================================================================
# PYTEST COMPATIBLE TESTS
# =============================================================================

def test_primary_format():
    """Test primary filename format: NUMBER-VIN -VINCODE.jpg"""
    result = extract_vin_from_filename("1-VIN -SAL1A2A40SA606662.jpg")
    assert result == "SAL1A2A40SA606662"


def test_with_suffix_number():
    """Test format with suffix number."""
    result = extract_vin_from_filename("42-VIN -SAL1A2A40SA606645 2.jpg")
    assert result == "SAL1A2A40SA606645"


def test_underscore_format():
    """Test legacy underscore format."""
    result = extract_vin_from_filename("1-VIN_-_SAL1A2A40SA606662_.jpg")
    assert result == "SAL1A2A40SA606662"


def test_lowercase_normalized():
    """Test that lowercase VINs are normalized to uppercase."""
    result = extract_vin_from_filename("1-VIN -sal1a2a40sa606662.jpg")
    assert result == "SAL1A2A40SA606662"


def test_no_vin_returns_none():
    """Test that invalid filenames return None."""
    result = extract_vin_from_filename("random_image.jpg")
    assert result is None


def test_short_vin_returns_none():
    """Test that partial VINs are not extracted."""
    result = extract_vin_from_filename("1-VIN -SAL1A2A40.jpg")
    assert result is None


# =============================================================================
# CLI FUNCTIONS (for manual testing - prefixed with _ to exclude from pytest)
# =============================================================================

def _cli_test_predefined_patterns():
    """Test with predefined filename patterns."""
    print("=" * 60)
    print("Testing Predefined Filename Patterns")
    print("=" * 60)
    
    test_cases = [
        # Primary format: number-VIN -VINCODE.jpg
        ("1-VIN -SAL1A2A40SA606662.jpg", "SAL1A2A40SA606662"),
        ("42-VIN -SAL1A2A40SA606645.jpg", "SAL1A2A40SA606645"),
        ("1-VIN -WVWZZZ3CZWE123456.jpg", "WVWZZZ3CZWE123456"),
        ("123-VIN -1HGBH41JXMN109186.jpg", "1HGBH41JXMN109186"),
        
        # Legacy formats
        ("42 -SAL1A2A40SA606645 2.jpg", "SAL1A2A40SA606645"),
        ("42-SAL1A2A40SA606645 2.jpg", "SAL1A2A40SA606645"),
        ("42 - SAL1A2A40SA606645.jpg", "SAL1A2A40SA606645"),
        
        # Legacy underscore format
        ("1-VIN_-_SAL119E90SA606112_.jpg", "SAL119E90SA606112"),
        ("10-VIN_-_SAL1A2A40SA606645_.jpg", "SAL1A2A40SA606645"),
    ]
    
    passed = 0
    failed = 0
    
    for filename, expected in test_cases:
        result = extract_vin_from_filename(filename)
        if result == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"\n{status}: \"{filename}\"")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def _cli_test_single_filename(filename: str):
    """Test extraction on a single filename."""
    print("=" * 60)
    print("Testing Single Filename")
    print("=" * 60)
    
    print(f"\nFilename: \"{filename}\"")
    result = extract_vin_from_filename(filename)
    
    if result:
        print(f"✓ Extracted VIN: {result}")
        print(f"  VIN Length: {len(result)} characters")
    else:
        print("✗ Could not extract VIN from filename")
        print("\nExpected formats:")
        print("  - number-VIN -VINCODE.jpg  (e.g., 42-VIN -SAL1A2A40SA606645.jpg)")
        print("  - number -VINCODE rest.jpg (e.g., 42 -SAL1A2A40SA606645 2.jpg)")
        print("  - number-VIN_-_VINCODE_.jpg (legacy)")
    
    return result


def _cli_test_directory(data_dir: str):
    """Test extraction on all images in a directory."""
    print("=" * 60)
    print(f"Testing Images in: {data_dir}")
    print("=" * 60)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"✗ Error: Directory not found: {data_dir}")
        return
    
    extracted = 0
    failed = 0
    failed_files = []
    
    for img_file in sorted(data_path.iterdir()):
        if img_file.suffix.lower() in image_extensions:
            vin = extract_vin_from_filename(img_file.name)
            if vin:
                extracted += 1
                print(f"✓ {img_file.name} -> {vin}")
            else:
                failed += 1
                failed_files.append(img_file.name)
                print(f"✗ {img_file.name} -> FAILED")
    
    print("\n" + "=" * 60)
    print(f"Results: {extracted} extracted, {failed} failed")
    print("=" * 60)
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Test VIN extraction from image filenames"
    )
    parser.add_argument(
        "--data-dir", "-d",
        help="Directory containing images to test"
    )
    parser.add_argument(
        "--filename", "-f",
        help="Single filename to test"
    )
    
    args = parser.parse_args()
    
    if args.filename:
        _cli_test_single_filename(args.filename)
    elif args.data_dir:
        _cli_test_directory(args.data_dir)
    else:
        # Run predefined tests
        _cli_test_predefined_patterns()


if __name__ == "__main__":
    main()
