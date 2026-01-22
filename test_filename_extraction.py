#!/usr/bin/env python3
"""
Test script to verify VIN extraction from filenames.

Usage:
    python test_filename_extraction.py
    python test_filename_extraction.py --data-dir /path/to/images
    python test_filename_extraction.py --filename "42-VIN -SAL1A2A40SA606645.jpg"
"""

import argparse
import re
from pathlib import Path
from typing import Optional


def extract_vin_from_filename(filename: str) -> Optional[str]:
    """
    Extract VIN from filename pattern.
    
    Expected patterns:
    - NUMBER-VIN -VINCODE.jpg (e.g., "42-VIN -SAL1A2A40SA606645.jpg")
    - NUMBER -VIN REST.jpg (legacy)
    - NUMBER-VIN_-_VINCODE_.jpg (legacy)
    """
    # Pattern 1 (NEW): "number-VIN -VINCODE.jpg"
    match = re.search(r'^\d+-VIN\s+-([A-Z0-9]{17})(?:\s|\.)', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: More flexible - "VIN -VINCODE" or "VIN-VINCODE" anywhere
    match = re.search(r'VIN\s*-\s*([A-Z0-9]{17})(?:\s|\.)', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: Flexible spacing around dash (previous format)
    match = re.search(r'^\d+\s*-\s*([A-Z0-9]{17})(?:\s|\.)', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Legacy Pattern: VIN_-_VINCODE_
    match = re.search(r'VIN_-_([A-Z0-9]{17})_', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Legacy Pattern: VIN_-_VINCODE (without trailing underscore)
    match = re.search(r'VIN_-_([A-Z0-9]{17})[._]', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: Look for 17-char alphanumeric sequence
    match = re.search(r'\b([A-HJ-NPR-Z0-9]{17})\b', filename, re.IGNORECASE)
    if match:
        vin = match.group(1).upper()
        if not any(c in vin for c in 'IOQ'):
            return vin
    
    return None


def test_predefined_patterns():
    """Test with predefined filename patterns."""
    print("=" * 60)
    print("Testing Predefined Filename Patterns")
    print("=" * 60)
    
    test_cases = [
        # New format: number-VIN -VINCODE.jpg
        ("42-VIN -SAL1A2A40SA606645.jpg", "SAL1A2A40SA606645"),
        ("1-VIN -WVWZZZ3CZWE123456.jpg", "WVWZZZ3CZWE123456"),
        ("123-VIN -1HGBH41JXMN109186.jpg", "1HGBH41JXMN109186"),
        
        # Previous formats
        ("42 -SAL1A2A40SA606645 2.jpg", "SAL1A2A40SA606645"),
        ("42-SAL1A2A40SA606645 2.jpg", "SAL1A2A40SA606645"),
        ("42 - SAL1A2A40SA606645.jpg", "SAL1A2A40SA606645"),
        
        # Legacy format
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


def test_single_filename(filename: str):
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


def test_directory(data_dir: str):
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
        test_single_filename(args.filename)
    elif args.data_dir:
        test_directory(args.data_dir)
    else:
        # Run predefined tests
        test_predefined_patterns()


if __name__ == "__main__":
    main()
