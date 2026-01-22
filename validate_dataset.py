#!/usr/bin/env python3
"""
Ground Truth Validation Script
==============================

Validates ground truth data quality by:
1. Checking filename VINs match label files
2. Verifying VIN format validity
3. Detecting duplicate VINs
4. Generating data quality report

Usage:
    python validate_dataset.py --data-dir /path/to/paddleocr_sample
    python validate_dataset.py --data-dir /path/to/images --labels-dir /path/to/labels

Author: JRL-VIN Project
Date: January 2026
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from vin_pipeline import validate_vin, VIN_LENGTH


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationReport:
    """Dataset validation report."""
    total_images: int = 0
    total_labels: int = 0
    
    # Ground truth extraction
    vins_from_filename: int = 0
    vins_from_labels: int = 0
    filename_extraction_failed: int = 0
    
    # VIN validity
    valid_vins: int = 0
    invalid_length: int = 0
    invalid_chars: int = 0
    invalid_checksum: int = 0
    
    # Consistency
    filename_label_matches: int = 0
    filename_label_mismatches: int = 0
    
    # Duplicates
    unique_vins: int = 0
    duplicate_vins: int = 0
    
    # Files
    images_without_labels: List[str] = None
    labels_without_images: List[str] = None
    mismatch_details: List[dict] = None
    invalid_vin_details: List[dict] = None
    
    def __post_init__(self):
        if self.images_without_labels is None:
            self.images_without_labels = []
        if self.labels_without_images is None:
            self.labels_without_images = []
        if self.mismatch_details is None:
            self.mismatch_details = []
        if self.invalid_vin_details is None:
            self.invalid_vin_details = []


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def extract_vin_from_filename(filename: str) -> Optional[str]:
    """
    Extract VIN from filename pattern.
    
    Primary format:
    - NUMBER-VIN -VINCODE.jpg (e.g., "1-VIN -SAL1A2A40SA606662.jpg")
    
    Legacy formats (still supported):
    - NUMBER-VIN_-_VINCODE_.jpg
    - NUMBER -VINCODE rest.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        Extracted VIN (17 characters) or None
    """
    # Primary Pattern: "number-VIN -VINCODE.jpg"
    # Example: "1-VIN -SAL1A2A40SA606662.jpg"
    match = re.search(r'^\d+-VIN\s+-([A-Z0-9]{17})\.', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Flexible: "VIN -VINCODE" or "VIN-VINCODE" anywhere in filename
    match = re.search(r'VIN\s*-\s*([A-Z0-9]{17})(?:\s|\.)', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Legacy: "number -VINCODE rest.jpg" or "number-VINCODE.jpg"
    match = re.search(r'^\d+\s*-\s*([A-Z0-9]{17})(?:\s|\.)', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Legacy: VIN_-_VINCODE_
    match = re.search(r'VIN_-_([A-Z0-9]{17})_', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Legacy: VIN_-_VINCODE (without trailing underscore)
    match = re.search(r'VIN_-_([A-Z0-9]{17})[._]', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


def parse_label_file(filepath: str) -> Optional[str]:
    """
    Parse label file and extract VIN.
    
    Handles:
    - Plain text format: '*SAL1A2A92SA606139*'
    - YOLO format: class_id x y w h per line
    """
    # Character class mapping for YOLO format
    class_map = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
        18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'R', 25: 'S',
        26: 'T', 27: 'U', 28: 'V', 29: 'W', 30: 'X', 31: 'Y', 32: 'Z', 33: '*'
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        
        # Check if plain text VIN format
        if content.startswith('*') or ('SAL' in content and len(content) < 30):
            return content.replace('*', '').strip()
        
        # Try YOLO format
        chars = []
        for line in content.split('\n'):
            line = line.strip().rstrip('%')
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    class_id = int(parts[0])
                    x_pos = float(parts[1])
                    char = class_map.get(class_id, '?')
                    chars.append((x_pos, char))
                except (ValueError, IndexError):
                    continue
        
        if chars:
            chars.sort(key=lambda x: x[0])
            return ''.join([c[1] for c in chars]).replace('*', '')
        
        return None
        
    except Exception as e:
        return None


def validate_vin_format(vin: str) -> Dict:
    """Validate VIN format and structure."""
    result = {
        'vin': vin,
        'is_valid': True,
        'issues': []
    }
    
    # Length check
    if len(vin) != VIN_LENGTH:
        result['is_valid'] = False
        result['issues'].append(f"Invalid length: {len(vin)} (expected {VIN_LENGTH})")
    
    # Character check
    invalid_chars = set()
    for c in vin.upper():
        if c in 'IOQ':
            invalid_chars.add(c)
        elif c not in '0123456789ABCDEFGHJKLMNPRSTUVWXYZ':
            invalid_chars.add(c)
    
    if invalid_chars:
        result['is_valid'] = False
        result['issues'].append(f"Invalid characters: {invalid_chars}")
    
    # Use pipeline validation for checksum
    if len(vin) == VIN_LENGTH and not invalid_chars:
        validation = validate_vin(vin)
        if not validation.get('checksum_valid', False):
            result['issues'].append("Invalid checksum")
            # Note: Checksum issues don't necessarily mean wrong ground truth
    
    return result


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def validate_dataset(
    data_dir: str,
    labels_dir: Optional[str] = None,
    verbose: bool = False
) -> ValidationReport:
    """
    Validate dataset ground truth.
    
    Args:
        data_dir: Directory containing images (and optionally labels)
        labels_dir: Separate labels directory (if not in data_dir)
        verbose: Print detailed output
        
    Returns:
        ValidationReport with findings
    """
    report = ValidationReport()
    
    data_path = Path(data_dir)
    
    # Determine labels location
    if labels_dir:
        labels_path = Path(labels_dir)
    elif (data_path / 'labels').exists():
        labels_path = data_path / 'labels'
    else:
        labels_path = None
    
    # Find images directory
    if (data_path / 'images').exists():
        images_path = data_path / 'images'
    else:
        images_path = data_path
    
    # Collect images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = {
        f.stem: f for f in images_path.iterdir() 
        if f.suffix.lower() in image_extensions
    }
    report.total_images = len(images)
    
    # Collect labels
    labels = {}
    if labels_path and labels_path.exists():
        labels = {
            f.stem: f for f in labels_path.iterdir()
            if f.suffix == '.txt'
        }
        report.total_labels = len(labels)
    
    print(f"Found {report.total_images} images, {report.total_labels} label files")
    
    # Track VINs and issues
    filename_vins = {}  # stem -> VIN from filename
    label_vins = {}     # stem -> VIN from label file
    all_vins = []
    
    # Process each image
    for stem, img_path in images.items():
        # Extract VIN from filename
        vin_from_filename = extract_vin_from_filename(img_path.name)
        if vin_from_filename:
            filename_vins[stem] = vin_from_filename
            report.vins_from_filename += 1
            all_vins.append(vin_from_filename)
        else:
            report.filename_extraction_failed += 1
            if verbose:
                print(f"  Warning: Could not extract VIN from filename: {img_path.name}")
        
        # Extract VIN from label file (if exists)
        if stem in labels:
            vin_from_label = parse_label_file(str(labels[stem]))
            if vin_from_label:
                label_vins[stem] = vin_from_label
                report.vins_from_labels += 1
    
    # Check for missing files
    image_stems = set(images.keys())
    label_stems = set(labels.keys())
    
    report.images_without_labels = [
        images[s].name for s in image_stems - label_stems
    ][:20]  # Limit to 20 examples
    
    report.labels_without_images = [
        labels[s].name for s in label_stems - image_stems
    ][:20]
    
    # Compare filename vs label VINs
    common_stems = set(filename_vins.keys()) & set(label_vins.keys())
    for stem in common_stems:
        fn_vin = filename_vins[stem]
        lbl_vin = label_vins[stem]
        
        if fn_vin == lbl_vin:
            report.filename_label_matches += 1
        else:
            report.filename_label_mismatches += 1
            if len(report.mismatch_details) < 20:
                report.mismatch_details.append({
                    'file': stem,
                    'filename_vin': fn_vin,
                    'label_vin': lbl_vin
                })
    
    # Validate VIN formats
    for vin in set(all_vins):
        validation = validate_vin_format(vin)
        if validation['is_valid']:
            report.valid_vins += 1
        else:
            if 'Invalid length' in str(validation['issues']):
                report.invalid_length += 1
            if 'Invalid characters' in str(validation['issues']):
                report.invalid_chars += 1
            if 'Invalid checksum' in str(validation['issues']):
                report.invalid_checksum += 1
            
            if len(report.invalid_vin_details) < 20:
                report.invalid_vin_details.append(validation)
    
    # Check for duplicates
    vin_counts = Counter(all_vins)
    report.unique_vins = len(vin_counts)
    report.duplicate_vins = sum(1 for count in vin_counts.values() if count > 1)
    
    return report


def print_report(report: ValidationReport):
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print("GROUND TRUTH VALIDATION REPORT")
    print("=" * 70)
    
    print("\n--- DATASET SIZE ---")
    print(f"Total Images:        {report.total_images}")
    print(f"Total Label Files:   {report.total_labels}")
    
    print("\n--- GROUND TRUTH EXTRACTION ---")
    print(f"VINs from Filenames: {report.vins_from_filename}")
    print(f"VINs from Labels:    {report.vins_from_labels}")
    print(f"Filename Extraction Failed: {report.filename_extraction_failed}")
    
    print("\n--- VIN VALIDITY ---")
    print(f"Valid VINs:          {report.valid_vins}")
    print(f"Invalid Length:      {report.invalid_length}")
    print(f"Invalid Characters:  {report.invalid_chars}")
    print(f"Invalid Checksum:    {report.invalid_checksum}")
    
    print("\n--- FILENAME vs LABEL CONSISTENCY ---")
    total_compared = report.filename_label_matches + report.filename_label_mismatches
    if total_compared > 0:
        match_rate = report.filename_label_matches / total_compared * 100
        print(f"Matching:            {report.filename_label_matches} ({match_rate:.1f}%)")
        print(f"Mismatching:         {report.filename_label_mismatches} ({100-match_rate:.1f}%)")
    else:
        print("No labels to compare")
    
    print("\n--- DUPLICATES ---")
    print(f"Unique VINs:         {report.unique_vins}")
    print(f"Duplicate VINs:      {report.duplicate_vins}")
    
    # Issues
    if report.mismatch_details:
        print("\n--- MISMATCH EXAMPLES (first 10) ---")
        for m in report.mismatch_details[:10]:
            print(f"  {m['file']}")
            print(f"    Filename: {m['filename_vin']}")
            print(f"    Label:    {m['label_vin']}")
    
    if report.invalid_vin_details:
        print("\n--- INVALID VIN EXAMPLES (first 10) ---")
        for v in report.invalid_vin_details[:10]:
            print(f"  {v['vin']}: {', '.join(v['issues'])}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    issues = []
    if report.filename_extraction_failed > 0:
        issues.append(f"{report.filename_extraction_failed} images without extractable VIN")
    if report.filename_label_mismatches > 0:
        issues.append(f"{report.filename_label_mismatches} filename/label mismatches")
    if report.invalid_length + report.invalid_chars > 0:
        issues.append(f"{report.invalid_length + report.invalid_chars} invalid VINs")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nRECOMMENDATION:")
        if report.filename_label_mismatches > report.filename_label_matches:
            print("  Label files appear to be from a different dataset.")
            print("  Use FILENAMES as ground truth, not label files.")
    else:
        print("No major issues found. Dataset is ready for use.")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate VIN dataset ground truth"
    )
    parser.add_argument(
        '--data-dir', '-d',
        required=True,
        help='Dataset directory (containing images/ and optionally labels/)'
    )
    parser.add_argument(
        '--labels-dir', '-l',
        help='Separate labels directory'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file for report'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    print(f"Validating dataset: {args.data_dir}")
    
    report = validate_dataset(
        args.data_dir,
        labels_dir=args.labels_dir,
        verbose=args.verbose
    )
    
    print_report(report)
    
    if args.output:
        # Convert to dict for JSON
        report_dict = asdict(report)
        with open(args.output, 'w') as f:
            json.dump(report_dict, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    # Return exit code based on issues
    if report.filename_label_mismatches > report.filename_label_matches:
        return 1  # Major issue
    return 0


if __name__ == '__main__':
    sys.exit(main())
