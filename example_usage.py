"""
Example usage of the VIN OCR Pipeline.
"""

from vin_pipeline import VINOCRPipeline, validate_vin, decode_vin
from pathlib import Path


def example_single_image():
    """Process a single VIN image."""
    print("=" * 60)
    print("Example 1: Single Image Recognition")
    print("=" * 60)
    
    # Create pipeline with default settings
    pipeline = VINOCRPipeline(
        preprocess_mode='engraved',
        enable_postprocess=True,
        verbose=True
    )
    
    # Process image
    image_path = 'path/to/your/vin_image.jpg'
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"Note: Replace '{image_path}' with actual image path")
        print("Demo output would be:")
        print("  VIN: SAL1P9EU2SA606664")
        print("  Confidence: 91.5%")
        return
    
    result = pipeline.recognize(image_path)
    
    print(f"\nResult:")
    print(f"  VIN: {result['vin']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Valid: {result.get('is_valid_length')} (length)")
    print(f"  Checksum: {result.get('checksum_valid')}")


def example_batch_processing():
    """Process multiple VIN images."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    
    # Create pipeline
    pipeline = VINOCRPipeline()
    
    # List of images
    images = [
        'image1.jpg',
        'image2.jpg', 
        'image3.jpg'
    ]
    
    # Check if any exist
    existing = [p for p in images if Path(p).exists()]
    
    if not existing:
        print("Note: Replace image paths with actual files")
        print("Demo output would be:")
        print("  Processing 1/3: image1.jpg")
        print("  Processing 2/3: image2.jpg")
        print("  Processing 3/3: image3.jpg")
        return
    
    results = pipeline.recognize_batch(existing)
    
    print("\nResults:")
    for r in results:
        status = "✓" if r.get('is_valid_length') else "✗"
        print(f"  {status} {Path(r['file']).name}: {r['vin']}")


def example_validation():
    """Validate VIN strings."""
    print("\n" + "=" * 60)
    print("Example 3: VIN Validation")
    print("=" * 60)
    
    test_vins = [
        "SAL1P9EU2SA606664",  # Valid
        "1HGBH41JXMN109186",  # Valid
        "WVWZZZ3CZWE123456",  # Example
        "INVALID123",         # Too short
        "11111111111111111",  # All 1s (checksum may fail)
    ]
    
    for vin in test_vins:
        result = validate_vin(vin)
        status = "✓" if result['is_fully_valid'] else "✗"
        print(f"  {status} {vin}")
        if not result['is_fully_valid']:
            if not result['is_valid_length']:
                print(f"      - Invalid length: {len(vin)}")
            if result['invalid_chars']:
                print(f"      - Invalid chars: {result['invalid_chars']}")
            if not result['checksum_valid'] and result['is_valid_length']:
                print(f"      - Checksum invalid")


def example_decode():
    """Decode VIN structure."""
    print("\n" + "=" * 60)
    print("Example 4: VIN Decoding")
    print("=" * 60)
    
    vin = "SAL1P9EU2SA606664"
    decoded = decode_vin(vin)
    
    print(f"  VIN: {decoded['vin']}")
    print(f"  WMI (Manufacturer): {decoded['wmi']}")
    print(f"  VDS (Descriptor): {decoded['vds']}")
    print(f"  Check Digit: {decoded['check_digit']}")
    print(f"  Model Year: {decoded['model_year_display']}")
    print(f"  Plant Code: {decoded['plant_code']}")
    print(f"  Sequential: {decoded['sequential']}")


def example_custom_config():
    """Use custom pipeline configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)
    
    # Minimal preprocessing (fastest)
    print("\nMode: 'none' (no preprocessing)")
    pipeline_fast = VINOCRPipeline(
        preprocess_mode='none',
        enable_postprocess=True
    )
    print("  - Fastest processing")
    print("  - Good for clean, high-quality images")
    
    # Balanced preprocessing
    print("\nMode: 'balanced'")
    pipeline_balanced = VINOCRPipeline(
        preprocess_mode='balanced',
        enable_postprocess=True
    )
    print("  - CLAHE contrast enhancement")
    print("  - Good for most images")
    
    # Full preprocessing (for engraved plates)
    print("\nMode: 'engraved' (recommended for metal plates)")
    pipeline_engraved = VINOCRPipeline(
        preprocess_mode='engraved',
        enable_postprocess=True
    )
    print("  - CLAHE + bilateral filtering")
    print("  - Best for engraved metal plates")
    
    # No postprocessing
    print("\nPostprocessing disabled:")
    pipeline_raw = VINOCRPipeline(
        preprocess_mode='engraved',
        enable_postprocess=False
    )
    print("  - Returns raw OCR output")
    print("  - Useful for debugging")


if __name__ == '__main__':
    print("VIN OCR Pipeline - Usage Examples")
    print("=" * 60)
    
    example_single_image()
    example_batch_processing()
    example_validation()
    example_decode()
    example_custom_config()
    
    print("\n" + "=" * 60)
    print("For command-line usage:")
    print("  python vin_pipeline.py path/to/image.jpg --verbose")
    print("=" * 60)
