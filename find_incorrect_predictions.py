#!/usr/bin/env python3
"""
Find the incorrect predictions that are missing from the sample output.
"""

import json
import sys
from pathlib import Path

def find_incorrect_predictions() -> int:
    """Find and display the incorrect predictions.

    Returns:
        0 on success, 1 when no metrics file or no sample results exist.
    """
    
    # Look for the most recent evaluation metrics file
    output_dir = Path("output/vin_rec_finetune")
    metrics_files = list(output_dir.glob("*evaluation_metrics.json"))
    
    if not metrics_files:
        print("❌ No evaluation metrics file found!")
        return 1
    
    # Use the most recent metrics file
    latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Loading metrics from: {latest_metrics}")
    
    with open(latest_metrics) as f:
        metrics = json.load(f)
    
    print("\n🔍 ANALYZING PREDICTIONS:")
    print("=" * 50)
    
    sample_results = metrics.get('sample_results', [])
    print(f"Total samples: {len(sample_results)}")
    
    if not sample_results:
        print(f"❌ {latest_metrics} contains no sample results - nothing to analyze.")
        return 1
    
    # Count exact matches
    exact_matches = sum(1 for sample in sample_results if sample['exact_match'])
    incorrect_predictions = len(sample_results) - exact_matches
    
    print(f"Exact matches: {exact_matches}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Reported accuracy: {exact_matches/len(sample_results):.2%}")
    
    # Find the incorrect predictions
    incorrect_samples = [sample for sample in sample_results if not sample['exact_match']]
    
    print(f"\n❌ INCORRECT PREDICTIONS ({len(incorrect_samples)}):")
    print("=" * 50)
    
    for i, sample in enumerate(incorrect_samples):
        gt_len = len(sample['ground_truth'])
        print(f"\nIncorrect Sample {i+1}:")
        print(f"  GT:    {sample['ground_truth']}")
        print(f"  Pred:  {sample['prediction']}")
        print(f"  Match: {sample['match_pattern']}")
        print(f"  Chars: {sample['chars_correct']}/{gt_len} correct ({sample['char_accuracy']:.1%})")
        print(f"  Edit Distance: {sample['edit_distance']}")
        print(f"  Confidence: {sample['confidence']:.2%}")
    
    # Show the first 10 samples (what the current output shows)
    print(f"\n📋 FIRST 10 SAMPLES (What current output shows):")
    print("=" * 50)
    
    for i, sample in enumerate(sample_results[:10]):
        gt_len = len(sample['ground_truth'])
        status = "✓ EXACT" if sample['exact_match'] else f"✗ {sample['chars_correct']}/{gt_len}"
        print(f"Sample {i+1}:")
        print(f"  GT:    {sample['ground_truth']}")
        print(f"  Pred:  {sample['prediction']}")
        print(f"  Match: {sample['match_pattern']} [{status}]")
    
    # Analysis of why the discrepancy occurs (computed from the data)
    print(f"\n💡 ANALYSIS OF THE DISCREPANCY:")
    print("=" * 50)
    
    shown = min(10, len(sample_results))
    first_10_correct = sum(1 for sample in sample_results[:10] if sample['exact_match'])
    incorrect_positions = [
        i + 1 for i, sample in enumerate(sample_results) if not sample['exact_match']
    ]
    print(f"First {shown} samples correct: {first_10_correct}/{shown}")
    if not incorrect_positions:
        print("All predictions are exact matches - no discrepancy to explain.")
    elif min(incorrect_positions) > shown:
        print(f"This explains why the sample output shows only correct predictions!")
        print(f"The incorrect predictions are in samples "
              f"{min(incorrect_positions)}-{max(incorrect_positions)}, "
              f"not shown in the text output.")
    else:
        print(f"Incorrect predictions start at sample {min(incorrect_positions)}, "
              f"within the first {shown} shown.")
    
    # Show where incorrect predictions appear
    print(f"\n📍 POSITION OF INCORRECT PREDICTIONS:")
    print("=" * 50)
    
    for i, sample in enumerate(sample_results):
        if not sample['exact_match']:
            gt_len = len(sample['ground_truth'])
            print(f"Sample {i+1}: {sample['ground_truth']} -> {sample['prediction']} "
                  f"({sample['chars_correct']}/{gt_len})")
    return 0

if __name__ == "__main__":
    sys.exit(find_incorrect_predictions())
