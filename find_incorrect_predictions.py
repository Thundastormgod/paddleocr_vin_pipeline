#!/usr/bin/env python3
"""
Find the incorrect predictions that are missing from the sample output.
"""

import json
import os
from pathlib import Path

def find_incorrect_predictions():
    """Find and display the incorrect predictions."""
    
    # Look for the most recent evaluation metrics file
    output_dir = Path("output/vin_rec_finetune")
    metrics_files = list(output_dir.glob("*evaluation_metrics.json"))
    
    if not metrics_files:
        print("❌ No evaluation metrics file found!")
        return
    
    # Use the most recent metrics file
    latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Loading metrics from: {latest_metrics}")
    
    with open(latest_metrics) as f:
        metrics = json.load(f)
    
    print("\n🔍 ANALYZING PREDICTIONS:")
    print("=" * 50)
    
    sample_results = metrics.get('sample_results', [])
    print(f"Total samples: {len(sample_results)}")
    
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
        print(f"\nIncorrect Sample {i+1}:")
        print(f"  GT:    {sample['ground_truth']}")
        print(f"  Pred:  {sample['prediction']}")
        print(f"  Match: {sample['match_pattern']}")
        print(f"  Chars: {sample['chars_correct']}/17 correct ({sample['char_accuracy']:.1%})")
        print(f"  Edit Distance: {sample['edit_distance']}")
        print(f"  Confidence: {sample['confidence']:.2%}")
    
    # Show the first 10 samples (what the current output shows)
    print(f"\n📋 FIRST 10 SAMPLES (What current output shows):")
    print("=" * 50)
    
    for i, sample in enumerate(sample_results[:10]):
        status = "✓ EXACT" if sample['exact_match'] else f"✗ {sample['chars_correct']}/17"
        print(f"Sample {i+1}:")
        print(f"  GT:    {sample['ground_truth']}")
        print(f"  Pred:  {sample['prediction']}")
        print(f"  Match: {sample['match_pattern']} [{status}]")
    
    # Analysis of why the discrepancy occurs
    print(f"\n💡 ANALYSIS OF THE DISCREPANCY:")
    print("=" * 50)
    
    first_10_correct = sum(1 for sample in sample_results[:10] if sample['exact_match'])
    print(f"First 10 samples correct: {first_10_correct}/10")
    print(f"This explains why the sample output shows only correct predictions!")
    print(f"The incorrect predictions are in samples 11-43, not shown in the text output.")
    
    # Show where incorrect predictions appear
    print(f"\n📍 POSITION OF INCORRECT PREDICTIONS:")
    print("=" * 50)
    
    for i, sample in enumerate(sample_results):
        if not sample['exact_match']:
            print(f"Sample {i+1}: {sample['ground_truth']} -> {sample['prediction']} ({sample['chars_correct']}/17)")

if __name__ == "__main__":
    find_incorrect_predictions()
