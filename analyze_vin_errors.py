#!/usr/bin/env python3
"""
Comprehensive Error Analysis for VIN Recognition
==============================================

Detailed analysis of the 15 failed VIN predictions including:
1. Common failure patterns
2. Position-specific accuracy analysis  
3. Character confusion matrix
4. Error categorization and recommendations

Usage:
    python analyze_vin_errors.py
"""

import difflib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from pathlib import Path

def load_training_metrics():
    """Load training metrics from JSON file."""
    metrics_path = Path("output/vin_rec_finetune/training_metrics.json")
    with open(metrics_path, 'r') as f:
        return json.load(f)

def extract_failed_predictions(metrics):
    """Extract the 15 failed VIN predictions."""
    failed = []
    for sample in metrics['evaluation_metrics']['sample_results']:
        if not sample['exact_match']:
            failed.append(sample)
    return failed

def analyze_error_patterns(failed_predictions):
    """Analyze common error patterns in failed predictions.

    Error positions, substitutions and error types are derived from
    edit-distance ALIGNMENT (difflib opcodes), not positional zip: a single
    insertion/deletion no longer smears a false "error" over every later
    position, and the insertion/deletion counters are actually reachable -
    a single-char deletion is classified as a deletion, not a substitution
    (audit M32/L40).
    """
    patterns = {
        'edit_distance_distribution': Counter(),
        'error_positions': defaultdict(int),
        'common_substitutions': Counter(),
        'error_types': {
            'substitution': 0,
            'insertion': 0,
            'deletion': 0,
            'multiple_errors': 0
        }
    }
    
    for pred in failed_predictions:
        gt = pred['ground_truth']
        pred_text = pred['prediction']
        edit_dist = pred['edit_distance']
        
        patterns['edit_distance_distribution'][edit_dist] += 1
        
        # Align ground truth and prediction, then walk the edit operations.
        n_sub = n_ins = n_del = 0
        matcher = difflib.SequenceMatcher(None, gt, pred_text, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            if tag == 'replace':
                gt_seg = gt[i1:i2]
                pr_seg = pred_text[j1:j2]
                paired = min(len(gt_seg), len(pr_seg))
                n_sub += paired
                for k in range(paired):
                    patterns['error_positions'][i1 + k + 1] += 1  # 1-indexed GT positions
                    patterns['common_substitutions'][f"{pr_seg[k]}→{gt_seg[k]}"] += 1
                if len(gt_seg) > paired:
                    n_del += len(gt_seg) - paired
                    for k in range(paired, len(gt_seg)):
                        patterns['error_positions'][i1 + k + 1] += 1
                elif len(pr_seg) > paired:
                    n_ins += len(pr_seg) - paired
            elif tag == 'delete':
                n_del += i2 - i1
                for k in range(i1, i2):
                    patterns['error_positions'][k + 1] += 1
            elif tag == 'insert':
                n_ins += j2 - j1
        
        # Categorize error types from the aligned operations
        total_ops = n_sub + n_ins + n_del
        if total_ops == 1:
            if n_sub == 1:
                patterns['error_types']['substitution'] += 1
            elif n_ins == 1:
                patterns['error_types']['insertion'] += 1
            else:
                patterns['error_types']['deletion'] += 1
        elif total_ops > 1:
            patterns['error_types']['multiple_errors'] += 1
    
    return patterns

def analyze_position_accuracy(metrics):
    """Analyze position-specific accuracy from metrics."""
    pos_acc = metrics['evaluation_metrics']['position_accuracy']
    
    # Sort positions by accuracy (worst first)
    sorted_positions = sorted(pos_acc.items(), key=lambda x: x[1])
    
    return {
        'worst_positions': sorted_positions[:5],
        'best_positions': sorted_positions[-5:],
        'average_accuracy': np.mean(list(pos_acc.values())),
        'position_std': np.std(list(pos_acc.values()))
    }

def create_confusion_matrix(metrics):
    """Create character confusion matrix from top confusions."""
    confusions = metrics['evaluation_metrics']['top_confusions']
    
    # Get all unique characters
    all_chars = set()
    for conf in confusions:
        all_chars.add(conf['predicted'])
        all_chars.add(conf['actual'])
    
    all_chars = sorted(list(all_chars))
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    
    # Create confusion matrix
    matrix = np.zeros((len(all_chars), len(all_chars)))
    
    for conf in confusions:
        pred_idx = char_to_idx[conf['predicted']]
        actual_idx = char_to_idx[conf['actual']]
        matrix[pred_idx][actual_idx] = conf['count']
    
    return matrix, all_chars

def analyze_character_performance(metrics):
    """Analyze per-character performance metrics."""
    per_class = metrics['evaluation_metrics']['per_class_metrics']
    
    # Sort characters by F1 score (worst first)
    sorted_by_f1 = sorted(per_class.items(), key=lambda x: x[1]['f1'])
    
    # Identify problematic characters
    # (loop var renamed: `metrics` shadowed this function's parameter, L40)
    problematic = []
    for char, char_metrics in per_class.items():
        if char_metrics['f1'] < 0.8 or char_metrics['precision'] < 0.8 or char_metrics['recall'] < 0.8:
            problematic.append((char, char_metrics))
    
    return {
        'worst_performing': sorted_by_f1[:5],
        'best_performing': sorted_by_f1[-5:],
        'problematic_chars': problematic
    }

def generate_error_report(metrics, failed_predictions, patterns, pos_analysis, char_analysis):
    """Generate comprehensive error analysis report."""
    
    report = []
    report.append("# VIN Recognition Error Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    # Executive Summary (denominator computed from the data, not hardcoded)
    total_predictions = len(metrics['evaluation_metrics']['sample_results'])
    failed_pct = (len(failed_predictions) / total_predictions * 100) if total_predictions else 0.0
    report.append("## Executive Summary")
    report.append(f"- **Total Failed Predictions**: {len(failed_predictions)}/{total_predictions} ({failed_pct:.1f}%)")
    report.append(f"- **Character Accuracy**: {metrics['evaluation_metrics']['character_level']['character_accuracy']:.1%}")
    report.append(f"- **VIN Accuracy**: {metrics['evaluation_metrics']['image_level']['exact_match_accuracy']:.1%}")
    report.append(f"- **Performance Gap**: {(metrics['evaluation_metrics']['character_level']['character_accuracy'] - metrics['evaluation_metrics']['image_level']['exact_match_accuracy'])*100:.1f} percentage points")
    report.append("")
    
    # Error Pattern Analysis
    report.append("## 1. Error Pattern Analysis")
    report.append("")
    
    report.append("### Edit Distance Distribution")
    for dist, count in sorted(patterns['edit_distance_distribution'].items()):
        report.append(f"- {dist} characters wrong: {count} VINs ({count/len(failed_predictions)*100:.1f}%)")
    report.append("")
    
    report.append("### Error Types")
    for error_type, count in patterns['error_types'].items():
        if count > 0:
            report.append(f"- {error_type.title()}: {count} cases")
    report.append("")
    
    # Position-Specific Analysis
    report.append("## 2. Position-Specific Accuracy Analysis")
    report.append("")
    
    report.append("### Most Problematic Positions")
    for pos, acc in pos_analysis['worst_positions']:
        report.append(f"- Position {pos}: {acc:.1%} accuracy")
    report.append("")
    
    report.append("### Most Reliable Positions") 
    for pos, acc in reversed(pos_analysis['best_positions']):
        report.append(f"- Position {pos}: {acc:.1%} accuracy")
    report.append("")
    
    report.append(f"**Position Statistics:**")
    report.append(f"- Average accuracy: {pos_analysis['average_accuracy']:.1%}")
    report.append(f"- Standard deviation: {pos_analysis['position_std']:.3f}")
    report.append("")
    
    # Character Confusion Analysis
    report.append("## 3. Character Confusion Analysis")
    report.append("")
    
    report.append("### Top Character Substitutions")
    for sub, count in patterns['common_substitutions'].most_common(10):
        report.append(f"- {sub}: {count} times")
    report.append("")
    
    # Character Performance Analysis
    report.append("## 4. Character Performance Analysis")
    report.append("")
    
    report.append("### Worst Performing Characters")
    for char, char_metrics in char_analysis['worst_performing']:
        report.append(f"- **{char}**: F1={char_metrics['f1']:.3f}, Precision={char_metrics['precision']:.3f}, Recall={char_metrics['recall']:.3f}")
    report.append("")
    
    if char_analysis['problematic_chars']:
        report.append("### Problematic Characters (F1 < 0.8)")
        for char, char_metrics in char_analysis['problematic_chars']:
            report.append(f"- **{char}**: F1={char_metrics['f1']:.3f}, Support={char_metrics['support']}")
        report.append("")
    
    # Detailed Failed VIN Analysis
    report.append("## 5. Detailed Failed VIN Analysis")
    report.append("")
    
    report.append("### Failed Predictions by Error Count")
    failed_by_errors = defaultdict(list)
    for pred in failed_predictions:
        failed_by_errors[pred['edit_distance']].append(pred)
    
    for error_count in sorted(failed_by_errors.keys()):
        report.append(f"#### {error_count} Character Errors ({len(failed_by_errors[error_count])} VINs)")
        for pred in failed_by_errors[error_count]:
            gt = pred['ground_truth']
            pred_text = pred['prediction']
            
            # Show character differences (explicit length handling: positions
            # present in only one of the strings are marked as errors instead
            # of being silently truncated by zip)
            diff = ""
            for i in range(max(len(gt), len(pred_text))):
                g = gt[i] if i < len(gt) else None
                p = pred_text[i] if i < len(pred_text) else None
                diff += "✓" if (g is not None and g == p) else "✗"
            
            report.append(f"  - GT:  {gt}")
            report.append(f"  - Pred:{pred_text}")
            report.append(f"  - Diff:{diff}")
            report.append(f"  - Confidence: {pred['confidence']:.3f}")
            report.append("")
    
    # Recommendations (computed from the actual input data, not hardcoded)
    report.append("## 6. Recommendations")
    report.append("")
    
    report.append("### Immediate Actions")
    action_num = 1
    if pos_analysis['worst_positions']:
        worst_pos, worst_acc = pos_analysis['worst_positions'][0]
        pos_label = worst_pos.replace('_', ' ').title()
        report.append(f"{action_num}. **Focus on {pos_label}**: Only {worst_acc:.1%} accuracy - highest priority")
        action_num += 1
    top_subs = [sub for sub, _count in patterns['common_substitutions'].most_common(3)]
    if top_subs:
        report.append(f"{action_num}. **Improve Character Discrimination**: Target {', '.join(top_subs)} confusions")
        action_num += 1
    problem_chars = sorted({char for char, _cm in char_analysis['problematic_chars']})
    if problem_chars:
        report.append(f"{action_num}. **Data Augmentation**: Add more examples for problematic characters ({', '.join(problem_chars)})")
    report.append("")
    
    report.append("### Medium-term Improvements")
    report.append("1. **Position-aware Training**: Weight loss more heavily for problematic positions")
    report.append("2. **Character-specific Augmentation**: Generate synthetic data for confusing character pairs")
    report.append("3. **Ensemble Methods**: Combine multiple model predictions")
    report.append("")
    
    report.append("### Long-term Strategies")
    report.append("1. **Post-processing Rules**: VIN format validation and correction")
    report.append("2. **Multi-stage Recognition**: Separate character detection from sequence recognition")
    report.append("3. **Context-aware Models**: Use VIN structure and position constraints")
    report.append("")
    
    return "\n".join(report)

def save_confusion_matrix_visualization(matrix, characters):
    """Create and save confusion matrix heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, 
                xticklabels=characters, 
                yticklabels=characters,
                annot=True, 
                fmt='g',
                cmap='Blues',
                cbar_kws={'label': 'Count'})
    plt.title('VIN Character Confusion Matrix')
    plt.xlabel('Actual Character')
    plt.ylabel('Predicted Character')
    plt.tight_layout()
    plt.savefig('output/vin_rec_finetune/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_position_accuracy_plot(position_accuracy):
    """Create position accuracy plot."""
    positions = list(range(1, 18))
    accuracies = [position_accuracy[f'position_{i}'] for i in positions]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(positions, accuracies, color='steelblue', alpha=0.7)
    
    # Color bars based on performance
    for bar, acc in zip(bars, accuracies, strict=True):
        if acc < 0.8:
            bar.set_color('red')
        elif acc < 0.9:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
    
    plt.xlabel('VIN Position')
    plt.ylabel('Accuracy')
    plt.title('Position-Specific Accuracy in VIN Recognition')
    plt.xticks(positions)
    plt.ylim(0.6, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/vin_rec_finetune/position_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function."""
    print("🔍 Starting VIN Error Analysis...")
    
    # Load metrics
    metrics = load_training_metrics()
    failed_predictions = extract_failed_predictions(metrics)
    
    print(f"📊 Found {len(failed_predictions)} failed predictions")
    
    # Perform analyses
    patterns = analyze_error_patterns(failed_predictions)
    pos_analysis = analyze_position_accuracy(metrics)
    char_analysis = analyze_character_performance(metrics)
    matrix, characters = create_confusion_matrix(metrics)
    
    # Generate report
    report = generate_error_report(metrics, failed_predictions, patterns, pos_analysis, char_analysis)
    
    # Save outputs
    with open('output/vin_rec_finetune/error_analysis_report.md', 'w') as f:
        f.write(report)
    
    # Create visualizations
    save_confusion_matrix_visualization(matrix, characters)
    save_position_accuracy_plot(metrics['evaluation_metrics']['position_accuracy'])
    
    print("✅ Analysis complete!")
    print("📁 Report saved to: output/vin_rec_finetune/error_analysis_report.md")
    print("📊 Confusion matrix saved to: output/vin_rec_finetune/confusion_matrix.png")
    print("📈 Position accuracy plot saved to: output/vin_rec_finetune/position_accuracy.png")
    
    # Print key findings (computed from the data, not hardcoded)
    print("\n🎯 Key Findings:")
    pos_acc = metrics['evaluation_metrics']['position_accuracy']
    if pos_acc:
        worst_pos, worst_acc = min(pos_acc.items(), key=lambda item: item[1])
        print(f"- Worst position: {worst_pos.replace('_', ' ').title()} ({worst_acc:.1%} accuracy)")
    if patterns['common_substitutions']:
        top_sub, top_count = patterns['common_substitutions'].most_common(1)[0]
        print(f"- Most common confusion: {top_sub} ({top_count} times)")
    print(f"- Average edit distance: {metrics['evaluation_metrics']['edit_distance_distribution']['mean']:.2f} characters")

if __name__ == "__main__":
    main()
