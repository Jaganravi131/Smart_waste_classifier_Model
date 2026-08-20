"""
Quick Accuracy Report for Trash Detection Model
Generates a concise summary of key performance metrics
"""

import pandas as pd
import os

def quick_accuracy_report(csv_path='logs/detections.csv'):
    """Generate a quick accuracy report with key metrics."""
    if not os.path.exists(csv_path):
        print(f"Error: Detection log file not found: {csv_path}")
        return
    
    # Load data
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    
    # Overall statistics
    total_frames = len(df)
    frames_with_detections = df['class'].notna().sum()
    detection_rate = frames_with_detections / total_frames * 100
    
    # Class distribution
    class_counts = df['class'].value_counts(dropna=False)
    class_percentages = df['class'].value_counts(normalize=True, dropna=False) * 100
    
    # Confidence analysis
    detected_df = df[df['class'].notna()].copy()
    
    # Confidence by class
    confidence_by_class = detected_df.groupby('class')['confidence'].agg(['count', 'mean', 'std']).round(4)
    
    # Confidence distribution
    high_conf = len(detected_df[detected_df['confidence'] > 0.8])
    medium_conf = len(detected_df[(detected_df['confidence'] >= 0.5) & (detected_df['confidence'] <= 0.8)])
    low_conf = len(detected_df[detected_df['confidence'] < 0.5])
    
    # Print report
    print("=" * 60)
    print("QUICK ACCURACY REPORT")
    print("=" * 60)
    print(f"{'Total Frames Processed:':<25} {total_frames:>10}")
    print(f"{'Detection Rate:':<25} {detection_rate:>9.2f}%")
    print()
    
    print("CLASS DISTRIBUTION:")
    print("-" * 30)
    for class_name, count in class_counts.items():
        class_label = class_name if pd.notna(class_name) else "No Detection"
        percentage = class_percentages[class_name]
        print(f"  {class_label:<15} {count:>6} ({percentage:>6.2f}%)")
    print()
    
    print("CONFIDENCE BY CLASS:")
    print("-" * 35)
    for class_name in confidence_by_class.index:
        count = confidence_by_class.loc[class_name, 'count']
        mean_conf = confidence_by_class.loc[class_name, 'mean']
        std_conf = confidence_by_class.loc[class_name, 'std']
        print(f"  {class_name:<12} {mean_conf:>7.4f} ± {std_conf:<7.4f} (n={count})")
    print()
    
    print("CONFIDENCE DISTRIBUTION:")
    print("-" * 35)
    print(f"  {'High   (>0.8):':<20} {high_conf:>6} ({high_conf/len(detected_df)*100:>6.2f}%)")
    print(f"  {'Medium (0.5-0.8):':<20} {medium_conf:>6} ({medium_conf/len(detected_df)*100:>6.2f}%)")
    print(f"  {'Low    (<0.5):':<20} {low_conf:>6} ({low_conf/len(detected_df)*100:>6.2f}%)")
    print()
    
    # Best and worst performing classes
    best_class = confidence_by_class['mean'].idxmax()
    best_confidence = confidence_by_class.loc[best_class, 'mean']
    worst_class = confidence_by_class['mean'].idxmin()
    worst_confidence = confidence_by_class.loc[worst_class, 'mean']
    
    print("PERFORMANCE SUMMARY:")
    print("-" * 25)
    print(f"  {'Best Class:':<15} {best_class} ({best_confidence:.4f})")
    print(f"  {'Worst Class:':<15} {worst_class} ({worst_confidence:.4f})")
    print(f"  {'Most Frequent:':<15} {class_counts.index[0] if pd.notna(class_counts.index[0]) else class_counts.index[1]} ({class_counts.iloc[0] if pd.notna(class_counts.index[0]) else class_counts.iloc[1]} detections)")

if __name__ == "__main__":
    quick_accuracy_report()