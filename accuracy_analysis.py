"""
Accuracy Data Comparison Analysis for Trash Detection Model
==========================================================

This script provides a comprehensive analysis of the trash detection model's performance
using the logged detection data. It includes:

1. Overall detection statistics
2. Class-wise accuracy analysis
3. Confidence distribution analysis
4. Temporal performance trends
5. Comparative analysis between classes
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def load_detection_data(csv_path='logs/detections.csv'):
    """Load detection data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Detection log file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    print(f"Loaded {len(df)} detection records")
    return df

def overall_statistics(df):
    """Calculate overall detection statistics."""
    total_frames = len(df)
    frames_with_detections = df['class'].notna().sum()
    frames_without_detections = df['class'].isna().sum()
    
    detection_rate = frames_with_detections / total_frames * 100
    
    print("=== OVERALL STATISTICS ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Frames without detections: {frames_without_detections}")
    print(f"Detection rate: {detection_rate:.2f}%")
    print()
    
    return {
        'total_frames': total_frames,
        'frames_with_detections': frames_with_detections,
        'frames_without_detections': frames_without_detections,
        'detection_rate': detection_rate
    }

def class_wise_analysis(df):
    """Perform class-wise analysis of detections."""
    # Count detections per class
    class_counts = df['class'].value_counts(dropna=False)
    
    # Calculate percentages
    class_percentages = df['class'].value_counts(normalize=True, dropna=False) * 100
    
    print("=== CLASS-WISE ANALYSIS ===")
    print("Detections per class:")
    for class_name, count in class_counts.items():
        class_label = class_name if pd.notna(class_name) else "No Detection"
        percentage = class_percentages[class_name]
        print(f"  {class_label}: {count} ({percentage:.2f}%)")
    print()
    
    # Confidence statistics per class
    print("Confidence statistics per class:")
    confidence_stats = df.groupby('class')['confidence'].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
    print(confidence_stats)
    print()
    
    return class_counts, class_percentages, confidence_stats

def confidence_analysis(df):
    """Analyze confidence distributions."""
    # Filter out NaN detections
    detected_df = df[df['class'].notna()].copy()
    
    print("=== CONFIDENCE ANALYSIS ===")
    
    # Overall confidence statistics
    print("Overall confidence statistics (detected objects only):")
    print(f"  Mean confidence: {detected_df['confidence'].mean():.4f}")
    print(f"  Std deviation: {detected_df['confidence'].std():.4f}")
    print(f"  Min confidence: {detected_df['confidence'].min():.4f}")
    print(f"  Max confidence: {detected_df['confidence'].max():.4f}")
    print()
    
    # Confidence distribution by class
    print("Confidence distribution by class:")
    for class_name in detected_df['class'].unique():
        class_data = detected_df[detected_df['class'] == class_name]
        print(f"  {class_name}:")
        print(f"    Count: {len(class_data)}")
        print(f"    Mean: {class_data['confidence'].mean():.4f}")
        print(f"    Std: {class_data['confidence'].std():.4f}")
        print(f"    Min: {class_data['confidence'].min():.4f}")
        print(f"    Max: {class_data['confidence'].max():.4f}")
    print()
    
    return detected_df

def temporal_analysis(df):
    """Analyze temporal trends in detection performance."""
    # Add time-based features
    df_with_detections = df[df['class'].notna()].copy()
    df_with_detections['hour'] = df_with_detections['timestamp'].dt.hour
    df_with_detections['minute'] = df_with_detections['timestamp'].dt.minute
    df_with_detections['time_bin'] = df_with_detections['timestamp'].dt.floor('10T')  # 10-minute bins
    
    print("=== TEMPORAL ANALYSIS ===")
    
    # Detections over time (10-minute intervals)
    detections_over_time = df_with_detections.groupby('time_bin').size()
    print("Average detections per 10-minute period:")
    print(f"  Mean: {detections_over_time.mean():.2f}")
    print(f"  Std: {detections_over_time.std():.2f}")
    print()
    
    # Class distribution over time
    if len(detections_over_time) > 1:
        class_time_dist = df_with_detections.groupby(['time_bin', 'class']).size().unstack(fill_value=0)
        print("Class distribution over time (sample):")
        print(class_time_dist.head())
        print()
    
    return df_with_detections

def comparative_analysis(df):
    """Perform comparative analysis between different classes."""
    detected_df = df[df['class'].notna()].copy()
    
    print("=== COMPARATIVE ANALYSIS ===")
    
    # High confidence detections (>0.8)
    high_confidence = detected_df[detected_df['confidence'] > 0.8]
    print(f"High confidence detections (>0.8): {len(high_confidence)} ({len(high_confidence)/len(detected_df)*100:.2f}%)")
    
    # Medium confidence detections (0.5-0.8)
    medium_confidence = detected_df[(detected_df['confidence'] >= 0.5) & (detected_df['confidence'] <= 0.8)]
    print(f"Medium confidence detections (0.5-0.8): {len(medium_confidence)} ({len(medium_confidence)/len(detected_df)*100:.2f}%)")
    
    # Low confidence detections (<0.5)
    low_confidence = detected_df[detected_df['confidence'] < 0.5]
    print(f"Low confidence detections (<0.5): {len(low_confidence)} ({len(low_confidence)/len(detected_df)*100:.2f}%)")
    print()
    
    # Class-wise confidence categories
    print("Confidence distribution by class:")
    for class_name in detected_df['class'].unique():
        class_data = detected_df[detected_df['class'] == class_name]
        high = len(class_data[class_data['confidence'] > 0.8])
        medium = len(class_data[(class_data['confidence'] >= 0.5) & (class_data['confidence'] <= 0.8)])
        low = len(class_data[class_data['confidence'] < 0.5])
        total = len(class_data)
        
        print(f"  {class_name} (n={total}):")
        print(f"    High (>0.8): {high} ({high/total*100:.1f}%)")
        print(f"    Medium (0.5-0.8): {medium} ({medium/total*100:.1f}%)")
        print(f"    Low (<0.5): {low} ({low/total*100:.1f}%)")
    print()

def generate_visualizations(df, detected_df):
    """Generate visualizations for the analysis."""
    print("=== GENERATING VISUALIZATIONS ===")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Class distribution pie chart
    ax1 = plt.subplot(2, 3, 1)
    class_counts = df['class'].value_counts(dropna=False)
    labels = [str(name) if pd.notna(name) else 'No Detection' for name in class_counts.index]
    ax1.pie(class_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Detection Class Distribution')
    
    # 2. Confidence distribution histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(detected_df['confidence'], bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Overall Confidence Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Class-wise confidence comparison
    ax3 = plt.subplot(2, 3, 3)
    detected_df.boxplot(column='confidence', by='class', ax=ax3)
    ax3.set_title('Confidence by Class')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Confidence')
    plt.suptitle('')  # Remove default title
    
    # 4. Confidence statistics bar chart
    ax4 = plt.subplot(2, 3, 4)
    class_stats = detected_df.groupby('class')['confidence'].mean().sort_values(ascending=False)
    bars = ax4.bar(range(len(class_stats)), class_stats.values)
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Mean Confidence')
    ax4.set_title('Mean Confidence by Class')
    ax4.set_xticks(range(len(class_stats)))
    ax4.set_xticklabels(class_stats.index, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 5. Detection rate over time
    ax5 = plt.subplot(2, 3, 5)
    df_with_time = df.copy()
    df_with_time['time_bin'] = df_with_time['timestamp'].dt.floor('30T')  # 30-minute bins
    detections_per_period = df_with_time[df_with_time['class'].notna()].groupby('time_bin').size()
    no_detections_per_period = df_with_time[df_with_time['class'].isna()].groupby('time_bin').size()
    
    time_bins = sorted(detections_per_period.index.union(no_detections_per_period.index))
    detection_counts = [detections_per_period.get(tb, 0) for tb in time_bins]
    no_detection_counts = [no_detections_per_period.get(tb, 0) for tb in time_bins]
    
    x_pos = range(len(time_bins))
    ax5.bar(x_pos, detection_counts, label='Detections', alpha=0.7)
    ax5.bar(x_pos, no_detection_counts, bottom=detection_counts, label='No Detections', alpha=0.7)
    ax5.set_xlabel('Time Period')
    ax5.set_ylabel('Count')
    ax5.set_title('Detections Over Time')
    ax5.legend()
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Confidence categories by class
    ax6 = plt.subplot(2, 3, 6)
    confidence_categories = ['High (>0.8)', 'Medium (0.5-0.8)', 'Low (<0.5)']
    class_names = detected_df['class'].unique()
    
    category_data = {}
    for category in confidence_categories:
        category_data[category] = []
        for class_name in class_names:
            class_data = detected_df[detected_df['class'] == class_name]
            if category == 'High (>0.8)':
                count = len(class_data[class_data['confidence'] > 0.8])
            elif category == 'Medium (0.5-0.8)':
                count = len(class_data[(class_data['confidence'] >= 0.5) & (class_data['confidence'] <= 0.8)])
            else:  # Low (<0.5)
                count = len(class_data[class_data['confidence'] < 0.5])
            category_data[category].append(count)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax6.bar(x - width, category_data['High (>0.8)'], width, label='High (>0.8)')
    ax6.bar(x, category_data['Medium (0.5-0.8)'], width, label='Medium (0.5-0.8)')
    ax6.bar(x + width, category_data['Low (<0.5)'], width, label='Low (<0.5)')
    
    ax6.set_xlabel('Class')
    ax6.set_ylabel('Count')
    ax6.set_title('Confidence Categories by Class')
    ax6.set_xticks(x)
    ax6.set_xticklabels(class_names, rotation=45)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_analysis_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'accuracy_analysis_report.png'")
    print()

def generate_summary_report(df, detected_df, overall_stats, class_counts, confidence_stats):
    """Generate a summary report of the analysis."""
    print("=== ACCURACY ANALYSIS SUMMARY REPORT ===")
    
    # Top performing class (highest mean confidence)
    best_class = confidence_stats['mean'].idxmax()
    best_confidence = confidence_stats.loc[best_class, 'mean']
    
    # Most frequently detected class
    most_frequent_class = class_counts.index[0] if pd.notna(class_counts.index[0]) else class_counts.index[1]
    most_frequent_count = class_counts.iloc[0] if pd.notna(class_counts.index[0]) else class_counts.iloc[1]
    
    # Least confident class
    worst_class = confidence_stats['mean'].idxmin()
    worst_confidence = confidence_stats.loc[worst_class, 'mean']
    
    print(f"Top performing class: {best_class} (mean confidence: {best_confidence:.4f})")
    print(f"Most frequently detected class: {most_frequent_class} ({most_frequent_count} detections)")
    print(f"Least confident class: {worst_class} (mean confidence: {worst_confidence:.4f})")
    print(f"Overall detection rate: {overall_stats['detection_rate']:.2f}%")
    print(f"Total detections analyzed: {len(detected_df)}")
    
    # Time period
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    duration = end_time - start_time
    
    print(f"Analysis period: {start_time} to {end_time} (duration: {duration})")
    print()

def main():
    """Main function to run the accuracy analysis."""
    try:
        # Load data
        df = load_detection_data()
        
        # Overall statistics
        overall_stats = overall_statistics(df)
        
        # Class-wise analysis
        class_counts, class_percentages, confidence_stats = class_wise_analysis(df)
        
        # Confidence analysis
        detected_df = confidence_analysis(df)
        
        # Temporal analysis
        temporal_df = temporal_analysis(df)
        
        # Comparative analysis
        comparative_analysis(df)
        
        # Generate visualizations
        generate_visualizations(df, detected_df)
        
        # Generate summary report
        generate_summary_report(df, detected_df, overall_stats, class_counts, confidence_stats)
        
        print("Accuracy analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()