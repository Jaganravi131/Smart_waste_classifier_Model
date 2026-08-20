"""
Generate Accuracy Graphs for Trash Detection Model
=================================================

This script generates all visualization graphs for the accuracy analysis
without displaying text reports, focusing solely on graphical output.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def load_detection_data(csv_path='logs/detections.csv'):
    """Load detection data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Detection log file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    print(f"Loaded {len(df)} detection records")
    return df

def generate_all_graphs():
    """Generate all accuracy graphs and save them as image files."""
    try:
        # Load data
        df = load_detection_data()
        detected_df = df[df['class'].notna()].copy()
        
        # Create a figure with subplots
        plt.style.use('seaborn-v0_8')  # Use seaborn style for better-looking plots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Class distribution pie chart
        ax1 = plt.subplot(2, 3, 1)
        class_counts = df['class'].value_counts(dropna=False)
        labels = [str(name) if pd.notna(name) else 'No Detection' for name in class_counts.index]
        # Create a color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax1.pie(class_counts.values, labels=labels, autopct='%1.1f%%', 
                                          startangle=90, colors=colors, textprops=dict(color="w"))
        ax1.set_title('Detection Class Distribution', fontsize=16, pad=20)
        # Improve the autopct text visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 2. Confidence distribution histogram
        ax2 = plt.subplot(2, 3, 2)
        n, bins, patches = ax2.hist(detected_df['confidence'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Overall Confidence Distribution', fontsize=16)
        ax2.grid(True, alpha=0.3)
        
        # 3. Class-wise confidence comparison (box plot)
        ax3 = plt.subplot(2, 3, 3)
        # Filter out classes with very few samples for better visualization
        class_data = [detected_df[detected_df['class'] == cls]['confidence'].values 
                     for cls in detected_df['class'].unique()]
        class_labels = [str(cls) for cls in detected_df['class'].unique()]
        
        box_plot = ax3.boxplot(class_data, labels=class_labels, patch_artist=True)
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_labels)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            
        ax3.set_title('Confidence by Class', fontsize=16)
        ax3.set_xlabel('Class', fontsize=12)
        ax3.set_ylabel('Confidence', fontsize=12)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
        
        # 4. Mean confidence by class (bar chart)
        ax4 = plt.subplot(2, 3, 4)
        class_stats = detected_df.groupby('class')['confidence'].mean().sort_values(ascending=False)
        bars = ax4.bar(range(len(class_stats)), class_stats.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(class_stats))))
        ax4.set_xlabel('Class', fontsize=12)
        ax4.set_ylabel('Mean Confidence', fontsize=12)
        ax4.set_title('Mean Confidence by Class', fontsize=16)
        ax4.set_xticks(range(len(class_stats)))
        ax4.set_xticklabels(class_stats.index, rotation=45, ha="right")
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, class_stats.values)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Confidence categories by class (stacked bar chart)
        ax5 = plt.subplot(2, 3, 5)
        confidence_categories = ['High (>0.8)', 'Medium (0.5-0.8)', 'Low (<0.5)']
        class_names = detected_df['class'].unique()
        
        # Prepare data for stacked bar chart
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
        
        # Create stacked bar chart
        x = np.arange(len(class_names))
        width = 0.6
        
        bottom = np.zeros(len(class_names))
        colors = ['red', 'orange', 'green']  # Low, Medium, High
        
        for i, (category, color) in enumerate(zip(confidence_categories, colors)):
            ax5.bar(x, category_data[category], width, label=category, 
                   bottom=bottom, color=color, alpha=0.7)
            bottom += np.array(category_data[category])
        
        ax5.set_xlabel('Class', fontsize=12)
        ax5.set_ylabel('Count', fontsize=12)
        ax5.set_title('Confidence Categories by Class', fontsize=16)
        ax5.set_xticks(x)
        ax5.set_xticklabels(class_names, rotation=45, ha="right")
        ax5.legend()
        
        # 6. Detections over time
        ax6 = plt.subplot(2, 3, 6)
        df_with_time = df.copy()
        # Group by 1-minute intervals for better visualization
        df_with_time['time_bin'] = df_with_time['timestamp'].dt.floor('1T')
        detections_per_period = df_with_time[df_with_time['class'].notna()].groupby('time_bin').size()
        
        ax6.plot(detections_per_period.index, detections_per_period.values, marker='o', linewidth=2, markersize=4)
        ax6.set_xlabel('Time', fontsize=12)
        ax6.set_ylabel('Detections per Minute', fontsize=12)
        ax6.set_title('Detections Over Time', fontsize=16)
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout(pad=3.0)
        
        # Save the figure
        plt.savefig('accuracy_analysis_graphs.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Graphs generated successfully!")
        print("Saved as 'accuracy_analysis_graphs.png'")
        
        # Also generate individual graphs for better clarity
        generate_individual_graphs(df, detected_df)
        
    except Exception as e:
        print(f"Error generating graphs: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_individual_graphs(df, detected_df):
    """Generate individual graphs for better clarity."""
    
    # 1. Class distribution pie chart (larger)
    plt.figure(figsize=(10, 8))
    class_counts = df['class'].value_counts(dropna=False)
    labels = [str(name) if pd.notna(name) else 'No Detection' for name in class_counts.index]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    plt.pie(class_counts.values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Detection Class Distribution', fontsize=16)
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confidence distribution histogram (larger)
    plt.figure(figsize=(10, 6))
    plt.hist(detected_df['confidence'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Overall Confidence Distribution', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class-wise confidence comparison (larger box plot)
    plt.figure(figsize=(12, 6))
    class_data = [detected_df[detected_df['class'] == cls]['confidence'].values 
                 for cls in detected_df['class'].unique()]
    class_labels = [str(cls) for cls in detected_df['class'].unique()]
    box_plot = plt.boxplot(class_data, labels=class_labels, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_labels)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Confidence by Class', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Confidence', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Individual graphs saved:")
    print("  - class_distribution.png")
    print("  - confidence_distribution.png")
    print("  - confidence_by_class.png")

def main():
    """Main function to generate all accuracy graphs."""
    print("Generating accuracy analysis graphs...")
    generate_all_graphs()
    print("All graphs generated successfully!")

if __name__ == "__main__":
    main()