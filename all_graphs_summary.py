"""
Summary of All Generated Graphs for Trash Detection Model
========================================================

This script provides information about all the graphs generated for the 
trash detection model analysis.
"""

import os
from datetime import datetime

def summarize_all_graphs():
    """Provide a summary of all generated graphs."""
    
    # Define all generated graphs with descriptions
    graphs_info = {
        "model_accuracy_vs_epoch.png": {
            "title": "Model Accuracy vs. Epoch",
            "description": "Simulated training and validation accuracy over epochs",
            "dimensions": "12x8 inches",
            "content": [
                "Training accuracy curve (blue line with circles)",
                "Validation accuracy curve (green dashed line with squares)",
                "Best validation accuracy annotation",
                "Grid and legend for clarity"
            ]
        },
        "model_loss_vs_epoch.png": {
            "title": "Model Loss vs. Epoch",
            "description": "Simulated training and validation loss over epochs",
            "dimensions": "12x8 inches",
            "content": [
                "Training loss curve (red line with circles)",
                "Validation loss curve (orange dashed line with squares)",
                "Lowest validation loss annotation",
                "Grid and legend for clarity"
            ]
        },
        "confidence_trend_over_time.png": {
            "title": "Confidence Trend Over Time",
            "description": "Actual detection confidence trends using real data",
            "dimensions": "12x8 inches",
            "content": [
                "Rolling mean confidence (purple line)",
                "Standard deviation band (shaded area)",
                "Time-based x-axis with rotated labels",
                "Grid for better readability"
            ]
        },
        "class_performance_comparison.png": {
            "title": "Class Performance Comparison",
            "description": "Model performance comparison across different trash classes",
            "dimensions": "12x8 inches",
            "content": [
                "Bar chart showing mean confidence by class",
                "Error bars representing standard deviation",
                "Sample count annotations (n=values)",
                "Color-coded bars for different classes"
            ]
        },
        "accuracy_analysis_graphs.png": {
            "title": "Combined Accuracy Analysis Dashboard",
            "description": "All accuracy analysis graphs in one comprehensive dashboard",
            "dimensions": "20x15 inches",
            "content": [
                "Class distribution pie chart (top left)",
                "Confidence distribution histogram (top middle)",
                "Class-wise confidence box plot (top right)",
                "Mean confidence by class bar chart (bottom left)",
                "Confidence categories stacked bar (bottom middle)",
                "Detections over time line plot (bottom right)"
            ]
        },
        "class_distribution.png": {
            "title": "Class Distribution Pie Chart",
            "description": "Large pie chart showing distribution of detected classes",
            "dimensions": "10x8 inches",
            "content": [
                "Colored segments for each class",
                "Percentage labels on each segment",
                "Legend with class names"
            ]
        },
        "confidence_distribution.png": {
            "title": "Confidence Distribution Histogram",
            "description": "Histogram showing overall confidence value distribution",
            "dimensions": "10x6 inches",
            "content": [
                "Frequency distribution of confidence values",
                "50 bins for detailed granularity",
                "Grid for better readability"
            ]
        },
        "confidence_by_class.png": {
            "title": "Confidence by Class Box Plot",
            "description": "Box plot comparing confidence distributions across classes",
            "dimensions": "12x6 inches",
            "content": [
                "Box plots for each class with quartiles",
                "Outliers represented as individual points",
                "Color-coded boxes for visual distinction"
            ]
        },
        "accuracy_analysis_report.png": {
            "title": "Original Analysis Report Graphs",
            "description": "Original analysis report graphs from accuracy_analysis.py",
            "dimensions": "15x12 inches",
            "content": [
                "Class distribution pie chart",
                "Confidence distribution histogram",
                "Class-wise confidence comparison",
                "Mean confidence by class bar chart",
                "Detection rate over time line plot",
                "Confidence categories by class stacked bar"
            ]
        },
        "model_training_visualization.png": {
            "title": "Model Training Visualization",
            "description": "Comprehensive visualization with training-like graphs",
            "dimensions": "15x10 inches",
            "content": [
                "Simulated accuracy vs epoch",
                "Simulated loss vs epoch",
                "Actual confidence trend over time",
                "Class-wise performance comparison"
            ]
        }
    }
    
    print("=" * 70)
    print("TRASH DETECTION MODEL - COMPLETE GRAPH INVENTORY")
    print("=" * 70)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check which files actually exist
    existing_graphs = {}
    for filename, info in graphs_info.items():
        if os.path.exists(filename):
            existing_graphs[filename] = info
    
    if not existing_graphs:
        print("No graph files found in the current directory.")
        return
    
    print(f"Found {len(existing_graphs)} graph file(s):")
    print("-" * 70)
    
    for i, (filename, info) in enumerate(existing_graphs.items(), 1):
        # Get file size
        file_size = os.path.getsize(filename)
        file_size_kb = file_size / 1024  # Convert to KB
        
        print(f"{i}. {filename}")
        print(f"   Title: {info['title']}")
        print(f"   Description: {info['description']}")
        print(f"   Dimensions: {info['dimensions']}")
        print(f"   File Size: {file_size_kb:.1f} KB")
        print("   Content:")
        for j, content_item in enumerate(info['content'], 1):
            print(f"     {j}. {content_item}")
        print()

def main():
    """Main function to display graph inventory."""
    summarize_all_graphs()

if __name__ == "__main__":
    main()