"""
Graph Inventory for Trash Detection Model Analysis
==================================================

This script lists all the generated graphs and provides information about each one.
"""

import os
from datetime import datetime

def list_generated_graphs():
    """List all generated graph files with descriptions."""
    
    graphs = {
        "accuracy_analysis_graphs.png": {
            "description": "Combined dashboard with all 6 analysis graphs in one image",
            "size": "20x15 inches",
            "subplots": [
                "Class Distribution Pie Chart",
                "Confidence Distribution Histogram",
                "Class-wise Confidence Box Plot",
                "Mean Confidence by Class Bar Chart",
                "Confidence Categories by Class Stacked Bar",
                "Detections Over Time Line Plot"
            ]
        },
        "class_distribution.png": {
            "description": "Large pie chart showing distribution of detected classes",
            "size": "10x8 inches"
        },
        "confidence_distribution.png": {
            "description": "Large histogram showing overall confidence distribution",
            "size": "10x6 inches"
        },
        "confidence_by_class.png": {
            "description": "Large box plot comparing confidence across classes",
            "size": "12x6 inches"
        },
        "accuracy_analysis_report.png": {
            "description": "Original analysis report graphs from accuracy_analysis.py",
            "size": "15x12 inches",
            "subplots": [
                "Class Distribution Pie Chart",
                "Confidence Distribution Histogram",
                "Class-wise Confidence Comparison",
                "Mean Confidence by Class",
                "Detection Rate Over Time",
                "Confidence Categories by Class"
            ]
        }
    }
    
    print("=" * 60)
    print("TRASH DETECTION MODEL - GRAPH INVENTORY")
    print("=" * 60)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check which files actually exist
    existing_graphs = {}
    for filename, info in graphs.items():
        if os.path.exists(filename):
            existing_graphs[filename] = info
    
    if not existing_graphs:
        print("No graph files found in the current directory.")
        return
    
    print(f"Found {len(existing_graphs)} graph file(s):")
    print("-" * 60)
    
    for i, (filename, info) in enumerate(existing_graphs.items(), 1):
        # Get file size
        file_size = os.path.getsize(filename)
        file_size_mb = file_size / (1024 * 1024)  # Convert to MB
        
        print(f"{i}. {filename}")
        print(f"   Description: {info['description']}")
        print(f"   Dimensions: {info['size']}")
        print(f"   File Size: {file_size_mb:.2f} MB")
        
        if 'subplots' in info:
            print("   Contains:")
            for j, subplot in enumerate(info['subplots'], 1):
                print(f"     {j}. {subplot}")
        
        print()

def main():
    """Main function to display graph inventory."""
    list_generated_graphs()

if __name__ == "__main__":
    main()