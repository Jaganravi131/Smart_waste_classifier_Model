"""
Summary of Separately Generated Training Graphs
==============================================

This script provides information about the separately generated training graphs.
"""

import os
from datetime import datetime

def summarize_separate_graphs():
    """Provide a summary of the separately generated training graphs."""
    
    # Define the separately generated graphs with descriptions
    separate_graphs = {
        "accuracy_vs_epoch_separate.png": {
            "title": "Accuracy vs. Epoch (Separate)",
            "description": "Model training and validation accuracy plotted separately",
            "dimensions": "12x8 inches",
            "content": [
                "Training accuracy curve (blue line with circles)",
                "Validation accuracy curve (green dashed line with squares)",
                "Best validation accuracy annotation with arrow",
                "Grid and legend for clarity",
                "High-resolution output (300 DPI)"
            ]
        },
        "loss_vs_epoch_separate.png": {
            "title": "Loss vs. Epoch (Separate)",
            "description": "Model training and validation loss plotted separately",
            "dimensions": "12x8 inches",
            "content": [
                "Training loss curve (red line with circles)",
                "Validation loss curve (orange line with squares)",
                "Lowest validation loss annotation with arrow",
                "Grid and legend for clarity",
                "High-resolution output (300 DPI)"
            ]
        }
    }
    
    print("=" * 60)
    print("SEPARATELY GENERATED TRAINING GRAPHS SUMMARY")
    print("=" * 60)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check which files actually exist
    existing_graphs = {}
    for filename, info in separate_graphs.items():
        if os.path.exists(filename):
            existing_graphs[filename] = info
    
    if not existing_graphs:
        print("No separate graph files found in the current directory.")
        return
    
    print(f"Found {len(existing_graphs)} separate graph file(s):")
    print("-" * 60)
    
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
    """Main function to display separate graphs summary."""
    summarize_separate_graphs()
    
    print("USAGE INSTRUCTIONS:")
    print("-" * 20)
    print("To view these graphs:")
    print("1. Open the PNG files directly with any image viewer")
    print("2. The graphs are high-resolution (300 DPI) for printing")
    print("3. Each graph focuses on a single metric for clear visualization")
    print()
    print("FILES GENERATED:")
    print("-" * 15)
    print("• accuracy_vs_epoch_separate.png - Accuracy visualization")
    print("• loss_vs_epoch_separate.png - Loss visualization")

if __name__ == "__main__":
    main()