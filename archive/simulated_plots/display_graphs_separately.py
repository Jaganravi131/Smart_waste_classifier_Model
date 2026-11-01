"""
Display Graphs Separately for Trash Detection Model Analysis
============================================================

This script displays each generated graph file separately using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def display_graph_separately(image_path, title):
    """Display a single graph image in a separate window."""
    if not os.path.exists(image_path):
        print(f"Warning: File not found - {image_path}")
        return False
    
    # Load and display the image
    img = mpimg.imread(image_path)
    
    # Create a new figure for this image
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.title(title, fontsize=16, pad=20)
    
    # Display file information
    file_size = os.path.getsize(image_path) / 1024  # Size in KB
    plt.suptitle(f"File: {image_path} | Size: {file_size:.1f} KB", 
                 y=0.02, fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    return True

def display_all_graphs_separately():
    """Display all generated graphs separately."""
    
    # List of graphs to display with their titles
    graphs = [
        ("class_distribution.png", "Class Distribution Pie Chart"),
        ("confidence_distribution.png", "Confidence Distribution Histogram"),
        ("confidence_by_class.png", "Class-wise Confidence Box Plot"),
        ("accuracy_analysis_graphs.png", "Combined Analysis Dashboard"),
        ("accuracy_analysis_report.png", "Original Analysis Report Graphs")
    ]
    
    print("Displaying graphs separately...")
    print("=" * 50)
    
    displayed_count = 0
    
    for image_path, title in graphs:
        print(f"Attempting to display: {title}")
        if display_graph_separately(image_path, title):
            displayed_count += 1
            print(f"  ✓ Displayed: {image_path}")
        else:
            print(f"  ✗ Skipped: {image_path} (file not found)")
        print()
    
    print(f"Successfully displayed {displayed_count} graph(s).")
    print("Close each graph window to proceed to the next one.")

def main():
    """Main function to display all graphs separately."""
    print("Trash Detection Model - Graph Display Tool")
    print("=" * 45)
    
    # Check if any graph files exist
    graph_files = [
        "class_distribution.png",
        "confidence_distribution.png", 
        "confidence_by_class.png",
        "accuracy_analysis_graphs.png",
        "accuracy_analysis_report.png"
    ]
    
    existing_files = [f for f in graph_files if os.path.exists(f)]
    
    if not existing_files:
        print("No graph files found in the current directory.")
        print("Please run the analysis scripts first to generate the graphs.")
        return
    
    print(f"Found {len(existing_files)} graph file(s):")
    for i, filename in enumerate(existing_files, 1):
        file_size = os.path.getsize(filename) / 1024  # Size in KB
        print(f"  {i}. {filename} ({file_size:.1f} KB)")
    
    print()
    input("Press Enter to display graphs separately (close each window to continue)...")
    print()
    
    display_all_graphs_separately()
    
    print("All graphs have been displayed.")

if __name__ == "__main__":
    main()