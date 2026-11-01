"""
Verify All Generated Graphs for Trash Detection Model
====================================================

This script verifies that all expected graph files have been generated and 
displays their basic information.
"""

import os
from datetime import datetime

def verify_graphs():
    """Verify all expected graph files exist and display their information."""
    
    # List of expected graph files
    expected_graphs = [
        "model_accuracy_vs_epoch.png",
        "model_loss_vs_epoch.png", 
        "confidence_trend_over_time.png",
        "class_performance_comparison.png",
        "accuracy_analysis_graphs.png",
        "class_distribution.png",
        "confidence_distribution.png",
        "confidence_by_class.png",
        "accuracy_analysis_report.png",
        "model_training_visualization.png"
    ]
    
    print("=" * 60)
    print("TRASH DETECTION MODEL - GRAPH VERIFICATION REPORT")
    print("=" * 60)
    print(f"Verification Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    found_graphs = []
    missing_graphs = []
    
    for graph_file in expected_graphs:
        if os.path.exists(graph_file):
            file_size = os.path.getsize(graph_file)
            file_size_kb = file_size / 1024
            found_graphs.append((graph_file, file_size_kb))
        else:
            missing_graphs.append(graph_file)
    
    print(f"Expected Graphs: {len(expected_graphs)}")
    print(f"Found Graphs: {len(found_graphs)}")
    print(f"Missing Graphs: {len(missing_graphs)}")
    print()
    
    if found_graphs:
        print("FOUND GRAPHS:")
        print("-" * 40)
        for graph_file, file_size_kb in found_graphs:
            print(f"✓ {graph_file:<35} ({file_size_kb:>6.1f} KB)")
        print()
    
    if missing_graphs:
        print("MISSING GRAPHS:")
        print("-" * 40)
        for graph_file in missing_graphs:
            print(f"✗ {graph_file}")
        print()
    
    # Summary
    if len(found_graphs) == len(expected_graphs):
        print("✅ ALL GRAPHS SUCCESSFULLY GENERATED!")
        print("The trash detection model analysis is complete.")
    else:
        print(f"⚠️  {len(missing_graphs)} graph(s) are missing.")
        print("Some visualizations may need to be regenerated.")
    
    return len(found_graphs) == len(expected_graphs)

def main():
    """Main function to verify all graphs."""
    print("Verifying generated graphs...")
    success = verify_graphs()
    
    if success:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("All graphs have been successfully generated.")
        print("You can now review the visualizations to analyze")
        print("the trash detection model performance.")
        print()
        print("Key files:")
        print("  - FINAL_ANALYSIS_REPORT.md (This summary report)")
        print("  - model_accuracy_vs_epoch.png (Training-like accuracy graph)")
        print("  - model_loss_vs_epoch.png (Training-like loss graph)")
        print("  - confidence_trend_over_time.png (Actual confidence trends)")
        print("  - class_performance_comparison.png (Performance by class)")

if __name__ == "__main__":
    main()