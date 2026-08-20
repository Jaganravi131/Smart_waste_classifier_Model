"""
Journal Submission Checklist for Trash Detection Model
=====================================================

This script verifies all files needed for journal submission are available
and provides information about each file's purpose.
"""

import os
from datetime import datetime

def journal_submission_checklist():
    """Verify all files needed for journal submission."""
    
    # Files needed for journal submission
    submission_files = {
        "Primary Graphs": [
            ("accuracy_vs_epoch_separate.png", "Model accuracy progression during training"),
            ("loss_vs_epoch_separate.png", "Model loss reduction during training")
        ],
        "Documentation": [
            ("GRAPH_DESCRIPTIONS_FOR_JOURNAL.md", "Detailed descriptions for figure captions"),
            ("JOURNAL_SUBMISSION_SUPPORT.md", "Comprehensive supporting information"),
            ("journal_graph_descriptions.py", "Programmatic access to descriptions")
        ],
        "Analysis Reports": [
            ("FINAL_ANALYSIS_REPORT.md", "Complete model performance analysis"),
            ("ACCURACY_ANALYSIS_SUMMARY.md", "Key findings summary")
        ],
        "Supplementary Materials": [
            ("class_performance_comparison.png", "Performance comparison by waste class"),
            ("confidence_trend_over_time.png", "Model confidence stability over time"),
            ("model_training_visualization.png", "Combined training visualization")
        ]
    }
    
    print("=" * 70)
    print("JOURNAL SUBMISSION CHECKLIST")
    print("=" * 70)
    print(f"Prepared: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    total_files = 0
    found_files = 0
    
    for category, files in submission_files.items():
        print(f"{category}:")
        print("-" * (len(category) + 1))
        
        category_total = 0
        category_found = 0
        
        for filename, description in files:
            total_files += 1
            category_total += 1
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                file_size_kb = file_size / 1024
                print(f"  ✓ {filename:<35} ({file_size_kb:>6.1f} KB)")
                print(f"    Description: {description}")
                found_files += 1
                category_found += 1
            else:
                print(f"  ✗ {filename:<35} (MISSING)")
                print(f"    Description: {description}")
        
        print(f"  Category Summary: {category_found}/{category_total} files found")
        print()
    
    print("=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"Total Files Required: {total_files}")
    print(f"Files Available:      {found_files}")
    print(f"Completion Rate:      {found_files/total_files*100:.1f}%")
    print()
    
    if found_files == total_files:
        print("🎉 ALL FILES READY FOR JOURNAL SUBMISSION!")
        print()
        print("PRIMARY FILES FOR IMMEDIATE USE:")
        print("1. accuracy_vs_epoch_separate.png - Figure 1")
        print("2. loss_vs_epoch_separate.png - Figure 2")
        print("3. GRAPH_DESCRIPTIONS_FOR_JOURNAL.md - Figure captions")
        print()
        print("SUPPORTING FILES:")
        print("- JOURNAL_SUBMISSION_SUPPORT.md - Comprehensive documentation")
        print("- FINAL_ANALYSIS_REPORT.md - Detailed results")
        print("- class_performance_comparison.png - Supplementary figure")
    else:
        missing_count = total_files - found_files
        print(f"⚠️  {missing_count} file(s) are missing.")
        print("Please run the respective generation scripts to create missing files.")
    
    return found_files == total_files

def main():
    """Main function for journal submission checklist."""
    print("Verifying journal submission files...")
    print()
    success = journal_submission_checklist()
    
    if success:
        print("\n" + "=" * 70)
        print("JOURNAL SUBMISSION READY")
        print("=" * 70)
        print("All required files for journal submission are available.")
        print()
        print("RECOMMENDED NEXT STEPS:")
        print("1. Review GRAPH_DESCRIPTIONS_FOR_JOURNAL.md for figure captions")
        print("2. Examine JOURNAL_SUBMISSION_SUPPORT.md for comprehensive info")
        print("3. Use journal_graph_descriptions.py for programmable access")
        print("4. Include the two primary PNG files in your manuscript")
        print()
        print("FILES TO CITE:")
        print("- accuracy_vs_epoch_separate.png (252.5 KB)")
        print("- loss_vs_epoch_separate.png (238.8 KB)")

if __name__ == "__main__":
    main()