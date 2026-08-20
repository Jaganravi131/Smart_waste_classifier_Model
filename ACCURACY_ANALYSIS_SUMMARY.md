# Trash Detection Model - Accuracy Analysis Summary

## Overview
This report provides a comprehensive analysis of the trash detection model's performance based on 13,352 frames of detection data collected over a 13-minute period.

## Key Findings

### Overall Performance
- **Detection Rate**: 95.94% (12,810 frames with detections out of 13,352 total frames)
- **No-Detection Frames**: 4.06% (542 frames)
- **Total Detections Analyzed**: 12,810

### Class Distribution
The model shows a strong bias toward detecting paper objects:
1. **Paper**: 12,601 detections (94.38%)
2. **Glass**: 149 detections (1.12%)
3. **Cardboard**: 26 detections (0.19%)
4. **Plastic**: 24 detections (0.18%)
5. **Metal**: 10 detections (0.07%)

### Confidence Analysis

#### Mean Confidence by Class
| Class | Mean Confidence | Standard Deviation |
|-------|----------------|-------------------|
| Cardboard | 0.5870 | 0.1228 |
| Metal | 0.5709 | 0.0731 |
| Glass | 0.4966 | 0.0756 |
| Paper | 0.4883 | 0.0572 |
| Plastic | 0.4451 | 0.0545 |

#### Overall Confidence Distribution
- **High Confidence (>0.8)**: 100 detections (0.78%)
- **Medium Confidence (0.5-0.8)**: 3,625 detections (28.30%)
- **Low Confidence (<0.5)**: 9,085 detections (70.92%)

### Performance Insights

#### Best Performing Classes
1. **Cardboard** - Highest mean confidence (0.5870)
2. **Metal** - Second highest mean confidence (0.5709)
3. **Glass** - Moderate confidence with good distribution (0.4966)

#### Areas for Improvement
1. **Plastic** - Lowest mean confidence (0.4451) with 91.7% of detections in the low confidence range
2. **Paper** - While frequently detected, 71.2% of detections fall in the low confidence range

### Recommendations

1. **Model Retraining**: 
   - Focus on improving plastic detection accuracy
   - Consider data augmentation for underrepresented classes (metal, cardboard)

2. **Confidence Threshold Adjustment**:
   - Current model produces many low-confidence detections
   - Consider implementing a higher confidence threshold for actionable detections

3. **Class Balancing**:
   - The extreme imbalance in class distribution (paper dominates) may affect model performance
   - Collect more diverse training data for underrepresented classes

4. **Temporal Analysis**:
   - Detection patterns remain consistent over time
   - No significant performance degradation observed during the test period

## Visualization
See `accuracy_analysis_report.png` for detailed charts including:
- Class distribution pie chart
- Confidence distribution histogram
- Class-wise confidence comparison
- Detection trends over time
- Confidence categories by class

## Conclusion
The model demonstrates strong detection capabilities with a high overall detection rate of 95.94%. However, the confidence levels are generally low, with 70.92% of detections falling below 0.5 confidence. The extreme class imbalance (paper comprising 94.38% of detections) suggests a need for more balanced training data. Cardboard and metal classes show the highest confidence levels and could serve as benchmarks for improving other classes.