"""Ensemble decision engine - weighted voting combining CNN + OpenCV features.

The CNN is the primary classifier (trained with 100% val accuracy).
OpenCV features provide supplementary analysis and explainability.
"""

# CNN is the trained model - it should dominate the decision
CNN_WEIGHT = 0.75
OPENCV_WEIGHT = 0.25

FEATURE_WEIGHTS = {
    "watermark": 0.20,
    "security_thread": 0.25,
    "color_analysis": 0.20,
    "texture_analysis": 0.15,
    "serial_number": 0.10,
    "dimensions": 0.10,
}

# Threshold: score must be >= this to be classified as REAL
REAL_THRESHOLD = 0.50

# When CNN is very confident, boost its influence
HIGH_CNN_CONFIDENCE = 0.85
HIGH_CNN_BOOST = 0.15  # Additional weight added to CNN when it's very confident


def compute_ensemble_score(
    cnn_result: str,
    cnn_confidence: float,
    features: dict,
) -> tuple:
    """Compute weighted ensemble score combining CNN + OpenCV features.
    
    Uses weighted voting with dynamic adjustment: when CNN is highly confident,
    it gets more weight. OpenCV features provide supplementary validation.
    
    Returns:
        (ensemble_score, final_result, overall_confidence)
    """
    # Dynamic CNN weighting: boost when CNN is very confident
    dynamic_cnn_weight = CNN_WEIGHT
    if cnn_confidence >= HIGH_CNN_CONFIDENCE:
        dynamic_cnn_weight = min(0.85, CNN_WEIGHT + HIGH_CNN_BOOST)
        dynamic_opencv_weight = 1.0 - dynamic_cnn_weight
    else:
        dynamic_opencv_weight = OPENCV_WEIGHT
    
    # CNN contribution
    cnn_score = 1.0 if cnn_result == "REAL" else 0.0
    cnn_contrib = cnn_score * cnn_confidence * dynamic_cnn_weight
    
    # OpenCV features contribution
    opencv_weighted_score = 0.0
    total_opencv_weight = 0.0
    features_passed = 0
    features_checked = 0
    
    for feature_name, weight in FEATURE_WEIGHTS.items():
        feature_data = features.get(feature_name, {})
        feature_confidence = feature_data.get("confidence", 0.5)
        feature_status = feature_data.get("status", "unknown")
        
        features_checked += 1
        
        # Score based on status - use raw confidence for passed features
        if feature_status in ("present", "match", "normal", "valid", "correct"):
            feature_score = feature_confidence
            features_passed += 1
        elif feature_status == "unknown":
            # Unknown features don't penalize
            feature_score = 0.5
        else:
            # Failed features: reduce but don't zero out
            feature_score = max(0.1, feature_confidence * 0.5)
        
        opencv_weighted_score += feature_score * weight
        total_opencv_weight += weight
    
    # Normalize OpenCV score
    opencv_avg = (opencv_weighted_score / total_opencv_weight
                  if total_opencv_weight > 0 else 0.5)
    opencv_contrib = opencv_avg * dynamic_opencv_weight
    
    # Final ensemble score
    ensemble_score = round(cnn_contrib + opencv_contrib, 4)
    
    # Classification
    final_result = "REAL" if ensemble_score >= REAL_THRESHOLD else "FAKE"
    
    # Confidence: distance from threshold
    if final_result == "REAL":
        overall_confidence = round(min(1.0, ensemble_score), 4)
    else:
        overall_confidence = round(max(0.0, 1.0 - ensemble_score), 4)
    
    return ensemble_score, final_result, overall_confidence
