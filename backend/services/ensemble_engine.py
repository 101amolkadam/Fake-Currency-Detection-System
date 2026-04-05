"""Ensemble decision engine - weighted voting combining CNN + OpenCV features.

The CNN is the primary classifier (trained with test-time augmentation and calibration).
OpenCV features provide supplementary analysis and explainability.
"""

# CNN is the trained model - it should dominate the decision
CNN_WEIGHT = 0.80
OPENCV_WEIGHT = 0.20

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
HIGH_CNN_CONFIDENCE = 0.80
HIGH_CNN_BOOST = 0.10  # Additional weight added to CNN when it's very confident


def compute_feature_score(feature_data: dict) -> float:
    """
    Compute score for a single OpenCV feature.
    
    Returns a score between 0 and 1 based on status and confidence.
    """
    feature_confidence = feature_data.get("confidence", 0.5)
    feature_status = feature_data.get("status", "unknown")
    
    # Score based on status with nuanced handling
    if feature_status in ("present", "match", "normal", "valid"):
        # Good feature - use confidence directly
        return feature_confidence
    elif feature_status == "unknown":
        # Unknown/uncertain - neutral score
        return 0.5
    elif feature_status == "invalid":
        # Invalid (e.g., serial number format wrong) - strong negative signal
        return max(0.1, 1.0 - feature_confidence)
    else:
        # Failed/missing feature - moderate negative signal
        return max(0.15, feature_confidence * 0.4)


def compute_ensemble_score(
    cnn_result: str,
    cnn_confidence: float,
    features: dict,
) -> tuple:
    """Compute weighted ensemble score combining CNN + OpenCV features.

    Uses weighted voting with dynamic adjustment: when CNN is highly confident,
    it gets more weight. OpenCV features provide supplementary validation.
    
    Improved scoring logic:
    - Invalid features (e.g., bad serial number) are stronger negative signals
    - Unknown features don't penalize the result
    - Feature agreement is measured and reported

    Returns:
        (ensemble_score, final_result, overall_confidence, feature_agreement)
    """
    # Dynamic CNN weighting: boost when CNN is very confident
    dynamic_cnn_weight = CNN_WEIGHT
    if cnn_confidence >= HIGH_CNN_CONFIDENCE:
        dynamic_cnn_weight = min(0.90, CNN_WEIGHT + HIGH_CNN_BOOST)
        dynamic_opencv_weight = 1.0 - dynamic_cnn_weight
    else:
        dynamic_opencv_weight = OPENCV_WEIGHT

    # CNN contribution
    cnn_score = 1.0 if cnn_result == "REAL" else 0.0
    cnn_contrib = cnn_score * cnn_confidence * dynamic_cnn_weight

    # OpenCV features contribution with improved scoring
    opencv_weighted_score = 0.0
    total_opencv_weight = 0.0
    features_passed = 0
    features_failed = 0
    features_unknown = 0
    feature_scores = {}

    for feature_name, weight in FEATURE_WEIGHTS.items():
        feature_data = features.get(feature_name, {})
        feature_score = compute_feature_score(feature_data)
        feature_status = feature_data.get("status", "unknown")
        
        # Track feature statistics
        if feature_status in ("present", "match", "normal", "valid"):
            features_passed += 1
        elif feature_status == "unknown":
            features_unknown += 1
        else:
            features_failed += 1
        
        feature_scores[feature_name] = feature_score
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
    
    # Feature agreement metric (how many features agree with the result)
    total_checked = features_passed + features_failed
    if total_checked > 0:
        if final_result == "REAL":
            feature_agreement = features_passed / total_checked
        else:
            feature_agreement = features_failed / total_checked
    else:
        feature_agreement = 0.5

    return ensemble_score, final_result, overall_confidence
