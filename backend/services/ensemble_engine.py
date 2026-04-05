"""Ensemble decision engine - weighted voting combining CNN + OpenCV features.

The CNN is the primary classifier (trained with test-time augmentation and calibration).
OpenCV features provide supplementary analysis and explainability.

ENHANCED: Now supports 15 security features with proper weighting based on importance.
CRITICAL features (Security Thread, Watermark, Serial Number) have higher weights.
If any critical feature fails, the note is likely FAKE.
"""

# CNN is the trained model - it should dominate the decision
CNN_WEIGHT = 0.75
OPENCV_WEIGHT = 0.25

# Enhanced feature weights for 15 security features (total = 1.0)
# Based on importance and difficulty to counterfeit
FEATURE_WEIGHTS = {
    # CRITICAL FEATURES (56% of OpenCV score)
    "security_thread": 0.225,      # 30/133 * 100 = 22.5% - Most critical
    "watermark": 0.188,            # 25/133 * 100 = 18.8% - Critical
    "serial_number": 0.150,        # 20/133 * 100 = 15.0% - Critical
    
    # IMPORTANT FEATURES (37% of OpenCV score)
    "optically_variable_ink": 0.113,  # 15/133 * 100 = 11.3%
    "latent_image": 0.090,           # 12/133 * 100 = 9.0%
    "intaglio_printing": 0.090,      # 12/133 * 100 = 9.0%
    "see_through_registration": 0.075, # 10/133 * 100 = 7.5%
    
    # SUPPORTING FEATURES (27% of OpenCV score)
    "microlettering": 0.060,        # 8/133 * 100 = 6.0%
    "fluorescence": 0.053,          # 7/133 * 100 = 5.3%
    "color_analysis": 0.053,        # 7/133 * 100 = 5.3%
    "texture_analysis": 0.038,      # 5/133 * 100 = 3.8%
    "dimensions": 0.038,            # 5/133 * 100 = 3.8%
    "identification_mark": 0.038,   # 5/133 * 100 = 3.8%
    "angular_lines": 0.023,         # 3/133 * 100 = 2.3%
}

# Threshold: score must be >= this to be classified as REAL
REAL_THRESHOLD = 0.50

# When CNN is very confident, boost its influence
HIGH_CNN_CONFIDENCE = 0.80
HIGH_CNN_BOOST = 0.10  # Additional weight added to CNN when it's very confident

# Critical features that MUST pass for a note to be REAL
# If any of these fail, the note is almost certainly FAKE
CRITICAL_FEATURES = {"security_thread", "watermark", "serial_number"}


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

    ENHANCED scoring logic:
    - CRITICAL feature failures (security thread, watermark, serial number) strongly penalize score
    - Invalid features are stronger negative signals
    - Unknown features don't penalize the result
    - Feature agreement is measured and reported
    - Critical feature failures can override CNN prediction

    Returns:
        (ensemble_score, final_result, overall_confidence, feature_agreement, critical_failures)
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
    critical_failures = []  # Track critical feature failures

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
            # Check if this is a critical feature failure
            if feature_name in CRITICAL_FEATURES:
                critical_failures.append({
                    "feature": feature_name,
                    "status": feature_status,
                    "confidence": feature_data.get("confidence", 0.0)
                })
                # Apply stronger penalty for critical feature failures
                feature_score = min(feature_score, 0.15)  # Cap at very low score

        feature_scores[feature_name] = feature_score
        opencv_weighted_score += feature_score * weight
        total_opencv_weight += weight

    # Normalize OpenCV score
    opencv_avg = (opencv_weighted_score / total_opencv_weight
                  if total_opencv_weight > 0 else 0.5)
    opencv_contrib = opencv_avg * dynamic_opencv_weight

    # Final ensemble score
    ensemble_score = round(cnn_contrib + opencv_contrib, 4)

    # CRITICAL FEATURE OVERRIDE: If any critical feature fails badly, mark as FAKE
    # This prevents false positives when CNN is wrong but critical features are missing
    if critical_failures:
        # Reduce ensemble score significantly
        penalty = 0.15 * len(critical_failures)  # 15% penalty per critical failure
        ensemble_score = round(max(0.0, ensemble_score - penalty), 4)

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

    return ensemble_score, final_result, overall_confidence, feature_agreement, critical_failures
