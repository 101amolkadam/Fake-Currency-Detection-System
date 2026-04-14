"""Ensemble decision engine -- weighted voting combining CNN + OpenCV features.

The CNN is the primary classifier (trained with test-time augmentation and
calibration).  OpenCV features provide supplementary analysis and explainability.

ENHANCED: Now supports 15 security features with proper weighting based on
importance.  CRITICAL features (Security Thread, Watermark, Serial Number) have
higher weights.  If any critical feature fails, the note is likely FAKE.
"""

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CNN_WEIGHT: float = 0.75
OPENCV_WEIGHT: float = 0.25

# Enhanced feature weights for 15 security features (total = 1.0)
# Based on importance and difficulty to counterfeit
FEATURE_WEIGHTS: Dict[str, float] = {
    # CRITICAL FEATURES (56% of OpenCV score)
    "security_thread": 0.225,           # 30/133 * 100 = 22.5% -- Most critical
    "watermark": 0.188,                 # 25/133 * 100 = 18.8% -- Critical
    "serial_number": 0.150,             # 20/133 * 100 = 15.0% -- Critical

    # IMPORTANT FEATURES (37% of OpenCV score)
    "optically_variable_ink": 0.113,    # 15/133 * 100 = 11.3%
    "latent_image": 0.090,              # 12/133 * 100 = 9.0%
    "intaglio_printing": 0.090,         # 12/133 * 100 = 9.0%
    "see_through_registration": 0.075,  # 10/133 * 100 = 7.5%

    # SUPPORTING FEATURES (27% of OpenCV score)
    "microlettering": 0.060,            # 8/133 * 100 = 6.0%
    "fluorescence": 0.053,              # 7/133 * 100 = 5.3%
    "color_analysis": 0.053,            # 7/133 * 100 = 5.3%
    "texture_analysis": 0.038,          # 5/133 * 100 = 3.8%
    "dimensions": 0.038,                # 5/133 * 100 = 3.8%
    "identification_mark": 0.038,       # 5/133 * 100 = 3.8%
    "angular_lines": 0.023,             # 3/133 * 100 = 2.3%
}

REAL_THRESHOLD: float = 0.50

# When CNN is very confident, boost its influence
HIGH_CNN_CONFIDENCE: float = 0.80
HIGH_CNN_BOOST: float = 0.10

# Critical features that MUST pass for a note to be REAL.
# If any of these fail, the note is almost certainly FAKE.
CRITICAL_FEATURES: frozenset = frozenset({"security_thread", "watermark", "serial_number"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_feature_score(feature_data: Dict[str, Any]) -> float:
    """Compute a normalised score for a single OpenCV feature.

    Args:
        feature_data: Dictionary with at least ``"status"`` and ``"confidence"`` keys.

    Returns:
        A float in ``[0, 1]`` where higher means the feature agrees with the note
        being genuine.
    """
    feature_confidence: float = feature_data.get("confidence", 0.5)
    feature_status: str = feature_data.get("status", "unknown")

    if feature_status in ("present", "match", "normal", "valid"):
        return feature_confidence
    if feature_status == "unknown":
        return 0.5
    if feature_status == "invalid":
        return max(0.1, 1.0 - feature_confidence)
    # Failed / missing feature -- moderate negative signal
    return max(0.15, feature_confidence * 0.4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ensemble_score(
    cnn_result: str,
    cnn_confidence: float,
    features: Dict[str, Any],
) -> Tuple[float, str, float, float, List[Dict[str, Any]]]:
    """Compute a weighted ensemble score combining CNN and OpenCV feature results.

    Uses dynamic weighting: when the CNN is highly confident it receives more
    weight.  OpenCV features provide supplementary validation.

    Scoring logic:
    * CRITICAL feature failures (security thread, watermark, serial number)
      strongly penalise the score.
    * Invalid features are stronger negative signals.
    * Unknown features do not penalise the result.
    * Critical feature failures can override the CNN prediction.

    Args:
        cnn_result: ``"REAL"`` or ``"FAKE"``.
        cnn_confidence: CNN confidence in ``[0, 1]``.
        features: Dictionary mapping feature names to their analysis dicts.

    Returns:
        A 5-tuple of
        ``(ensemble_score, final_result, overall_confidence, feature_agreement,
          critical_failures)``.
    """
    # Dynamic CNN weighting: boost when CNN is very confident
    dynamic_cnn_weight: float = CNN_WEIGHT
    if cnn_confidence >= HIGH_CNN_CONFIDENCE:
        dynamic_cnn_weight = min(0.90, CNN_WEIGHT + HIGH_CNN_BOOST)

    dynamic_opencv_weight: float = 1.0 - dynamic_cnn_weight

    # CNN contribution
    cnn_score = 1.0 if cnn_result == "REAL" else 0.0
    cnn_contrib: float = cnn_score * cnn_confidence * dynamic_cnn_weight

    # OpenCV features contribution
    opencv_weighted_score: float = 0.0
    total_opencv_weight: float = 0.0
    features_passed: int = 0
    features_failed: int = 0
    critical_failures: List[Dict[str, Any]] = []

    for feature_name, weight in FEATURE_WEIGHTS.items():
        feature_data: Dict[str, Any] = features.get(feature_name, {})
        feature_score: float = compute_feature_score(feature_data)
        feature_status: str = feature_data.get("status", "unknown")

        if feature_status in ("present", "match", "normal", "valid"):
            features_passed += 1
        elif feature_status == "unknown":
            pass  # neutral -- no penalty
        else:
            features_failed += 1
            if feature_name in CRITICAL_FEATURES:
                critical_failures.append({
                    "feature": feature_name,
                    "status": feature_status,
                    "confidence": feature_data.get("confidence", 0.0),
                })
                feature_score = min(feature_score, 0.15)

        opencv_weighted_score += feature_score * weight
        total_opencv_weight += weight

    opencv_avg: float = (
        opencv_weighted_score / total_opencv_weight if total_opencv_weight > 0 else 0.5
    )
    opencv_contrib: float = opencv_avg * dynamic_opencv_weight

    ensemble_score: float = round(cnn_contrib + opencv_contrib, 4)

    # CRITICAL FEATURE OVERRIDE
    if critical_failures:
        penalty: float = 0.15 * len(critical_failures)
        ensemble_score = round(max(0.0, ensemble_score - penalty), 4)

    final_result: str = "REAL" if ensemble_score >= REAL_THRESHOLD else "FAKE"

    if final_result == "REAL":
        overall_confidence: float = round(min(1.0, ensemble_score), 4)
    else:
        overall_confidence = round(max(0.0, 1.0 - ensemble_score), 4)

    # Feature agreement
    total_checked: int = features_passed + features_failed
    if total_checked > 0:
        feature_agreement: float = (
            features_passed / total_checked
            if final_result == "REAL"
            else features_failed / total_checked
        )
    else:
        feature_agreement = 0.5

    return ensemble_score, final_result, overall_confidence, feature_agreement, critical_failures
