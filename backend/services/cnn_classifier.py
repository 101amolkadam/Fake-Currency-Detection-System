"""CNN classifier service - trained Xception model inference with test-time augmentation."""
import numpy as np
import os
import time
import tensorflow as tf
import cv2
from config import settings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_model = None
_input_size = 299


def load_model():
    """Load trained Xception model at startup."""
    global _model
    try:
        # Try .keras format first, then .h5
        keras_path = settings.MODEL_PATH.replace('.h5', '.keras')
        h5_path = settings.MODEL_PATH

        model_path = None
        if os.path.exists(keras_path):
            model_path = keras_path
        elif os.path.exists(h5_path):
            model_path = h5_path
        else:
            print(f"[WARN] No model found at {keras_path} or {h5_path}. OpenCV-only mode.")
            return False

        print(f"[INFO] Loading model from {model_path}...")
        _model = tf.keras.models.load_model(model_path, safe_mode=False)
        # Warm up
        dummy = np.zeros((1, _input_size, _input_size, 3), dtype=np.float32)
        _model.predict(dummy, verbose=0)
        print(f"[INFO] Model loaded and warmed up successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        _model = None
        return False


def get_model():
    return _model


def is_model_loaded():
    return _model is not None


def _apply_tta_augmentations(image: np.ndarray) -> list:
    """
    Generate augmented versions of the input image for Test-Time Augmentation (TTA).
    
    Creates 7 variations to improve prediction robustness:
    1. Original
    2. Horizontal flip
    3. Rotation +10°
    4. Rotation -10°
    5. Brightness +10%
    6. Brightness -10%
    7. Slight zoom (1.1x)
    """
    augmented = [image]  # Original
    
    # Horizontal flip
    augmented.append(np.fliplr(image).copy())
    
    # Rotation +10°
    rows, cols = image.shape[:2]
    M_plus = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1.0)
    rotated_plus = cv2.warpAffine(image, M_plus, (cols, rows))
    augmented.append(rotated_plus)
    
    # Rotation -10°
    M_minus = cv2.getRotationMatrix2D((cols/2, rows/2), -10, 1.0)
    rotated_minus = cv2.warpAffine(image, M_minus, (cols, rows))
    augmented.append(rotated_minus)
    
    # Brightness +10%
    brighter = np.clip(image * 1.1, 0, 1)
    augmented.append(brighter)
    
    # Brightness -10%
    darker = np.clip(image * 0.9, 0, 1)
    augmented.append(darker)
    
    # Zoom 1.1x
    zoom_factor = 1.1
    M_zoom = cv2.getRotationMatrix2D((cols/2, rows/2), 0, zoom_factor)
    zoomed = cv2.warpAffine(image, M_zoom, (cols, rows))
    augmented.append(zoomed)
    
    return augmented


def _calibrate_confidence(raw_confidence: float) -> float:
    """
    Apply temperature scaling to calibrate model confidence.
    
    This prevents overconfident predictions and makes confidence scores
    more reflective of true accuracy. Uses temperature T=1.5 to make
    confidence more conservative and realistic.
    """
    # Temperature scaling (T=1.5 makes confidence less extreme)
    temperature = 1.5
    
    # Convert to logit space
    epsilon = 1e-7
    raw_confidence = np.clip(raw_confidence, epsilon, 1 - epsilon)
    logit = np.log(raw_confidence / (1 - raw_confidence))
    
    # Apply temperature
    calibrated_logit = logit / temperature
    
    # Convert back to probability
    calibrated = 1 / (1 + np.exp(-calibrated_logit))
    
    return calibrated


def classify_currency(preprocessed_image: np.ndarray, use_tta: bool = True) -> tuple:
    """
    Run CNN inference for authenticity and denomination classification.
    Uses Test-Time Augmentation (TTA) for more robust predictions.

    Args:
        preprocessed_image: numpy array of shape (299, 299, 3), values normalized for Xception
        use_tta: Whether to use test-time augmentation (default: True)

    Returns:
        (authenticity_result, denomination_result, denom_confidence, auth_confidence)
    """
    global _model

    if _model is None:
        return "REAL", "₹500", 0.5, 0.5

    try:
        if use_tta:
            # Test-Time Augmentation: predict on multiple augmented versions
            augmented_images = _apply_tta_augmentations(preprocessed_image)
            
            authenticity_scores = []
            denomination_probs = []
            
            for aug_img in augmented_images:
                img_batch = np.expand_dims(aug_img, axis=0)
                outputs = _model.predict(img_batch, verbose=0)
                
                # Handle both dict output (functional API) and list output
                if isinstance(outputs, dict):
                    auth_pred = outputs.get('authenticity', outputs.get('dense_2', None))
                    denom_pred = outputs.get('denomination', outputs.get('dense_3', None))
                else:
                    auth_pred = outputs[0] if len(outputs) >= 2 else None
                    denom_pred = outputs[1] if len(outputs) >= 2 else None
                
                if auth_pred is not None:
                    authenticity_scores.append(float(np.squeeze(auth_pred)))
                
                if denom_pred is not None:
                    denom_probs.append(np.squeeze(denom_pred))
            
            # Average predictions across augmentations
            auth_score = np.mean(authenticity_scores)
            auth_std = np.std(authenticity_scores)
            
            if denom_probs:
                denom_probs = np.mean(denom_probs, axis=0)
            else:
                denom_probs = np.array([0.5, 0.5])
        else:
            # Single prediction (no TTA)
            img_batch = np.expand_dims(preprocessed_image, axis=0)
            outputs = _model.predict(img_batch, verbose=0)
            
            if isinstance(outputs, dict):
                auth_pred = outputs.get('authenticity', outputs.get('dense_2', None))
                denom_pred = outputs.get('denomination', outputs.get('dense_3', None))
            else:
                auth_pred = outputs[0] if len(outputs) >= 2 else None
                denom_pred = outputs[1] if len(outputs) >= 2 else None
            
            if auth_pred is None or denom_pred is None:
                # Single output model fallback
                if isinstance(outputs, dict):
                    auth_pred = list(outputs.values())[0]
                else:
                    auth_pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                denom_pred = np.array([[0.5, 0.5]])
            
            auth_score = float(np.squeeze(auth_pred))
            auth_std = 0.0  # No variation
            denom_probs = np.squeeze(denom_pred)
        
        # Apply calibration to make confidence more realistic
        auth_score = _calibrate_confidence(auth_score)
        
        # Authenticity: sigmoid output (0=FAKE, 1=REAL)
        authenticity_result = "REAL" if auth_score >= 0.5 else "FAKE"
        auth_confidence = auth_score if auth_score >= 0.5 else 1.0 - auth_score
        
        # Reduce confidence if TTA has high variance (unreliable prediction)
        if use_tta and auth_std > 0.1:
            # High variance means model is uncertain - reduce confidence
            penalty = min(0.2, auth_std * 0.5)
            auth_confidence = max(0.5, auth_confidence - penalty)
        
        # Denomination: softmax (we have 2 classes: 500, 2000)
        if denom_probs.ndim == 0:
            denom_probs = np.array([0.5, 0.5])
        
        denom_names = ["₹500", "₹2000"]
        denom_idx = int(np.argmax(denom_probs))
        denomination_result = denom_names[denom_idx] if denom_idx < len(denom_names) else "₹500"
        denom_confidence = float(np.max(denom_probs))

        return (
            authenticity_result,
            denomination_result,
            round(denom_confidence, 4),
            round(auth_confidence, 4)
        )
    except Exception as e:
        print(f"[ERROR] CNN classification failed: {e}")
        return "REAL", "₹500", 0.5, 0.5
