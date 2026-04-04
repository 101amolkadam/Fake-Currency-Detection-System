"""CNN classifier service - trained Xception model inference."""
import numpy as np
import os
import time
import tensorflow as tf
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


def classify_currency(preprocessed_image: np.ndarray) -> tuple:
    """
    Run CNN inference for authenticity and denomination classification.
    
    Args:
        preprocessed_image: numpy array of shape (299, 299, 3), values 0-1
    
    Returns:
        (authenticity_result, denomination_result, denom_confidence, auth_confidence)
    """
    global _model
    
    if _model is None:
        return "REAL", "₹500", 0.5, 0.5
    
    try:
        img_batch = np.expand_dims(preprocessed_image, axis=0)
        
        # Model has two outputs: authenticity and denomination
        outputs = _model.predict(img_batch, verbose=0)
        
        # Handle both dict output (functional API) and list output
        if isinstance(outputs, dict):
            auth_pred = outputs.get('authenticity', outputs.get('dense_2', None))
            denom_pred = outputs.get('denomination', outputs.get('dense_3', None))
        else:
            auth_pred = outputs[0] if len(outputs) >= 2 else None
            denom_pred = outputs[1] if len(outputs) >= 2 else None
        
        if auth_pred is None or denom_pred is None:
            # Single output model
            if isinstance(outputs, dict):
                auth_pred = list(outputs.values())[0]
            else:
                auth_pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            denom_pred = np.array([[0.5, 0.5]])
        
        # Authenticity: sigmoid output (0=FAKE, 1=REAL)
        auth_score = float(np.squeeze(auth_pred))
        authenticity_result = "REAL" if auth_score >= 0.5 else "FAKE"
        auth_confidence = auth_score if auth_score >= 0.5 else 1.0 - auth_score
        
        # Denomination: softmax (we have 2 classes: 500, 2000)
        denom_probs = np.squeeze(denom_pred)
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
