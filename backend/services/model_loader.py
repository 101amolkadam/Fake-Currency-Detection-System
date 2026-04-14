"""Model loading and management.

Loads the CNN classifier at application startup and exposes convenience
functions for checking model state.
"""

from services.cnn_classifier import load_model, get_model, is_model_loaded

# Load model at startup so it is ready for incoming requests.
load_model()

__all__ = ["load_model", "get_model", "is_model_loaded"]
