"""Model loading and management."""
from services.cnn_classifier import load_model, get_model, is_model_loaded

# Load model at startup
load_model()
