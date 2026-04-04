"""Image preprocessing utilities - base64 decoding and preprocessing."""
import base64
import cv2
import numpy as np


def decode_base64_image(base64_string: str) -> tuple:
    """Decode a base64 data URI string to an OpenCV image.
    
    Returns:
        (image, mime_type) where image is a numpy array and mime_type is str
    """
    header, encoded_data = base64_string.split(",", 1)
    
    mime_type = header.split(":")[1].split(";")[0]
    if mime_type not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError(f"Unsupported image type: {mime_type}")
    
    image_bytes = base64.b64decode(encoded_data)
    
    max_size = 10 * 1024 * 1024  # 10MB
    if len(image_bytes) > max_size:
        raise ValueError(f"Image size ({len(image_bytes) / 1024 / 1024:.1f}MB) exceeds 10MB limit")
    
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    return image, mime_type


def preprocess_image(image: np.ndarray) -> tuple:
    """Preprocess image for CNN and OpenCV analysis.
    
    Returns:
        (cnn_input, denoised, enhanced)
        - cnn_input: normalized 299x299 float32 array for CNN
        - denoised: denoised full-size image for OpenCV analysis
        - enhanced: CLAHE-enhanced grayscale image
    """
    resized = cv2.resize(image, (299, 299))
    normalized = resized.astype(np.float32) / 255.0
    
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return normalized, denoised, enhanced
