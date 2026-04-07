"""Image preprocessing utilities - PyTorch compatible, no TensorFlow dependency."""
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


def preprocess_image_for_mobilenet(image: np.ndarray) -> np.ndarray:
    """Preprocess image for MobileNetV3-Large CNN.

    MobileNetV3 preprocessing:
    1. Resize to 224x224
    2. Convert BGR to RGB
    3. Scale to [0, 1] and normalize with ImageNet stats
    """
    # Resize to MobileNetV3 input size
    resized = cv2.resize(image, (224, 224))

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize with ImageNet mean/std
    rgb_image = rgb_image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb_image - mean) / std

    return normalized


def preprocess_image_for_xception(image: np.ndarray) -> np.ndarray:
    """Preprocess image for Xception CNN without TensorFlow.

    Xception preprocessing:
    1. Resize to 299x299
    2. Convert BGR to RGB
    3. Scale pixels to [-1, 1] range (Xception expects this)
    """
    # Resize to Xception input size
    resized = cv2.resize(image, (299, 299))

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [-1, 1] range (same as tf.keras.applications.xception.preprocess_input)
    # Xception uses: (pixel / 127.5) - 1.0
    normalized = (rgb_image.astype(np.float32) / 127.5) - 1.0

    return normalized


def preprocess_image(image: np.ndarray) -> tuple:
    """Preprocess image for CNN and OpenCV analysis.

    Returns:
        (cnn_input, denoised, enhanced)
        - cnn_input: properly normalized 224x224 float32 array for MobileNetV3 CNN
        - denoised: denoised full-size image for OpenCV analysis
        - enhanced: CLAHE-enhanced grayscale image
    """
    # CNN preprocessing (MobileNetV3-Large)
    cnn_input = preprocess_image_for_mobilenet(image)

    # Denoise for OpenCV analysis
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # CLAHE enhancement for OpenCV features
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return cnn_input, denoised, enhanced
