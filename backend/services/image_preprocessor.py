"""Image preprocessing utilities - PyTorch compatible, no TensorFlow dependency.

╔══════════════════════════════════════════════════════════════════════╗
║  IMAGE PREPROCESSING PIPELINE                                       ║
║  ┌────────────────────────────────────────────────────────────┐    ║
║  │  Input: Base64 Data URI (JPEG/PNG/WebP)                    │    ║
║  └────────────────────┬───────────────────────────────────────┘    ║
║                       │                                             ║
║                       ▼                                             ║
║  ┌────────────────────────────────────────────────────────┐       ║
║  │  decode_base64_image()                                  │       ║
║  │  ├─ Parse MIME type header                              │       ║
║  │  ├─ Validate size (< 10MB)                              │       ║
║  │  └─ cv2.imdecode() → BGR numpy array                   │       ║
║  └────────────────────┬───────────────────────────────────┘       ║
║                       │                                             ║
║                       ▼                                             ║
║  ┌────────────────────────────────────────────────────────┐       ║
║  │  preprocess_image() - THREE PARALLEL STREAMS           │       ║
║  │                                                          │       ║
║  │  Stream 1: CNN Input (MobileNetV3)                     │       ║
║  │  ├─ Resize: Any → 224×224                               │       ║
║  │  ├─ Color: BGR → RGB                                    │       ║
║  │  ├─ Scale: [0, 255] → [0, 1]                           │       ║
║  │  └─ Normalize: ImageNet mean/std                       │       ║
║  │                                                          │       ║
║  │  Stream 2: Denoised (OpenCV Features)                  │       ║
║  │  └─ fastNlMeansDenoisingColored()                      │       ║
║  │     ├─ h=10, hColor=10                                 │       ║
║  │     ├─ templateWindowSize=7                            │       ║
║  │     └─ searchWindowSize=21                             │       ║
║  │                                                          │       ║
║  │  Stream 3: Enhanced (Edge/Texture Analysis)            │       ║
║  │  ├─ BGR → Grayscale                                     │       ║
║  │  └─ CLAHE (Contrast Limited Adaptive HE)               │       ║
║  │     ├─ clipLimit=2.0                                   │       ║
║  │     └─ tileGridSize=(8, 8)                             │       ║
║  └────────────────────────────────────────────────────────┘       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import base64
from typing import Tuple

import cv2
import numpy as np


def decode_base64_image(base64_string: str) -> Tuple[np.ndarray, str]:
    """Decode a base64 data URI string to an OpenCV image.

    Args:
        base64_string: A data URI string in the format ``data:image/<type>;base64,<data>``.

    Returns:
        A tuple of ``(image, mime_type)`` where ``image`` is a NumPy array (BGR format)
        and ``mime_type`` is the MIME type string (e.g. ``"image/jpeg"``).

    Raises:
        ValueError: If the MIME type is unsupported, the image exceeds 10 MB,
            or the image cannot be decoded.
    """
    header, encoded_data = base64_string.split(",", 1)

    mime_type = header.split(":")[1].split(";")[0]
    if mime_type not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError(f"Unsupported image type: {mime_type}")

    image_bytes = base64.b64decode(encoded_data)

    max_size = 10 * 1024 * 1024  # 10 MB
    if len(image_bytes) > max_size:
        raise ValueError(f"Image size ({len(image_bytes) / 1024 / 1024:.1f}MB) exceeds 10MB limit")

    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image")

    return image, mime_type


def preprocess_image_for_mobilenet(image: np.ndarray) -> np.ndarray:
    """Preprocess an image for MobileNetV3-Large CNN inference.

    MobileNetV3 preprocessing pipeline:
    1. Resize to 224 x 224.
    2. Convert BGR to RGB.
    3. Scale to ``[0, 1]`` and normalize with ImageNet mean/std.

    Args:
        image: Input image as a BGR NumPy array.

    Returns:
        Normalized RGB image of shape ``(224, 224, 3)`` with ``float32`` dtype.
    """
    resized = cv2.resize(image, (224, 224))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    rgb_image = rgb_image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb_image - mean) / std

    return normalized


def preprocess_image_for_xception(image: np.ndarray) -> np.ndarray:
    """Preprocess an image for Xception CNN inference.

    Xception preprocessing pipeline:
    1. Resize to 299 x 299.
    2. Convert BGR to RGB.
    3. Scale pixels to ``[-1, 1]`` range (equivalent to
       ``tf.keras.applications.xception.preprocess_input``).

    Args:
        image: Input image as a BGR NumPy array.

    Returns:
        Normalized RGB image of shape ``(299, 299, 3)`` with ``float32`` dtype.
    """
    resized = cv2.resize(image, (299, 299))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Xception uses: (pixel / 127.5) - 1.0  =>  maps [0, 255] to [-1, 1]
    normalized = (rgb_image.astype(np.float32) / 127.5) - 1.0

    return normalized


def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess an image for both CNN and OpenCV security feature analysis.

    Args:
        image: Input image as a BGR NumPy array.

    Returns:
        A tuple of ``(cnn_input, denoised, enhanced)``:

        - ``cnn_input``: Normalized 224 x 224 ``float32`` array ready for MobileNetV3 CNN.
        - ``denoised``: Full-size denoised image for OpenCV feature detection.
        - ``enhanced``: CLAHE-enhanced grayscale image for edge/texture analysis.
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
