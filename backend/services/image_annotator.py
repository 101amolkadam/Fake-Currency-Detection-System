"""Generate annotated images and thumbnails, encode as base64."""

from typing import Any, Dict

import base64

import cv2
import numpy as np


def generate_annotated_image(
    image: np.ndarray,
    analysis_results: Dict[str, Any],
) -> str:
    """Draw security-feature annotations on a copy of *image* and return as a JPEG data URI.

    The annotations include bounding boxes and status labels for watermark,
    security thread, color analysis, texture analysis, serial number, and an
    overall-result banner.

    Args:
        image: Input BGR image as a NumPy array.
        analysis_results: Dictionary produced by the analysis pipeline containing
            keys such as ``"watermark"``, ``"security_thread"``,
            ``"color_analysis"``, ``"texture_analysis"``, ``"serial_number"``,
            ``"overall_result"``, and ``"ensemble_score"``.

    Returns:
        A base64-encoded JPEG data URI string (``data:image/jpeg;base64,...``).
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]

    # -- Watermark ---------------------------------------------------------
    wm: Dict[str, Any] = analysis_results.get("watermark", {})
    if wm.get("location"):
        loc = wm["location"]
        color = (0, 255, 0) if wm["status"] == "present" else (0, 0, 255)
        cv2.rectangle(
            annotated,
            (loc["x"], loc["y"]),
            (loc["x"] + loc["width"], loc["y"] + loc["height"]),
            color,
            2,
        )
        label = (
            f"Watermark: {wm['confidence']:.0%}"
            if wm["status"] == "present"
            else "Watermark: MISSING"
        )
        cv2.putText(
            annotated, label, (loc["x"], max(loc["y"] - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    # -- Security Thread ---------------------------------------------------
    st: Dict[str, Any] = analysis_results.get("security_thread", {})
    if st.get("coordinates"):
        coords = st["coordinates"]
        color = (0, 255, 0) if st["status"] == "present" else (0, 0, 255)
        cv2.line(
            annotated,
            (coords["x_start"], 0),
            (coords.get("x_end", coords["x_start"]), height),
            color,
            2,
        )
        label = (
            f"Thread: {st['confidence']:.0%}"
            if st["status"] == "present"
            else "Thread: MISSING"
        )
        cv2.putText(
            annotated, label, (coords["x_start"], 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    # -- Color Analysis indicator ------------------------------------------
    ca: Dict[str, Any] = analysis_results.get("color_analysis", {})
    if ca.get("status"):
        color_indicator = (0, 255, 0) if ca["status"] == "match" else (0, 0, 255)
        cv2.putText(
            annotated,
            f"Color: {ca['status'].upper()} ({ca['confidence']:.0%})",
            (10, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_indicator,
            2,
        )

    # -- Texture indicator -------------------------------------------------
    tx: Dict[str, Any] = analysis_results.get("texture_analysis", {})
    if tx.get("status"):
        texture_color = (0, 255, 0) if tx["status"] == "normal" else (0, 0, 255)
        cv2.putText(
            annotated,
            f"Texture: {tx['status'].upper()} ({tx['confidence']:.0%})",
            (10, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            texture_color,
            2,
        )

    # -- Serial Number -----------------------------------------------------
    sn: Dict[str, Any] = analysis_results.get("serial_number", {})
    if sn.get("status"):
        sn_color = (0, 255, 0) if sn["status"] == "valid" else (0, 0, 255)
        sn_text = f"Serial: {sn.get('extracted_text', 'N/A')}"
        cv2.putText(
            annotated,
            sn_text,
            (width - 350, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            sn_color,
            2,
        )

    # -- Result banner -----------------------------------------------------
    result: str = analysis_results.get("overall_result", "UNKNOWN")
    confidence: float = analysis_results.get("ensemble_score", 0.0)
    banner_color = (0, 255, 0) if result == "REAL" else (0, 0, 255)
    cv2.rectangle(annotated, (0, 0), (width, 60), banner_color, -1)
    cv2.putText(
        annotated,
        f"{result} - {confidence:.1%}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    encoded_string = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


def generate_thumbnail(image: np.ndarray, max_size: int = 200) -> str:
    """Resize *image* so its longest dimension is at most *max_size* and return as a JPEG data URI.

    Args:
        image: Input BGR image as a NumPy array.
        max_size: Maximum dimension (width or height) of the thumbnail.

    Returns:
        A base64-encoded JPEG data URI string (``data:image/jpeg;base64,...``).
    """
    height, width = image.shape[:2]
    scale = max_size / max(height, width)
    new_dims = (int(width * scale), int(height * scale))
    thumb = cv2.resize(image, new_dims)

    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    encoded_string = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"
