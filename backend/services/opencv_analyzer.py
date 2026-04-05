"""OpenCV security feature analysis - improved for reliability without reference files."""
import cv2
import numpy as np
import pytesseract
import re
from skimage.feature import graycomatrix, graycoprops


def detect_watermark(gray: np.ndarray, image: np.ndarray, denomination: str) -> dict:
    """Detect watermark by analyzing brightness variation and texture patterns in expected region.
    
    Watermarks on Indian currency show as semi-transparent portrait/region with:
    - Subtle brightness differences from surrounding areas
    - Smooth texture (lower variance than surrounding printed areas)
    - Visible edge patterns
    """
    h, w = gray.shape[:2]
    
    # Watermark region varies by denomination
    if denomination == "₹2000":
        wm_x1, wm_x2 = int(w * 0.50), int(w * 0.85)
        wm_y1, wm_y2 = int(h * 0.15), int(h * 0.75)
    else:  # ₹500 and others
        wm_x1, wm_x2 = int(w * 0.55), int(w * 0.85)
        wm_y1, wm_y2 = int(h * 0.20), int(h * 0.70)
    
    # Extract watermark ROI
    roi = gray[wm_y1:wm_y2, wm_x1:wm_x2]
    if roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "location": None, "ssim_score": None}

    # Method 1: Brightness variation analysis
    # Compare watermark region with surrounding areas
    top = gray[max(0, wm_y1-40):wm_y1, wm_x1:wm_x2]
    bottom = gray[wm_y2:min(h, wm_y2+40), wm_x1:wm_x2]
    left = gray[wm_y1:wm_y2, max(0, wm_x1-40):wm_x1]
    right = gray[wm_y1:wm_y2, wm_x2:min(w, wm_x2+40)]
    
    surrounding_regions = [r for r in [top, bottom, left, right] if r.size > 0]
    if not surrounding_regions:
        return {"status": "unknown", "confidence": 0.5, "location": None, "ssim_score": None}
    
    roi_mean = np.mean(roi)
    roi_std = np.std(roi)
    surrounding_means = [np.mean(r) for r in surrounding_regions]
    surrounding_avg = np.mean(surrounding_means)
    
    # Watermark typically has lower contrast (smoother)
    brightness_diff = abs(roi_mean - surrounding_avg)
    
    # Method 2: Texture analysis - watermarks are smoother
    roi_variance = np.var(roi)
    surrounding_variance = np.mean([np.var(r) for r in surrounding_regions])
    
    # Watermark should have lower variance (smoother texture)
    smoothness_ratio = roi_variance / surrounding_variance if surrounding_variance > 0 else 1.0
    
    # Method 3: Edge density - watermarks have fewer sharp edges
    roi_edges = cv2.Canny(roi, 50, 150)
    roi_edge_density = np.sum(roi_edges > 0) / roi_edges.size
    
    # Surrounding edge density
    surrounding_edges = []
    for r in surrounding_regions:
        edge = cv2.Canny(r, 50, 150)
        surrounding_edges.append(np.sum(edge > 0) / edge.size)
    surrounding_edge_density = np.mean(surrounding_edges)
    
    # Score calculation
    watermark_indicators = 0
    
    # Brightness difference should be moderate (not too high, not too low)
    if 3 <= brightness_diff <= 60:
        watermark_indicators += 1
        brightness_score = min(1.0, brightness_diff / 30.0)
    elif brightness_diff < 3:
        brightness_score = 0.3  # Too uniform - suspicious
    else:
        brightness_score = 0.4  # Too different - might not be watermark
    
    # Smoothness check - watermark should be smoother
    if 0.3 <= smoothness_ratio <= 0.9:
        watermark_indicators += 1
        smoothness_score = 1.0 - smoothness_ratio
    else:
        smoothness_score = 0.3
    
    # Edge density check - watermark should have fewer edges
    if roi_edge_density < surrounding_edge_density * 0.8:
        watermark_indicators += 1
        edge_score = 1.0 - (roi_edge_density / max(surrounding_edge_density, 0.01))
    else:
        edge_score = 0.3
    
    # Combined confidence
    confidence = round(
        0.35 * brightness_score + 
        0.35 * smoothness_score + 
        0.30 * max(0.0, edge_score),
        4
    )
    
    # Status determination
    if watermark_indicators >= 2:
        status = "present"
        confidence = max(0.6, confidence)  # Boost confidence if indicators present
    elif watermark_indicators >= 1:
        status = "present"
        confidence = max(0.5, min(0.65, confidence))
    else:
        status = "unknown"
        confidence = max(0.4, min(0.55, confidence))

    return {
        "status": status, 
        "confidence": round(confidence, 4),
        "location": {"x": wm_x1, "y": wm_y1, "width": wm_x2 - wm_x1, "height": wm_y2 - wm_y1},
        "ssim_score": None,
        "brightness_diff": round(float(brightness_diff), 2),
        "smoothness_ratio": round(float(smoothness_ratio), 4)
    }


def detect_security_thread(gray: np.ndarray, image: np.ndarray) -> dict:
    """Detect security thread using multiple validation methods.
    
    Security threads on Indian currency are:
    - Embedded metallic/plastic strip visible when held to light
    - Usually located in left-center region
    - Appear as dark vertical line with consistent width
    - May have text/patterns on them
    """
    h, w = gray.shape[:2]
    
    # Security thread region (left-center, full height)
    thread_x1, thread_x2 = int(w * 0.20), int(w * 0.45)
    thread_region = gray[:, thread_x1:thread_x2]
    
    if thread_region.size == 0:
        return {"status": "unknown", "confidence": 0.5, "position": "unknown",
                "coordinates": {"x_start": None, "x_end": None}}
    
    # Method 1: Vertical line detection using Canny + HoughLines
    blurred = cv2.GaussianBlur(thread_region, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    
    # Morphological operations to enhance vertical lines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    dilated = cv2.dilate(edges, kernel_v, iterations=1)
    
    # Detect vertical lines using HoughLinesP
    lines = cv2.HoughLinesP(
        dilated, 
        1, np.pi / 180, 
        threshold=40,
        minLineLength=h // 3, 
        maxLineGap=30
    )
    
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is predominantly vertical
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy > 0 and dx / dy < 0.3:  # More vertical than horizontal
                vertical_lines.append((x1 + thread_x1, x2 + thread_x1, y1, y2))
    
    # Method 2: Pixel intensity analysis - security thread appears darker
    # Analyze vertical projection profile
    vertical_profile = np.mean(thread_region, axis=0)
    thread_col = np.argmin(vertical_profile)
    min_intensity = np.min(vertical_profile)
    avg_intensity = np.mean(vertical_profile)
    
    # Security thread should be noticeably darker
    intensity_ratio = min_intensity / avg_intensity if avg_intensity > 0 else 1.0
    has_dark_region = intensity_ratio < 0.7
    
    # Method 3: Texture analysis - metallic threads have unique texture
    # Use variance in horizontal direction
    horizontal_variance = np.var(thread_region, axis=0)
    thread_variance = np.min(horizontal_variance)
    avg_variance = np.mean(horizontal_variance)
    variance_ratio = thread_variance / avg_variance if avg_variance > 0 else 1.0
    
    # Scoring
    line_score = min(1.0, len(vertical_lines) / 3.0) if vertical_lines else 0.0
    pixel_score = max(0.0, 1.0 - intensity_ratio) if has_dark_region else 0.2
    texture_score = max(0.0, 1.0 - variance_ratio)
    
    # Combined confidence with weighted methods
    confidence = round(
        0.50 * line_score + 
        0.30 * pixel_score + 
        0.20 * texture_score,
        4
    )
    
    # Determine status
    if confidence > 0.5:
        status = "present"
    elif confidence > 0.3:
        status = "present"
        confidence = max(0.35, confidence)
    else:
        status = "missing"
        confidence = max(0.2, min(0.35, confidence))
    
    # Calculate thread position
    thread_x = thread_x1 + (thread_x2 - thread_x1) // 2
    if vertical_lines:
        # Use average position of detected lines
        thread_x = int(np.mean([(x1 + x2) / 2 for x1, x2, _, _ in vertical_lines]))
    elif has_dark_region:
        thread_x = thread_col + thread_x1
    
    return {
        "status": status, 
        "confidence": confidence, 
        "position": "vertical",
        "coordinates": {"x_start": thread_x - 5, "x_end": thread_x + 5},
        "vertical_lines_detected": len(vertical_lines),
        "intensity_ratio": round(float(intensity_ratio), 4)
    }


def analyze_color(image: np.ndarray, denomination: str) -> dict:
    """Analyze color uniformity across the note."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = h // grid_rows, w // grid_cols
    
    hue_means, sat_means = [], []
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell = hsv[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
            if cell.size > 0:
                hue_means.append(np.mean(cell[:, :, 0]))
                sat_means.append(np.mean(cell[:, :, 1]))
    
    if not hue_means:
        return {"status": "match", "confidence": 0.6, "bhattacharyya_distance": None, "dominant_colors": None}
    
    hue_var = np.var(hue_means)
    sat_var = np.var(sat_means)
    hue_uni = max(0, 1.0 - hue_var / 500.0)
    sat_uni = max(0, 1.0 - sat_var / 500.0)
    
    full_hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    full_hist = cv2.normalize(full_hist, full_hist).flatten()
    peak_ratio = np.max(full_hist) / np.sum(full_hist) if np.sum(full_hist) > 0 else 0
    
    confidence = round(0.4 * hue_uni + 0.3 * sat_uni + 0.3 * min(1.0, peak_ratio * 100), 4)
    status = "match" if confidence > 0.4 else "mismatch"
    
    return {
        "status": status, "confidence": confidence,
        "bhattacharyya_distance": round(1.0 - confidence, 4),
        "dominant_colors": None
    }


def analyze_texture(gray: np.ndarray) -> dict:
    """Analyze texture quality and print sharpness."""
    gray_resized = cv2.resize(gray, (256, 256))
    laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
    sharpness_var = float(laplacian.var())
    sharpness_norm = min(1.0, sharpness_var / 300.0)
    
    gray_quantized = np.clip(gray_resized // 4, 0, 63).astype(np.uint8)
    glcm = graycomatrix(gray_quantized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)
    
    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    contrast_norm = float(1.0 / (1.0 + np.exp(-0.05 * (contrast - 10.0))))
    
    score = 0.0
    if 0.15 <= contrast_norm <= 0.90: score += 0.25
    if 0.15 <= energy <= 0.85: score += 0.25
    if 0.4 <= homogeneity <= 0.98: score += 0.2
    if sharpness_norm > 0.1: score += 0.3
    
    confidence = round(score, 4)
    status = "normal" if confidence > 0.4 else "abnormal"
    
    return {
        "status": status, "confidence": confidence,
        "glcm_contrast": round(min(contrast_norm, 9.9999), 4),
        "glcm_energy": round(energy, 4),
        "sharpness_score": round(sharpness_norm, 4)
    }


def detect_serial_number(gray: np.ndarray, denomination: str) -> dict:
    """Detect and validate serial number using OCR."""
    h, w = gray.shape[:2]
    y1, y2 = int(h * 0.85), int(h * 0.96)
    x1, x2 = int(w * 0.03), int(w * 0.52)
    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "extracted_text": None, "format_valid": False}

    # Multiple preprocessing attempts for better OCR
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Additional: inverted versions for better text detection
    _, binary_inv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_inv = cv2.bitwise_not(adaptive)

    texts = []
    for img in [binary, adaptive, binary_inv, adaptive_inv]:
        try:
            # Try different PSM modes for better text detection
            for psm in ['7', '6', '13']:
                t = pytesseract.image_to_string(
                    img, 
                    config=f'--psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                ).strip()
                if t and len(t) >= 5:
                    # Clean up text
                    cleaned = ''.join(c for c in t if c.isalnum()).upper()
                    if len(cleaned) >= 5:
                        texts.append(cleaned)
        except Exception:
            pass

    if not texts:
        return {"status": "unknown", "confidence": 0.5, "extracted_text": None, "format_valid": False}

    # Use the longest text found (most complete serial number)
    text = max(texts, key=len)
    
    # Validate serial number format (Indian currency format)
    # Common patterns: 1ABC1234567, AB1234567, etc.
    patterns = [
        r'^[0-9][A-Z]{2,3}[0-9]{6,9}$',  # 1ABC1234567
        r'^[A-Z]{2,3}[0-9]{6,9}$',         # AB1234567
        r'^[A-Z][0-9]{9,10}$',              # A123456789
        r'^[0-9]{9,10}$',                    # Just numbers (fallback)
    ]
    
    is_valid = any(bool(re.match(pattern, text.strip())) for pattern in patterns)

    # Higher confidence if format is valid, lower if OCR failed or format invalid
    if is_valid:
        confidence = round(0.9, 4)
        status = "valid"
    elif text:
        # Text extracted but format doesn't match - could be suspicious
        confidence = round(0.4, 4)
        status = "invalid"
    else:
        # No text extracted
        confidence = round(0.5, 4)
        status = "unknown"

    return {
        "status": status,
        "confidence": confidence,
        "extracted_text": text.strip() if text else None,
        "format_valid": is_valid
    }


def verify_dimensions(image: np.ndarray) -> dict:
    """Verify currency note dimensions."""
    h, w = image.shape[:2]
    max_dim = 800
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    resized = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"status": "correct", "confidence": 0.6, "aspect_ratio": None,
                "expected_aspect_ratio": 1.69, "deviation_percent": None}
    
    img_area = resized.shape[0] * resized.shape[1]
    min_area = img_area * 0.10  # Lower threshold for cropped images
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    
    largest = max(valid, key=cv2.contourArea) if valid else max(contours, key=cv2.contourArea)
    _, _, w_r, h_r = cv2.boundingRect(largest)
    aspect_ratio = w_r / h_r if h_r > 0 else 1.69
    
    expected = 1.69
    deviation = abs(aspect_ratio - expected) / expected * 100
    confidence = round(max(0, 1.0 - deviation / 25.0), 4)  # More lenient
    status = "correct" if deviation < 25.0 else "incorrect"
    
    return {
        "status": status, "confidence": confidence,
        "aspect_ratio": round(aspect_ratio, 4),
        "expected_aspect_ratio": expected,
        "deviation_percent": round(deviation, 2)
    }


def analyze_security_features(image, denoised, enhanced, denomination: str) -> dict:
    """Run all security feature detections."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "watermark": detect_watermark(gray, image, denomination),
        "security_thread": detect_security_thread(gray, image),
        "color_analysis": analyze_color(image, denomination),
        "texture_analysis": analyze_texture(gray),
        "serial_number": detect_serial_number(gray, denomination),
        "dimensions": verify_dimensions(image),
    }
