"""Enhanced OpenCV security feature analysis - comprehensive detection of 15+ security features.

Indian currency notes have multiple critical security features that are extremely difficult to replicate.
This module detects all detectable features using computer vision techniques.

CRITICAL FEATURES (note is FAKE if any fail):
- Security Thread (30% weight)
- Watermark (25% weight)  
- Serial Number with progressive sizing (20% weight)

IMPORTANT FEATURES:
- Optically Variable Ink (15%)
- Latent Image (12%)
- Intaglio Printing (12%)
- See-Through Registration (10%)

SUPPORTING FEATURES:
- Microlettering (8%)
- Fluorescence (7%)
- Color Analysis (7%)
- Texture (5%)
- Dimensions (5%)
- Identification Mark (5%)
- Angular Lines (3%)
"""
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
        "intaglio_printing": detect_intaglio_printing(gray, image),
        "latent_image": detect_latent_image(gray, image, denomination),
        "optically_variable_ink": detect_optically_variable_ink(image, denomination),
        "microlettering": detect_microlettering(gray, image, denomination),
        "identification_mark": detect_identification_mark(gray, image, denomination),
        "angular_lines": detect_angular_lines(gray, image, denomination),
        "fluorescence": detect_fluorescence(gray, image, denomination),
        "see_through_registration": detect_see_through_registration(gray, image, denomination),
    }


def detect_intaglio_printing(gray: np.ndarray, image: np.ndarray) -> dict:
    """Detect intaglio (raised) printing using texture and edge analysis.
    
    Intaglio printing creates raised ink that can be felt by touch. Visually, it shows:
    - Higher edge density in specific regions
    - Distinctive texture patterns
    - Strong local contrast variations
    - Shadow effects from angled lighting
    
    Key locations:
    - Mahatma Gandhi portrait
    - RBI seal
    - Guarantee clause
    - Ashoka Pillar Emblem
    """
    h, w = gray.shape[:2]
    
    # Define regions where intaglio printing is prominent
    # These regions vary by denomination but generally include:
    # 1. Central portrait area
    # 2. Left side elements (RBI seal, etc.)
    
    # Region 1: Central portrait (main area)
    portrait_x1, portrait_x2 = int(w * 0.35), int(w * 0.70)
    portrait_y1, portrait_y2 = int(h * 0.15), int(h * 0.70)
    portrait_roi = gray[portrait_y1:portrait_y2, portrait_x1:portrait_x2]
    
    # Region 2: Left side elements
    left_x1, left_x2 = int(w * 0.05), int(w * 0.30)
    left_y1, left_y2 = int(h * 0.20), int(h * 0.60)
    left_roi = gray[left_y1:left_y2, left_x1:left_x2]
    
    if portrait_roi.size == 0 or left_roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "edge_density": None}
    
    # Method 1: Edge density analysis
    # Intaglio printing has higher edge density due to raised ink
    portrait_edges = cv2.Canny(portrait_roi, 50, 150)
    portrait_edge_density = np.sum(portrait_edges > 0) / portrait_roi.size
    
    left_edges = cv2.Canny(left_roi, 50, 150)
    left_edge_density = np.sum(left_edges > 0) / left_roi.size
    
    # Method 2: Local variance analysis
    # Intaglio regions show higher local contrast
    portrait_std = np.std(portrait_roi)
    left_std = np.std(left_roi)
    overall_std = np.std(gray)
    
    # Method 3: Gradient magnitude analysis
    # Calculate gradient magnitude (Sobel operators)
    sobelx = cv2.Sobel(portrait_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(portrait_roi, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    avg_gradient = np.mean(gradient_magnitude)
    
    # Scoring
    # Real intaglio: edge density 15-40%, high variance, strong gradients
    edge_score = 0.0
    if 0.10 <= portrait_edge_density <= 0.45:
        edge_score = min(1.0, portrait_edge_density / 0.25)
    elif portrait_edge_density > 0.45:
        edge_score = 0.6  # Too dense might indicate overprinting
    
    variance_ratio = portrait_std / overall_std if overall_std > 0 else 1.0
    variance_score = 0.0
    if 1.1 <= variance_ratio <= 2.5:
        variance_score = min(1.0, (variance_ratio - 1.0) / 1.0)
    
    gradient_score = min(1.0, avg_gradient / 80.0)
    
    # Combined confidence
    confidence = round(
        0.45 * edge_score +
        0.30 * variance_score +
        0.25 * gradient_score,
        4
    )
    
    # Status determination
    if confidence > 0.55:
        status = "present"
    elif confidence > 0.40:
        status = "present"
        confidence = max(0.45, confidence)
    else:
        status = "missing"
        confidence = max(0.2, min(0.40, confidence))
    
    return {
        "status": status,
        "confidence": confidence,
        "edge_density": round(float(portrait_edge_density), 4),
        "variance_ratio": round(float(variance_ratio), 4),
        "gradient_magnitude": round(float(avg_gradient), 2)
    }


def detect_latent_image(gray: np.ndarray, image: np.ndarray, denomination: str) -> dict:
    """Detect latent image (hidden denomination numeral visible at angle).
    
    The latent image is:
    - Located in vertical band right of Mahatma Gandhi portrait
    - Only visible when note is held horizontally at eye level
    - Shows denomination numeral when visible
    - Created using specialized printing technique
    
    Detection strategy:
    - Analyze the latent image region for hidden patterns
    - Look for denomination-specific numeral shapes
    - Check for edge patterns that would indicate hidden text
    """
    h, w = gray.shape[:2]
    
    # Latent image region (varies by denomination)
    # Generally located right of portrait, in vertical band
    if denomination == "₹2000":
        latent_x1, latent_x2 = int(w * 0.72), int(w * 0.78)
        latent_y1, latent_y2 = int(h * 0.35), int(h * 0.55)
    else:  # ₹500 and others
        latent_x1, latent_x2 = int(w * 0.70), int(w * 0.76)
        latent_y1, latent_y2 = int(h * 0.38), int(h * 0.58)
    
    latent_roi = gray[latent_y1:latent_y2, latent_x1:latent_x2]
    
    if latent_roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "pattern_detected": False}
    
    # Method 1: Edge pattern analysis
    # Latent images create subtle edge patterns even when not visible
    edges = cv2.Canny(latent_roi, 40, 120)
    edge_density = np.sum(edges > 0) / latent_roi.size
    
    # Method 2: Texture uniformity check
    # Region should have some structure (not completely uniform)
    local_std = np.std(latent_roi)
    
    # Method 3: Look for horizontal line patterns (numeral structure)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dilated_h = cv2.dilate(edges, kernel_h, iterations=1)
    horizontal_lines = np.sum(dilated_h > 0) / dilated_h.size
    
    # Method 4: Vertical line patterns
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    dilated_v = cv2.dilate(edges, kernel_v, iterations=1)
    vertical_lines = np.sum(dilated_v > 0) / dilated_v.size
    
    # Scoring
    # Real latent image: moderate edge density (0.05-0.20), some texture
    edge_score = 0.0
    if 0.03 <= edge_density <= 0.25:
        edge_score = min(1.0, edge_density / 0.12)
    
    texture_score = 0.0
    if 15 <= local_std <= 60:
        texture_score = min(1.0, local_std / 35.0)
    
    pattern_score = min(1.0, (horizontal_lines + vertical_lines) / 0.10)
    
    confidence = round(
        0.40 * edge_score +
        0.35 * texture_score +
        0.25 * pattern_score,
        4
    )
    
    # Status determination - be lenient as this is hard to detect
    if confidence > 0.50:
        status = "present"
    elif confidence > 0.35:
        status = "present"
        confidence = max(0.40, confidence)
    else:
        status = "unknown"
        confidence = max(0.3, min(0.50, confidence))
    
    return {
        "status": status,
        "confidence": confidence,
        "edge_density": round(float(edge_density), 4),
        "texture_std": round(float(local_std), 2)
    }


def detect_optically_variable_ink(image: np.ndarray, denomination: str) -> dict:
    """Detect optically variable ink (color-shifting denomination numeral).
    
    OVI characteristics:
    - Color changes from green to blue when tilted
    - Used on ₹500, ₹2000 (New Series)
    - Located on front of note
    - Very difficult to replicate
    
    Detection strategy:
    - Analyze color in denomination numeral region
    - Check for green-blue color range
    - Measure color saturation and hue consistency
    """
    h, w = image.shape[:2]
    
    # OVI numeral location (front, varies by denomination)
    if denomination == "₹2000":
        ovi_x1, ovi_x2 = int(w * 0.15), int(w * 0.35)
        ovi_y1, ovi_y2 = int(h * 0.55), int(h * 0.75)
    elif denomination == "₹500":
        ovi_x1, ovi_x2 = int(w * 0.12), int(w * 0.32)
        ovi_y1, ovi_y2 = int(h * 0.58), int(h * 0.78)
    else:
        # Older series may not have OVI
        return {"status": "unknown", "confidence": 0.5, "note": "OVI not applicable"}
    
    ovi_roi = image[ovi_y1:ovi_y2, ovi_x1:ovi_x2]
    
    if ovi_roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "color_analysis": None}
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(ovi_roi, cv2.COLOR_BGR2HSV)
    
    # OVI typically shifts between green (60-80) and blue (100-130) in HSV
    # Look for hue in this range
    green_mask = cv2.inRange(hsv, np.array([55, 50, 50]), np.array([85, 255, 255]))
    blue_mask = cv2.inRange(hsv, np.array([95, 50, 50]), np.array([135, 255, 255]))
    
    green_pixels = np.sum(green_mask > 0) / hsv.shape[0] / hsv.shape[1]
    blue_pixels = np.sum(blue_mask > 0) / hsv.shape[0] / hsv.shape[1]
    
    # OVI should have significant green OR blue pixels
    ovi_color_presence = green_pixels + blue_pixels
    
    # Check saturation (OVI is highly saturated)
    saturation = np.mean(hsv[:, :, 1])
    saturation_score = min(1.0, saturation / 150.0)
    
    # Check value/brightness
    value = np.mean(hsv[:, :, 2])
    value_score = min(1.0, value / 180.0)
    
    # Look for color variation (indicates shifting ink)
    hue_variance = np.var(hsv[:, :, 0])
    hue_variation_score = min(1.0, hue_variance / 500.0)
    
    # Scoring
    color_score = min(1.0, ovi_color_presence / 0.15)
    
    confidence = round(
        0.40 * color_score +
        0.25 * saturation_score +
        0.20 * value_score +
        0.15 * hue_variation_score,
        4
    )
    
    # Status - be lenient as OVI detection is challenging without multi-angle
    if confidence > 0.50:
        status = "present"
    elif confidence > 0.35:
        status = "present"
        confidence = max(0.40, confidence)
    else:
        status = "unknown"
        confidence = max(0.3, min(0.50, confidence))
    
    return {
        "status": status,
        "confidence": confidence,
        "green_pixel_ratio": round(float(green_pixels), 4),
        "blue_pixel_ratio": round(float(blue_pixels), 4),
        "saturation": round(float(saturation), 2)
    }


def detect_microlettering(gray: np.ndarray, image: np.ndarray, denomination: str) -> dict:
    """Detect microprinting (microscopic text: RBI, denomination value).
    
    Microlettering characteristics:
    - Extremely small text between vertical band and portrait
    - Shows 'RBI' and denomination value
    - Requires magnification to read
    - Precision-printed, very difficult to replicate
    
    Detection strategy:
    - High-resolution OCR in microtext region
    - Edge density analysis (text creates specific patterns)
    - Texture analysis for fine details
    """
    h, w = gray.shape[:2]
    
    # Microlettering region (between vertical band and portrait)
    micro_x1, micro_x2 = int(w * 0.48), int(w * 0.65)
    micro_y1, micro_y2 = int(h * 0.40), int(h * 0.65)
    
    micro_roi = gray[micro_y1:micro_y2, micro_x1:micro_x2]
    
    if micro_roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "text_detected": False}
    
    # Method 1: High-resolution OCR with multiple preprocessing
    # Resize for better OCR on small text
    micro_resized = cv2.resize(micro_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Multiple thresholding approaches
    _, binary1 = cv2.threshold(micro_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary2 = cv2.adaptiveThreshold(micro_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    texts = []
    for img in [binary1, binary2]:
        try:
            # Try different PSM modes for small text
            for psm in ['6', '7', '11', '13']:
                t = pytesseract.image_to_string(
                    img,
                    config=f'--psm {psm} -c tessedit_char_whitelist=RBI0123456789'
                ).strip()
                if t and len(t) >= 2:
                    cleaned = ''.join(c for c in t if c.isalnum()).upper()
                    if len(cleaned) >= 2:
                        texts.append(cleaned)
        except Exception:
            pass
    
    # Method 2: Edge density analysis
    # Microtext creates high edge density in small region
    micro_edges = cv2.Canny(micro_roi, 40, 120)
    edge_density = np.sum(micro_edges > 0) / micro_roi.size
    
    # Method 3: Texture variance (fine text creates specific texture)
    texture_var = np.var(micro_roi)
    
    # Scoring
    ocr_score = 0.0
    text_found = len(texts) > 0
    
    # Check if any text contains RBI (strong indicator)
    rbi_found = any('RBI' in t for t in texts)
    if rbi_found:
        ocr_score = 1.0
    elif text_found:
        ocr_score = 0.7
    else:
        # Fall back to edge density analysis
        # Real microtext: edge density 0.10-0.30
        if 0.08 <= edge_density <= 0.35:
            ocr_score = min(1.0, edge_density / 0.18)
    
    texture_score = min(1.0, texture_var / 400.0)
    
    confidence = round(
        0.50 * ocr_score +
        0.30 * min(1.0, edge_density / 0.18) +
        0.20 * texture_score,
        4
    )
    
    # Status - lenient as microlettering is hard to detect
    if confidence > 0.55:
        status = "present"
    elif confidence > 0.40:
        status = "present"
        confidence = max(0.45, confidence)
    else:
        status = "unknown"
        confidence = max(0.3, min(0.50, confidence))
    
    return {
        "status": status,
        "confidence": confidence,
        "text_detected": text_found,
        "rbi_found": rbi_found,
        "edge_density": round(float(edge_density), 4)
    }


def detect_identification_mark(gray: np.ndarray, image: np.ndarray, denomination: str) -> dict:
    """Detect identification mark for visually impaired (raised intaglio shape).
    
    Identification marks by denomination:
    - ₹20: Vertical rectangle
    - ₹50: Square
    - ₹100: Triangle
    - ₹500: Circle
    - ₹200, ₹500, ₹2000: Various shapes
    
    Location: Left of watermark window
    
    Detection strategy:
    - Shape detection using contour analysis
    - Expected shape for denomination
    - Verify presence and shape correctness
    """
    h, w = gray.shape[:2]
    
    # Expected shapes by denomination
    expected_shapes = {
        "₹20": "rectangle",
        "₹50": "square",
        "₹100": "triangle",
        "₹500": "circle",
        "₹2000": "diamond"
    }
    
    expected_shape = expected_shapes.get(denomination)
    if not expected_shape:
        return {"status": "unknown", "confidence": 0.5, "note": "Unknown denomination"}
    
    # Identification mark region (left of watermark)
    id_x1, id_x2 = int(w * 0.45), int(w * 0.55)
    id_y1, id_y2 = int(h * 0.25), int(h * 0.45)
    
    id_roi = gray[id_y1:id_y2, id_x1:id_x2]
    
    if id_roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "shape_detected": None}
    
    # Shape detection using contour analysis
    _, binary = cv2.threshold(id_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_detected = None
    shape_confidence = 0.0
    
    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 50:  # Minimum area threshold
            # Bounding rectangle
            x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 1.0
            
            # Minimum enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            circle_area = np.pi * (radius ** 2)
            circularity = area / circle_area if circle_area > 0 else 0
            
            # Polygon approximation
            epsilon = 0.04 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            num_vertices = len(approx)
            
            # Shape classification
            if 0.85 <= circularity <= 1.15:
                shape_detected = "circle"
                shape_confidence = min(1.0, circularity)
            elif num_vertices == 3:
                shape_detected = "triangle"
                shape_confidence = 0.8
            elif num_vertices == 4:
                if 0.85 <= aspect_ratio <= 1.15:
                    shape_detected = "square"
                    shape_confidence = 0.8
                else:
                    shape_detected = "rectangle"
                    shape_confidence = 0.7
            elif 0.5 <= aspect_ratio <= 0.7 and num_vertices == 4:
                shape_detected = "diamond"
                shape_confidence = 0.7
    
    # Scoring
    if shape_detected == expected_shape:
        status = "present"
        confidence = round(max(0.6, shape_confidence), 4)
    elif shape_detected:
        status = "mismatch"
        confidence = round(max(0.3, shape_confidence * 0.5), 4)
    else:
        status = "missing"
        confidence = round(0.2, 4)
    
    return {
        "status": status,
        "confidence": confidence,
        "expected_shape": expected_shape,
        "detected_shape": shape_detected
    }


def detect_angular_lines(gray: np.ndarray, image: np.ndarray, denomination: str) -> dict:
    """Detect angular lines (geometric lines for accessibility).
    
    Angular lines characteristics:
    - Present on ₹100, ₹200, ₹500, ₹2000
    - Located on left and right sides of front
    - Series of geometric lines at angles
    - Helps visually impaired identify notes
    
    Detection strategy:
    - Hough line detection in angular line regions
    - Verify presence of lines at expected angles
    - Check line count and arrangement
    """
    h, w = gray.shape[:2]
    
    # Angular lines only on certain denominations
    if denomination not in ["₹100", "₹200", "₹500", "₹2000"]:
        return {"status": "unknown", "confidence": 0.5, "note": "Not applicable"}
    
    # Regions for angular lines (left and right sides)
    left_x1, left_x2 = int(w * 0.02), int(w * 0.12)
    left_y1, left_y2 = int(h * 0.30), int(h * 0.70)
    
    right_x1, right_x2 = int(w * 0.88), int(w * 0.98)
    right_y1, right_y2 = int(h * 0.30), int(h * 0.70)
    
    left_roi = gray[left_y1:left_y2, left_x1:left_x2]
    right_roi = gray[right_y1:right_y2, right_x1:right_x2]
    
    if left_roi.size == 0 and right_roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "lines_detected": 0}
    
    # Detect lines using HoughLinesP
    lines_detected = 0
    
    for roi in [left_roi, right_roi]:
        if roi.size == 0:
            continue
        
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1, np.pi / 180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10
        )
        
        if lines is not None:
            # Filter for angular lines (not horizontal or vertical)
            angular_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Angular lines typically between 30-60 degrees
                if 25 <= abs(angle) <= 70:
                    angular_count += 1
            
            lines_detected += angular_count
    
    # Scoring
    # Expected: 3-8 angular lines
    line_score = 0.0
    if 2 <= lines_detected <= 10:
        line_score = min(1.0, lines_detected / 5.0)
    
    confidence = round(line_score, 4)
    
    # Status
    if confidence > 0.50:
        status = "present"
    elif confidence > 0.30:
        status = "present"
        confidence = max(0.35, confidence)
    else:
        status = "unknown"
        confidence = max(0.25, min(0.45, confidence))
    
    return {
        "status": status,
        "confidence": confidence,
        "angular_lines_count": lines_detected
    }


def detect_fluorescence(gray: np.ndarray, image: np.ndarray, denomination: str) -> dict:
    """Detect fluorescence (UV-responsive ink in number panels).
    
    Fluorescence characteristics:
    - Number panels glow under UV light
    - Central band fluoresces
    - Optical fibers embedded in paper
    
    NOTE: This requires UV illumination to detect properly.
    Without UV setup, we can only do basic analysis.
    
    Detection strategy (without UV):
    - Analyze brightness in number panel regions
    - Look for unusual brightness patterns
    - Provide baseline measurement
    """
    h, w = image.shape[:2]
    
    # Number panel regions (top left, bottom right)
    top_left_x1, top_left_x2 = int(w * 0.02), int(w * 0.20)
    top_left_y1, top_left_y2 = int(h * 0.02), int(h * 0.12)
    
    bottom_right_x1, bottom_right_x2 = int(w * 0.80), int(w * 0.98)
    bottom_right_y1, bottom_right_y2 = int(h * 0.88), int(h * 0.98)
    
    top_left_roi = image[top_left_y1:top_left_y2, top_left_x1:top_left_x2]
    bottom_right_roi = image[bottom_right_y1:bottom_right_y2, bottom_right_x1:bottom_right_x2]
    
    if top_left_roi.size == 0 or bottom_right_roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "note": "ROI not available"}
    
    # Analyze brightness in number panel regions
    # Without UV, this is a baseline measurement
    hsv_top = cv2.cvtColor(top_left_roi, cv2.COLOR_BGR2HSV)
    hsv_bottom = cv2.cvtColor(bottom_right_roi, cv2.COLOR_BGR2HSV)
    
    brightness_top = np.mean(hsv_top[:, :, 2])
    brightness_bottom = np.mean(hsv_bottom[:, :, 2])
    
    avg_brightness = (brightness_top + brightness_bottom) / 2.0
    
    # Without UV setup, we can only provide baseline
    # Mark as unknown but provide data
    return {
        "status": "unknown",
        "confidence": 0.5,
        "note": "Requires UV illumination for accurate detection",
        "brightness_top_panel": round(float(brightness_top), 2),
        "brightness_bottom_panel": round(float(brightness_bottom), 2)
    }


def detect_see_through_registration(gray: np.ndarray, image: np.ndarray, denomination: str) -> dict:
    """Detect see-through registration device (front-back alignment).
    
    See-through registration characteristics:
    - Denomination numeral printed on both sides
    - Front and back align perfectly when held to light
    - Requires precise manufacturing
    
    NOTE: This ideally requires images of both front and back.
    Without both sides, we can only analyze one side for presence of registration marks.
    
    Detection strategy (single-side analysis):
    - Look for registration marks/numerals on front
    - Check for partial numeral in expected region
    - Verify geometric alignment markers
    """
    h, w = image.shape[:2]
    
    # See-through registration region (lower left front, lower right back)
    # We'll analyze the front-side marker
    reg_x1, reg_x2 = int(w * 0.08), int(w * 0.18)
    reg_y1, reg_y2 = int(h * 0.75), int(h * 0.90)
    
    reg_roi = image[reg_y1:reg_y2, reg_x1:reg_x2]
    
    if reg_roi.size == 0:
        return {"status": "unknown", "confidence": 0.5, "note": "ROI not available"}
    
    # Analyze for presence of registration marks
    # Convert to grayscale for analysis
    reg_gray = cv2.cvtColor(reg_roi, cv2.COLOR_BGR2GRAY)
    
    # Look for geometric patterns (partial numeral)
    edges = cv2.Canny(reg_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Look for line structures (numeral parts)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                           minLineLength=15, maxLineGap=8)
    line_count = len(lines) if lines is not None else 0
    
    # Pattern analysis
    has_pattern = edge_density > 0.05 and line_count >= 2
    
    # Scoring
    pattern_score = 0.0
    if has_pattern:
        pattern_score = min(1.0, (edge_density / 0.10 + min(line_count, 6) / 6.0) / 2.0)
    
    confidence = round(pattern_score, 4)
    
    # Status - lenient as single-side analysis is limited
    if confidence > 0.50:
        status = "present"
    elif confidence > 0.30:
        status = "present"
        confidence = max(0.35, confidence)
    else:
        status = "unknown"
        confidence = max(0.3, min(0.50, confidence))
    
    return {
        "status": status,
        "confidence": confidence,
        "edge_density": round(float(edge_density), 4),
        "line_count": line_count,
        "note": "Single-side analysis - full detection requires both sides"
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
        "intaglio_printing": detect_intaglio_printing(gray, image),
        "latent_image": detect_latent_image(gray, image, denomination),
        "optically_variable_ink": detect_optically_variable_ink(image, denomination),
        "microlettering": detect_microlettering(gray, image, denomination),
        "identification_mark": detect_identification_mark(gray, image, denomination),
        "angular_lines": detect_angular_lines(gray, image, denomination),
        "fluorescence": detect_fluorescence(gray, image, denomination),
        "see_through_registration": detect_see_through_registration(gray, image, denomination),
    }
