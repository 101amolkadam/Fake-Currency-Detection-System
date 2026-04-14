# Quick Reference: Image Processing in Fake Currency Detection System

## 🎯 Visual Guide to Edge Detection, Segmentation & Preprocessing

---

## 📊 Processing Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE (Base64)                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  DECODE & VALIDATE                                    │
│  File: services/image_preprocessor.py:9-41                           │
│  ✓ MIME type validation (JPEG/PNG/WebP)                             │
│  ✓ Size limit check (< 10MB)                                        │
│  ✓ Base64 decoding                                                  │
│  ✓ cv2.imdecode() → BGR numpy array                                 │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  PREPROCESS_IMAGE()                                  │
│  File: services/image_preprocessor.py:93-117                         │
│                                                                      │
│  ┌───────────────────┬──────────────────┬──────────────────────┐   │
│  │  STREAM 1: CNN    │  STREAM 2:       │  STREAM 3: EDGES    │   │
│  │  (MobileNetV3)    │  DENOISED        │  (CLAHE)            │   │
│  │                   │                  │                     │   │
│  │  Resize → 224×224 │  fastNlMeans     │  BGR → Gray         │   │
│  │  BGR → RGB        │  Denoising       │  CLAHE              │   │
│  │  Normalize        │  Colored         │  clipLimit=2.0      │   │
│  │  ImageNet stats   │  h=10, hColor=10 │  tileGrid=8×8       │   │
│  └─────────┬─────────┴────────┬─────────┴─────────┬────────────┘   │
└────────────┼──────────────────┼──────────────────┼─────────────────┘
             │                  │                  │
             ▼                  ▼                  ▼
        CNN Inference     OpenCV Features    Edge Detection
        (PyTorch)         (15 features)      (Canny/Sobel)
```

---

## 🔍 Edge Detection Quick Reference

### **Canny Edge Detection** - Used in 10+ Features

```python
# Generic pattern used throughout:
edges = cv2.Canny(image, low_threshold, high_threshold)
edge_density = np.sum(edges > 0) / edges.size
```

| Feature | Thresholds | File:Line | Purpose |
|---------|-----------|-----------|---------|
| **Watermark** | (50, 150) | `opencv_analyzer.py:113` | Compare edge density with surrounding |
| **Security Thread** | (40, 120) | `opencv_analyzer.py:207` | Detect vertical metallic line |
| **Dimensions** | (50, 150) | `opencv_analyzer.py:491` | Find currency note boundaries |
| **Latent Image** | (40, 120) | `opencv_analyzer.py:601` | Detect hidden numeral edges |
| **Intaglio Printing** | (50, 150) | `opencv_analyzer.py:679` | Raised ink edge patterns |
| **Microlettering** | (40, 120) | `opencv_analyzer.py:813` | Fine text edge detection |
| **Angular Lines** | (50, 150) | `opencv_analyzer.py:1012` | Accessibility line detection |

**Why Two Threshold Sets?**
- **(40, 120)**: Lower thresholds → More sensitive, detects finer details
- **(50, 150)**: Higher thresholds → Less noise, better for texture analysis

---

### **Laplacian Edge Detection** - Texture Sharpness

```python
# Location: opencv_analyzer.py:343-344
laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
sharpness_var = float(laplacian.var())
```

**Purpose**: Measures overall image sharpness/blur
- High variance = sharp image (original)
- Low variance = blurry image (possible photocopy)

---

### **Sobel Gradient Operators** - Intaglio Detection

```python
# Location: opencv_analyzer.py:692-695
sobelx = cv2.Sobel(portrait_roi, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(portrait_roi, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
```

**Purpose**: Detects directional gradients from raised intaglio ink
- Creates shadow effects visible in gradient magnitude
- Better than Canny for texture measurement

---

## 🎨 Image Segmentation Methods

### **1. Threshold-Based Segmentation**

#### **Otsu's Binarization** (Automatic Threshold)
```python
# Locations: opencv_analyzer.py:412, 418, 905
_, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```
**Use Cases**:
- Serial number OCR preprocessing
- Identification mark shape detection

#### **Adaptive Gaussian Thresholding**
```python
# Location: opencv_analyzer.py:413-415
adaptive = cv2.adaptiveThreshold(
    roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
```
**Parameters**:
- Block size: 11×11 pixels
- Constant C: 2

**Use Case**: Serial number OCR with uneven illumination

#### **Multiple Threshold Strategy**
```python
# 4 variations tried for maximum OCR success:
1. Otsu binary (normal)
2. Adaptive Gaussian (normal)
3. Otsu inverted
4. Adaptive inverted
```

---

### **2. Color-Based Segmentation (HSV)**

```python
# Location: opencv_analyzer.py:719-721
hsv = cv2.cvtColor(ovi_roi, cv2.COLOR_BGR2HSV)

# Green mask for OVI
green_mask = cv2.inRange(hsv, np.array([55, 50, 50]), np.array([85, 255, 255]))

# Blue mask for OVI
blue_mask = cv2.inRange(hsv, np.array([95, 50, 50]), np.array([135, 255, 255]))
```

**HSV Ranges**:
- **Green**: H∈[55,85], S∈[50,255], V∈[50,255]
- **Blue**: H∈[95,135], S∈[50,255], V∈[50,255]

**Why HSV?**
- Hue is illumination-invariant
- Better color separation than RGB/BGR
- Matches human color perception

---

### **3. Contour-Based Segmentation**

```python
# Location: opencv_analyzer.py:491-494 (dimensions)
edges = cv2.Canny(blurred, 50, 150)
dilated = cv2.dilate(edges, kernel, iterations=2)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**Shape Classification** (`opencv_analyzer.py:915-937`):
```python
# Circle detection via circularity
circularity = area / circle_area

# Polygon approximation
epsilon = 0.04 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)
num_vertices = len(approx)  # 3=triangle, 4=square/rectangle
```

---

### **4. Morphological Operations**

```python
# Vertical line enhancement: opencv_analyzer.py:209-210
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
dilated = cv2.dilate(edges, kernel_v, iterations=1)

# General enhancement: opencv_analyzer.py:492
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(edges, kernel, iterations=2)
```

**Purpose**: Enhance specific shapes before detection
- Vertical kernels for security threads
- Square kernels for general edge enhancement

---

## 📐 Feature Extraction Techniques

### **1. ROI Extraction (15+ Features)**

Each security feature has denomination-specific ROI:

```python
# Example: Watermark ROI varies by denomination
if denomination == "₹2000":
    wm_x1, wm_x2 = int(w * 0.50), int(w * 0.85)  # Wider region
    wm_y1, wm_y2 = int(h * 0.15), int(h * 0.75)
else:  # ₹500
    wm_x1, wm_x2 = int(w * 0.55), int(w * 0.85)  # Narrower
    wm_y1, wm_y2 = int(h * 0.20), int(h * 0.70)
```

**All ROI Locations**:
- Watermark: Lines 65-70
- Security Thread: Lines 192-194
- Serial Number: Lines 395-398
- OVI: Lines 701-706
- Latent Image: Lines 580-586
- Intaglio: Lines 681-689 (2 regions)
- Microlettering: Lines 787-789
- Identification Mark: Lines 892-894
- Angular Lines: Lines 990-996 (2 regions)

---

### **2. GLCM Texture Analysis**

```python
# Location: opencv_analyzer.py:348-351
glcm = graycomatrix(
    gray_quantized, 
    distances=[1], 
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],  # 4 directions
    levels=64, 
    symmetric=True, 
    normed=True
)

# Extract features
contrast = graycoprops(glcm, 'contrast')
energy = graycoprops(glcm, 'energy')
homogeneity = graycoprops(glcm, 'homogeneity')
```

**Why 4 Angles?** Captures texture in all directions:
- 0° (horizontal)
- 45° (diagonal)
- 90° (vertical)
- 135° (anti-diagonal)

---

### **3. Edge Density Analysis**

```python
# Pattern used in 8+ features
edges = cv2.Canny(roi, 50, 150)
edge_density = float(np.sum(edges > 0) / edges.size)
```

**Expected Densities**:

| Feature | Expected Range | Interpretation |
|---------|---------------|----------------|
| Watermark | <80% of surrounding | Smoother than background ✓ |
| Intaglio | 10-45% | High density = raised ink ✓ |
| Microlettering | 8-35% | Fine text creates edges ✓ |
| Latent Image | 3-25% | Moderate patterns ✓ |

---

### **4. Hough Line Transform**

```python
# Location: opencv_analyzer.py:213-218 (Security Thread)
lines = cv2.HoughLinesP(
    dilated,
    rho=1,                    # Distance resolution (pixels)
    theta=np.pi / 180,        # Angle resolution (1 degree)
    threshold=40,             # Minimum votes
    minLineLength=h // 3,     # Minimum line length
    maxLineGap=30             # Max gap to connect segments
)
```

**Vertical Line Filtering**:
```python
for line in lines:
    x1, y1, x2, y2 = line[0]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dy > 0 and dx / dy < 0.3:  # More vertical than horizontal
        vertical_lines.append(...)
```

---

## 🛡️ Edge Cases & Robustness

### **1. Empty ROI Protection**
```python
# Pattern used in 10+ locations
roi = gray[y1:y2, x1:x2]
if roi.size == 0:
    return {"status": "unknown", "confidence": 0.5, ...}
```

**Locations**: Watermark:74, Thread:197, Serial:401, Intaglio:669, Latent:591, OVI:713, Micro:795, ID Mark:901, Angular:997, See-Through:1089

---

### **2. Division by Zero Prevention**
```python
# Aspect ratio (Line 508)
aspect_ratio = w_r / h_r if h_r > 0 else 1.69

# Smoothness ratio (Line 107)
smoothness_ratio = roi_variance / surrounding_variance if surrounding_variance > 0 else 1.0

# Edge density (Line 141)
edge_score = 1.0 - (roi_edge_density / max(surrounding_edge_density, 0.01))
```

---

### **3. Boundary Clamping**
```python
# Watermark surrounding regions (Lines 80-83)
top = gray[max(0, wm_y1 - 40):wm_y1, wm_x1:wm_x2]
bottom = gray[wm_y2:min(h, wm_y2 + 40), wm_x1:wm_x2]
left = gray[wm_y1:wm_y2, max(0, wm_x1 - 40):wm_x1]
right = gray[wm_y1:wm_y2, wm_x2:min(w, wm_x2 + 40)]
```

---

### **4. Numerical Stability**
```python
# Confidence clipping (cnn_classifier.py:213)
raw_confidence = np.clip(raw_confidence, epsilon, 1 - epsilon)

# Value clamping (cnn_classifier.py:180-184)
brighter = torch.clamp(image_tensor * 1.1, -1.0, 1.0)
darker = torch.clamp(image_tensor * 0.9, -1.0, 1.0)
```

---

## 📁 File Quick Reference

| File | Primary Purpose | Key Functions |
|------|----------------|---------------|
| `image_preprocessor.py` | Base64 decode, CNN prep, denoising, CLAHE | `decode_base64_image()`, `preprocess_image()` |
| `opencv_analyzer.py` | 15 security features with edge detection | `detect_*()`, `analyze_*()` |
| `cnn_classifier.py` | PyTorch inference with TTA | `classify_currency()`, `_apply_tta_augmentations()` |
| `image_annotator.py` | Visual result generation | `generate_annotated_image()`, `generate_thumbnail()` |

---

## 🔢 Performance Numbers

| Operation | Time (CPU) | Time (GPU) | Technique |
|-----------|-----------|-----------|-----------|
| Base64 Decode | 5ms | 5ms | `base64.b64decode()` + `cv2.imdecode()` |
| Resize (224×224) | 2ms | 2ms | `cv2.resize()` |
| Denoising | 8ms | 8ms | `fastNlMeansDenoisingColored()` |
| CLAHE | 3ms | 3ms | `cv2.createCLAHE().apply()` |
| Canny Edges | 1-3ms | 1-3ms | `cv2.Canny()` per feature |
| CNN (single) | 450ms | 80ms | MobileNetV3 forward pass |
| CNN (TTA 7x) | 3150ms | 560ms | 7 augmented predictions |
| GLCM Texture | 15ms | 15ms | `graycomatrix()` 4 angles |
| OCR (Tesseract) | 50-200ms | 50-200ms | Multiple preprocessing attempts |

---

## 🎓 Key Algorithms Summary

| Algorithm | OpenCV Function | Use Count | Purpose |
|-----------|----------------|-----------|---------|
| **Canny Edge** | `cv2.Canny()` | 10+ | Feature boundary detection |
| **Hough Transform** | `cv2.HoughLinesP()` | 3 | Line detection (threads, angles) |
| **Otsu Threshold** | `cv2.THRESH_OTSU` | 3 | Automatic binarization |
| **Adaptive Threshold** | `cv2.adaptiveThreshold()` | 1 | Illumination-invariant OCR |
| **Morphological Dilation** | `cv2.dilate()` | 5+ | Edge/line enhancement |
| **Contour Detection** | `cv2.findContours()` | 2 | Shape segmentation |
| **CLAHE** | `cv2.createCLAHE()` | 1 | Contrast enhancement |
| **Laplacian** | `cv2.Laplacian()` | 1 | Sharpness measurement |
| **Sobel** | `cv2.Sobel()` | 2 | Gradient detection |
| **GLCM** | `graycomatrix()` | 1 | Texture analysis |
| **Denoising** | `fastNlMeansDenoisingColored()` | 1 | Noise reduction |

---

## 📚 Further Reading

- **Detailed Documentation**: `docs/IMAGE_PROCESSING_TECHNIQUES.md`
- **Refactoring Summary**: `REFACTORING.md`
- **API Documentation**: http://localhost:8000/api/docs (Swagger UI)

---

**Version**: 2.0.0  
**Last Updated**: April 14, 2026
