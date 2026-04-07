# Backend Test & Debug Report

**Date**: 7 April 2026
**Tests Run**: 12 (7 unit + 5 API)
**Status**: ✅ **ALL TESTS PASSED**

---

## Test Results Summary

### Unit Tests (test_application.py) — 7/7 ✅

| # | Test | Status | Details |
|---|------|--------|---------|
| 1 | Image Preprocessing | ✅ PASS | CNN input shape (224,224,3), ImageNet normalization |
| 2 | OpenCV Analyzer | ✅ PASS | All 14 features analyzed with status and confidence |
| 3 | Ensemble Engine | ✅ PASS | Scoring works, critical override triggers correctly |
| 4 | CNN Classifier | ✅ PASS | Model loads on GPU, functional |
| 5 | Image Annotator | ✅ PASS | Annotated image and thumbnail generated |
| 6 | Database | ✅ PASS | MySQL connection successful |
| 7 | API Endpoints | ✅ PASS | All 3 routes exist (/health, /model/info, /analyze) |

### API Tests (test_api.py) — 5/5 ✅

| # | Test | Status | Details |
|---|------|--------|---------|
| 1 | Health Endpoint | ✅ PASS | Returns 200 with model_loaded=true, db_connected |
| 2 | Model Info Endpoint | ✅ PASS | Returns MobileNetV3-Large architecture |
| 3 | Analyze Endpoint | ✅ PASS | Returns 200 with result, confidence, 15 features |
| 4 | Invalid Image Handling | ✅ PASS | Correctly rejects invalid images (422) |
| 5 | History Endpoints | ✅ PASS | Returns 200 with empty history |

---

## Bugs Found & Fixed

### Bug 1: TTA Zoom Crop — Float Slice Indices
**Issue**: `_apply_tta_augmentations()` used float division `//` for array slicing
**Error**: `TypeError: slice indices must be integers or None or have an __index__ method`
**Fix**: Convert crop offsets to int before slicing
**File**: `services/cnn_classifier.py` line ~148

### Bug 2: CNN Input Size Mismatch
**Issue**: `image_preprocessor.py` resized to 299×299 (Xception) but model uses MobileNetV3-Large (224×224)
**Error**: Shape mismatch at inference, model received wrong input size
**Fix**: Added `preprocess_image_for_mobilenet()` with 224×224 resize + ImageNet normalization
**File**: `services/image_preprocessor.py`

### Bug 3: Model Architecture References
**Issue**: `main.py` and `routers/analyze.py` still referenced "Xception" architecture
**Fix**: Updated to "MobileNetV3-Large" in model info endpoint and CNN classification response
**Files**: `main.py`, `routers/analyze.py`

### Bug 4: Training Script — Xception Not Available
**Issue**: `torchvision.models` doesn't have Xception (only available in TensorFlow/Keras)
**Error**: `AttributeError: module 'torchvision.models' has no attribute 'Xception_Weights'`
**Fix**: Replaced with MobileNetV3-Large (depthwise-separable, fits in 3GB VRAM)
**File**: `cnn_classifier.py`, `services/cnn_classifier.py`

### Bug 5: Training Script — Nested Dataset Structure
**Issue**: `CurrencyDataset` only supported flat `fake/*.jpg` / `real/*.jpg` structure
**Fix**: Added recursive subdirectory scanning for denomination folders (`fake/500/*.jpg`, etc.)
**File**: `cnn_classifier.py`

---

## Architecture Changes

| Component | Before | After |
|-----------|--------|-------|
| CNN Backbone | Xception (not in torchvision) | MobileNetV3-Large |
| Input Size | 299×299 | 224×224 |
| Normalization | [-1, 1] (Xception style) | ImageNet mean/std |
| Total Params | ~22M | 3.25M |
| Model Size | ~85 MB | 13.2 MB |

---

## System Status

### Working Components:
✅ **PyTorch Backend**: MobileNetV3-Large on CUDA (GTX 1050, 3GB)
✅ **Model Loading**: `cnn_pytorch_best.pth` loads at startup (96%+ val accuracy)
✅ **Image Preprocessing**: Correct 224×224 normalization with ImageNet stats
✅ **TTA**: 7 augmentations working (flip, rotation, brightness, zoom)
✅ **OpenCV Analyzer**: All 14 features detecting correctly
✅ **Ensemble Engine**: Dynamic weighting + critical override working
✅ **FastAPI Server**: All endpoints responding correctly
✅ **Database**: MySQL connection successful
✅ **Error Handling**: Invalid inputs rejected with proper error codes

---

## Performance Metrics

| Operation | Time | Status |
|-----------|------|--------|
| Image Preprocessing | <10ms | ✅ |
| OpenCV Feature Analysis (14 features) | ~750ms | ✅ |
| CNN Inference (GPU) | ~200ms | ✅ |
| CNN + TTA (7 augments) | ~500ms | ✅ |
| Total Analysis Time | ~1-2s | ✅ |

---

**Total Tests**: 12
**Passed**: 12/12 (100%)
**Failed**: 0
**Bugs Fixed**: 5
