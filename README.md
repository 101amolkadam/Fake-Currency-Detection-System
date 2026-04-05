# Fake Currency Detection System

**Advanced AI-Powered Counterfeit Detection for Indian Currency**

A production-ready full-stack application that uses PyTorch-based Xception CNN and 15 OpenCV-based security feature detectors to authenticate Indian banknotes (₹500, ₹2000) with **92.88% accuracy** and full explainability.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org)

---

## 🎯 Features

### AI-Powered Detection
- **PyTorch Xception CNN**: Pretrained on ImageNet, fine-tuned on 7,442 currency images
- **15 Security Features**: Comprehensive analysis matching RBI specifications
- **Class-Balanced Training**: Handles imbalanced datasets with automatic weighting
- **92.88% Accuracy**: Validated on 800+ test images
- **78.33% Fake Detection Rate**: Dramatically improved from 0% with balanced training

### Security Feature Analysis
| Feature | Detection Method |
|---------|-----------------|
| **Security Thread** (22.5%) | Vertical lines, intensity, texture, color shift |
| **Watermark** (18.8%) | Brightness variation, smoothness, edge density |
| **Serial Number** (15.0%) | Tesseract OCR, format validation, progressive sizing |
| **Optically Variable Ink** (11.3%) | HSV color analysis (green→blue shift) |
| **Latent Image** (9.0%) | Edge patterns, texture analysis |
| **Intaglio Printing** (9.0%) | Edge density, variance, gradient magnitude |
| **See-Through Registration** (7.5%) | Pattern matching, line detection |
| **Microlettering** (6.0%) | High-res OCR, edge density |
| **Fluorescence** (5.3%) | Brightness analysis (UV required) |
| **Color Analysis** (5.3%) | HSV histogram uniformity |
| **Texture** (3.8%) | GLCM, Laplacian variance |
| **Dimensions** (3.8%) | Contour detection, aspect ratio |
| **Identification Mark** (3.8%) | Shape detection (circle/square/etc) |
| **Angular Lines** (2.3%) | Hough lines with angle filtering |

### Critical Feature Override
If ANY critical feature (Security Thread, Watermark, Serial Number) fails, the note is marked **FAKE** regardless of CNN prediction. This prevents false positives and mimics expert authentication.

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 92.88% |
| **Fake Detection** | 78.33% |
| **Real Detection** | 100.00% |
| **AUC Score** | 0.9947 |
| **Processing Time** | 1-3 seconds (CPU) |
| **False Positive Rate** | <3% |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- MySQL 8.0+ (or SQLite for testing)
- Tesseract OCR: `apt install tesseract-ocr` or download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Access Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000/docs (Swagger UI)
- **Health Check**: http://localhost:8000/api/v1/health

---

## 🔬 Training Your Own Model

### Dataset Preparation

1. **Download Dataset** from Kaggle:
   - [Indian Currency Real vs Fake Notes](https://www.kaggle.com/datasets/preetrank/indian-currency-real-vs-fake-notes-dataset)
   
2. **Extract** to `backend/dataset_downloads/preetrank/`:
   ```
   preetrank/
   ├── fake/
   └── real/
   ```

### Train Model

```bash
cd backend
python cnn_classifier.py --epochs 15 --batch-size 8
```

**Training Configuration:**
- 15 epochs (8 head training + 7 fine-tuning)
- Class balancing (fake notes weighted 8.5x)
- Heavy augmentation (rotation, zoom, flip, brightness, contrast)
- Batch size 8 for stability

**Expected Results:**
- Training time: ~4 hours (CPU), ~45 min (GPU)
- Validation accuracy: 90-95%
- Fake detection: 75-85%

### Validate Model

```bash
python validate_model.py --model-path models/cnn_pytorch_best.pth
```

---

## 📁 Project Structure

```
Fake Currency Detection System/
├── backend/
│   ├── services/
│   │   ├── cnn_classifier.py          # PyTorch Xception model
│   │   ├── opencv_analyzer.py          # 15 security feature detectors
│   │   ├── ensemble_engine.py          # Critical feature override logic
│   │   ├── image_preprocessor.py       # Image preparation
│   │   └── image_annotator.py          # Result visualization
│   ├── routers/
│   │   ├── analyze.py                  # Main analysis endpoint
│   │   └── history.py                  # History management
│   ├── models/
│   │   ├── cnn_pytorch_best.pth        # Best checkpoint
│   │   └── schemas.py                  # Pydantic models
│   ├── requirements.txt                # Python dependencies
│   ├── main.py                         # FastAPI app
│   ├── config.py                       # Configuration
│   └── database.py                     # Database connection
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── HomePage.tsx            # Upload/camera
│   │   │   ├── ResultsPage.tsx         # Analysis results
│   │   │   └── HistoryPage.tsx         # Past analyses
│   │   └── components/
│   │       ├── ImageUploader.tsx       # Drag & drop
│   │       ├── CameraCapture.tsx       # Webcam
│   │       └── AnalysisTable.tsx       # Feature results
│   └── package.json
│
└── docs/
    └── IEEE_Research_Paper.md          # Complete research paper
```

---

## 🛠️ API Reference

### Analyze Currency Image
```http
POST /api/v1/analyze
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,...",
  "source": "upload"
}
```

**Response:**
```json
{
  "result": "REAL",
  "confidence": 0.9288,
  "currency_denomination": "₹500",
  "analysis": {
    "security_thread": {"status": "present", "confidence": 0.95},
    "watermark": {"status": "present", "confidence": 0.91},
    "serial_number": {"status": "valid", "confidence": 0.88},
    "critical_failures": [],
    "feature_agreement": 0.93
  },
  "processing_time_ms": 1850
}
```

### Other Endpoints
- `GET /api/v1/health` - Health check
- `GET /api/v1/model/info` - Model metadata
- `GET /api/v1/analyze/history` - Analysis history

---

## 📈 System Architecture

```
Input Image → Preprocessing → PyTorch CNN (Xception)
                                    ↓
                              REAL/FAKE + Confidence
                                    ↓
                    OpenCV Feature Analyzer (15 features)
                                    ↓
                    Ensemble Engine (Critical Override)
                                    ↓
                          Final Result + Explainability
```

---

## 🎓 Research & Validation

### Dataset Statistics
- **Total Images**: 7,442
- **Real Notes**: 4,937 (66.4%)
- **Fake Notes**: 581 (7.8%) + augmented samples
- **Balance Ratio**: 8.5:1 (handled with class weights)

### Training Methodology
1. **Phase 1** (Epochs 1-8): Train classification head with frozen backbone
2. **Phase 2** (Epochs 9-15): Fine-tune top 30% of backbone
3. **Class Balancing**: Fake notes weighted 8.5x to compensate for imbalance
4. **Augmentation**: Rotation (±20°), zoom (0.85-1.0x), flip, brightness, contrast

### Validation Results
| Metric | Before Balancing | After Balancing | Improvement |
|--------|------------------|-----------------|-------------|
| Overall Accuracy | 87.50% | **92.88%** | +5.37% |
| Fake Detection | 0.00% | **78.33%** | +78.33% |
| Real Detection | 100.00% | **100.00%** | Maintained |
| AUC Score | 0.9048 | **0.9947** | +0.0899 |

---

## 🌐 Deployment

### Production URLs
- **Backend**: https://validcash.duckdns.org
- **Frontend**: https://validcash.netlify.app

### Production Considerations
- ✅ CORS configured
- ✅ Rate limiting (10 req/min)
- ✅ Model caching
- ✅ Connection pooling
- ⚠️ Add authentication for production
- ⚠️ Enable HTTPS
- ⚠️ Set up monitoring

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push and open PR

### Areas for Improvement
- [ ] Add UV camera for fluorescence detection
- [ ] Multi-angle OVI verification
- [ ] Front+back image analysis
- [ ] Mobile app (React Native/Flutter)
- [ ] Federated learning with banks
- [ ] Support more denominations (₹10, ₹20, ₹50, ₹100, ₹200)

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **Reserve Bank of India** for security feature specifications
- **Kaggle Contributors** for currency datasets
- **PyTorch Team** for the deep learning framework
- **OpenCV Team** for computer vision library

---

## ⚠️ Disclaimer

This system is a decision support tool, not a definitive authentication device. Always consult currency experts for critical authentication decisions. The system provides probabilistic assessments based on detectable features.

---

**Version**: 2.0 (PyTorch Edition)  
**Last Updated**: April 2026  
**Framework**: PyTorch 2.x + FastAPI + React
