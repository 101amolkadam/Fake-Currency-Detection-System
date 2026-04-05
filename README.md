# Fake Currency Detection System

**Advanced AI-Powered Counterfeit Detection for Indian Currency**

A production-ready full-stack application that uses PyTorch-based Xception CNN and 15 OpenCV-based security feature detectors to authenticate Indian banknotes (₹500, ₹2000) with **92.88% accuracy** and full explainability.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+cu118](https://img.shields.io/badge/PyTorch-2.7.1+cu118-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 🎯 Features

### AI-Powered Detection
- **PyTorch Xception CNN**: Pretrained on ImageNet, fine-tuned on 7,442 currency images with CUDA acceleration
- **15 Security Features**: Comprehensive analysis matching RBI specifications with independent scoring
- **Class-Balanced Training**: Handles 8.5:1 imbalance with weighted BCELoss (8.5x for fake notes)
- **92.88% Accuracy**: Validated on 800+ test images (78.33% fake detection, 100% real detection)
- **Critical Feature Override**: Prevents false positives when Security Thread/Watermark/Serial Number fail

### Security Feature Analysis

| Priority | Feature | Detection Method | Weight |
|----------|---------|-----------------|--------|
| **Critical** | Security Thread | HoughLinesP, intensity, texture | 22.5% |
| **Critical** | Watermark | Brightness, smoothness, edges | 18.8% |
| **Critical** | Serial Number | Tesseract OCR, format validation | 15.0% |
| Important | Optically Variable Ink | HSV color, green/blue ratio | 11.3% |
| Important | Latent Image | Edge patterns, texture | 9.0% |
| Important | Intaglio Printing | Edge density, variance, gradient | 9.0% |
| Important | See-Through Registration | Pattern matching | 7.5% |
| Supporting | Microlettering | High-res OCR | 6.0% |
| Supporting | Fluorescence | Brightness (UV required) | 5.3% |
| Supporting | Color Analysis | HSV histogram | 5.3% |
| Supporting | Texture | GLCM, Laplacian | 3.8% |
| Supporting | Dimensions | Contour detection | 3.8% |
| Supporting | Identification Mark | Shape detection | 3.8% |
| Supporting | Angular Lines | Hough lines, angle filtering | 2.3% |

### Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 92.88% |
| **Fake Detection** | 78.33% (206/263) |
| **Real Detection** | 100.00% (537/537) |
| **AUC Score** | 0.9947 |
| **Processing Time (CPU)** | 1-3 seconds |
| **Processing Time (GPU)** | 0.7-1.4 seconds |
| **False Positive Rate** | 0% |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- MySQL 8.0+ (or SQLite for testing)
- Tesseract OCR ([Windows installer](https://github.com/UB-Mannheim/tesseract/wiki))
- Optional: NVIDIA GPU with CUDA 11.8+ (GTX 1050 or better)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database URL

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

### Step 1: Download Dataset

Download from Kaggle: [Indian Currency Real vs Fake Notes](https://www.kaggle.com/datasets/preetrank/indian-currency-real-vs-fake-notes-dataset)

Extract to `backend/dataset_downloads/preetrank/` with structure:
```
preetrank/
├── fake/  (581 images)
└── real/  (4,937 images)
```

### Step 2: Train Model

```bash
cd backend
python cnn_classifier.py --epochs 15 --batch-size 8
```

**Training Configuration:**
- 15 epochs (8 head training + 7 fine-tuning)
- Class balancing: fake notes weighted 8.5x
- Heavy augmentation: rotation, zoom, flip, brightness, contrast
- Batch size 8 for stability
- Expected time: ~75 min (GPU), ~4 hours (CPU)

### Step 3: Validate Model

```bash
python validate_model.py --model-path models/cnn_pytorch_best.pth
```

**Expected Results:**
- Overall Accuracy: 90-95%
- Fake Detection: 75-85%
- Real Detection: 98-100%
- AUC Score: >0.95

---

## 📁 Project Structure

```
Fake Currency Detection System/
├── backend/
│   ├── services/
│   │   ├── cnn_classifier.py          # PyTorch Xception model + TTA
│   │   ├── opencv_analyzer.py          # 15 security feature detectors
│   │   ├── ensemble_engine.py          # Critical feature override logic
│   │   ├── image_preprocessor.py       # Image preparation (no TensorFlow)
│   │   └── image_annotator.py          # Result visualization
│   ├── routers/
│   │   ├── analyze.py                  # Main analysis endpoint
│   │   └── history.py                  # History management
│   ├── models/
│   │   ├── cnn_pytorch_best.pth        # Best checkpoint
│   │   └── schemas.py                  # Pydantic models
│   ├── requirements.txt                # Python dependencies (PyTorch)
│   ├── main.py                         # FastAPI app
│   ├── config.py                       # Pydantic settings
│   └── database.py                     # SQLAlchemy connection
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── HomePage.tsx            # Upload/camera input
│   │   │   ├── ResultsPage.tsx         # Detailed analysis
│   │   │   └── HistoryPage.tsx         # Past analyses
│   │   └── components/
│   │       ├── ImageUploader.tsx       # Drag & drop
│   │       ├── CameraCapture.tsx       # Webcam capture
│   │       └── AnalysisTable.tsx       # Feature results table
│   └── package.json
│
└── docs/
    └── IEEE_Research_Paper.md          # Complete IEEE-format paper
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
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check (model loaded, DB connected) |
| `/api/v1/model/info` | GET | Model metadata (architecture, status) |
| `/api/v1/analyze/history` | GET | Paginated analysis history |

---

## 📈 System Architecture

```
Input Image → Preprocessing → PyTorch CNN (Xception + TTA)
                                    ↓
                              REAL/FAKE + Confidence
                                    ↓
                    OpenCV Feature Analyzer (15 features)
                                    ↓
                    Ensemble Engine (Critical Override)
                                    ↓
                          Final Result + Explainability
                                    ↓
                              MySQL Database
```

---

## 🎓 Research & Validation

### Dataset Statistics

| Class | Count | Percentage |
|-------|-------|------------|
| Real | 4,937 | 89.5% |
| Fake | 581 | 10.5% |
| **Total** | **5,518** | **100%** |

**Balance Ratio**: 8.5:1 (real:fake) — handled with weighted BCELoss

### Training Methodology

1. **Phase 1** (Epochs 1-8): Train classification head with frozen backbone
2. **Phase 2** (Epochs 9-15): Fine-tune top 30% of backbone
3. **Class Balancing**: Fake notes weighted 8.5x to compensate for imbalance
4. **Augmentation**: Rotation (±20°), zoom (0.85-1.0x), flip, brightness, contrast
5. **TTA**: 7 augmented predictions averaged at inference

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
- ✅ CORS configured for production domains
- ✅ Rate limiting (10 req/min per IP)
- ✅ Model caching on startup
- ✅ Connection pooling (MySQL)
- ⚠️ Add authentication for production use
- ⚠️ Enable HTTPS
- ⚠️ Set up monitoring and alerting

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push and open Pull Request

### Areas for Improvement
- [ ] Add UV camera for fluorescence detection
- [ ] Multi-angle OVI verification
- [ ] Front+back image analysis
- [ ] Mobile app (React Native/Flutter)
- [ ] Federated learning with banks
- [ ] Support more denominations (₹10, ₹20, ₹50, ₹100, ₹200)
- [ ] MixUp/CutMix augmentation for +2-4% accuracy

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **Reserve Bank of India** for security feature specifications
- **Kaggle Contributors** (preetrank) for currency dataset
- **PyTorch Team** for the deep learning framework
- **OpenCV Team** for computer vision library

---

## ⚠️ Disclaimer

This system is a decision support tool, not a definitive authentication device. Always consult currency experts for critical authentication decisions. The system provides probabilistic assessments based on detectable features.

---

**Version**: 3.0 (PyTorch CUDA Edition)  
**Last Updated**: April 2026  
**Framework**: PyTorch 2.7.1+cu118 + FastAPI + React  
**GPU**: NVIDIA GTX 1050 (3GB, CUDA 11.8)
