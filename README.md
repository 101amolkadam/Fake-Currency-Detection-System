# Fake Currency Detection System

**Advanced AI-Powered Counterfeit Detection for Indian Currency**

A production-ready full-stack application that uses PyTorch-based MobileNetV3-Large CNN and 15 OpenCV-based security feature detectors to authenticate Indian banknotes (₹500, ₹2000) with **96%+ validation accuracy** and full explainability.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+cu118](https://img.shields.io/badge/PyTorch-2.7.1+cu118-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org)
[![CPU/GPU](https://img.shields.io/badge/Hardware-CPU%2FGPU-yellow.svg)](https://pytorch.org)

---

## 🎯 Features

### AI-Powered Detection
- **PyTorch MobileNetV3-Large CNN**: Pretrained on ImageNet, fine-tuned on 7,443 currency images with **CPU/GPU support** (no GPU required)
- **15 Security Features**: Comprehensive analysis matching RBI specifications with independent scoring
- **Class-Balanced Training**: Handles 2:1 imbalance with weighted BCELoss (1.48x for fake notes)
- **96%+ Validation Accuracy**: Achieved during training on 1,116 validation images
- **Test-Time Augmentation (TTA)**: 7 augmented predictions averaged at inference for robustness
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
| **Validation Accuracy** | 96%+ |
| **Dataset Size** | 7,443 images (2,506 fake, 4,937 real) |
| **Denominations** | ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000 |
| **Model Size** | 13.2 MB (MobileNetV3-Large) |
| **Processing Time (GPU)** | 0.5-1.5 seconds |
| **Processing Time (CPU)** | 3-6 seconds |
| **Class Balance** | 2:1 (real:fake) with weighted loss |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Node.js 18+
- MySQL 8.0+ (or SQLite for testing)
- Tesseract OCR ([Windows installer](https://github.com/UB-Mannheim/tesseract/wiki))
- **GPU Optional**: NVIDIA GPU with CUDA 11.8+ recommended but **NOT required** - system works fully on CPU

### Backend Setup

```bash
cd backend

# Create virtual environment
# python -m venv .venv
# .venv\Scripts\activate  # Windows
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

### Step 1: Prepare Dataset

Place your currency note images in the following structure under `backend/dataset_downloads/preetrank/`:

```
preetrank/
├── fake/
│   ├── 10/    (₹10 fake notes)
│   ├── 20/    (₹20 fake notes)
│   ├── 50/    (₹50 fake notes)
│   ├── 100/   (₹100 fake notes)
│   ├── 200/   (₹200 fake notes)
│   ├── 500/   (₹500 fake notes)
│   └── 2000/  (₹2000 fake notes)
└── real/
    ├── 10/    (₹10 real notes)
    ├── 20/    (₹20 real notes)
    ├── 50/    (₹50 real notes)
    ├── 100/   (₹100 real notes)
    ├── 200/   (₹200 real notes)
    ├── 500/   (₹500 real notes)
    └── 2000/  (₹2000 real notes)
```

**Or use a flat structure:** `fake/*.jpg` and `real/*.jpg`

**Current Dataset**: 7,443 images (2,506 fake, 4,937 real) across 7 denominations.

### Step 2: Train Model

```bash
cd backend

# Train with default settings (uses all images, 85/15 train/val split)
python train_fast.py

# Or use the original script
python cnn_classifier.py --epochs 15 --batch-size 8
```

**Training Configuration (train_fast.py):**
- **Architecture**: MobileNetV3-Large (3.25M params, ImageNet pretrained)
- **10 epochs** (5 head training + 5 fine-tuning with 30% backbone unfreeze)
- **Class balancing**: weighted BCELoss (1.48x for fake, 0.76x for real)
- **Augmentation**: RandomResizedCrop, horizontal flip, color jitter
- **Batch size**: 32 (optimized for GTX 1050 3GB)
- **Input size**: 224×224 (ImageNet normalization)
- **Expected time**: ~14 min/epoch on GTX 1050, ~2.5 hours total

### Step 3: Validate Model

After training, the model is automatically saved to:
- `models/cnn_pytorch_best.pth` — Best checkpoint (highest val accuracy)
- `models/cnn_pytorch_final.pth` — Final model after all epochs
- `models/training_history_pytorch.json` — Training metrics

**Expected Results:**
- Validation Accuracy: 95-98%
- Model Size: ~13 MB
- Inference Time: 0.5-1.5s (GPU with TTA)

---

## 📁 Project Structure

```
Fake-Currency-Detection-System/
│
├── backend/
│   ├── services/
│   │   ├── model_loader.py             # Startup model loading
│   │   ├── cnn_classifier.py           # PyTorch Xception inference + TTA
│   │   ├── opencv_analyzer.py          # 15 security feature detectors
│   │   ├── ensemble_engine.py          # Critical feature override logic
│   │   ├── image_preprocessor.py       # Image preparation pipeline
│   │   └── image_annotator.py          # Result visualization
│   ├── routers/
│   │   ├── analyze.py                  # Main analysis endpoint
│   │   └── history.py                  # History management
│   ├── orm_models/
│   │   └── analysis.py                 # SQLAlchemy models
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                  # Pydantic response models
│   ├── cnn_classifier.py               # Training script (standalone)
│   ├── main.py                         # FastAPI application
│   ├── config.py                       # Pydantic settings
│   ├── database.py                     # SQLAlchemy connection
│   ├── requirements.txt                # pip dependencies
│   ├── pyproject.toml                  # uv project config
│   ├── test_api.py                     # API endpoint tests
│   ├── test_application.py             # Unit/integration tests
│   └── TEST_REPORT.md                  # Test results report
│
├── frontend/                           # React + Vite + TypeScript
│   ├── src/
│   │   ├── pages/
│   │   │   ├── HomePage.tsx            # Upload/camera input
│   │   │   ├── ResultsPage.tsx         # Detailed analysis display
│   │   │   └── HistoryPage.tsx         # Past analyses view
│   │   └── components/
│   │       ├── ImageUploader.tsx       # Drag & drop component
│   │       ├── CameraCapture.tsx       # Webcam capture component
│   │       └── AnalysisTable.tsx       # Feature results table
│   ├── package.json
│   └── vite.config.ts
│
├── test_images/                        # Sample test currency images
│   ├── 500_dataset/                    # ₹500 note test images
│   └── 2000_dataset/                   # ₹2000 note test images
│
├── docs/                               # Documentation files
│   ├── IEEE_Research_Paper.docx
│   ├── setup.docx
│   ├── technical_documentation.docx
│   └── user_guide.docx
│
└── README.md
```

---

## 📂 Training Dataset Folder Structure

**⚠️ No trained model is currently present.** You need to train the model before using the system.

### Where to Place Training Data

Create the following folder structure under `backend/`:

```
backend/
└── dataset_downloads/
    └── preetrank/
        ├── fake/                       # Place FAKE currency note images here
        │   ├── fake_note_1.jpg
        │   ├── fake_note_2.jpg
        │   └── ...
        └── real/                       # Place REAL currency note images here
            ├── real_note_1.jpg
            ├── real_note_2.jpg
            └── ...
```

### Steps to Prepare Training Data:

1. **Download Dataset** from Kaggle: [Indian Currency Real vs Fake Notes](https://www.kaggle.com/datasets/preetrank/indian-currency-real-vs-fake-notes-dataset)

2. **Extract** the downloaded zip file so that:
   - All **fake** currency images go into: `backend/dataset_downloads/preetrank/fake/`
   - All **real** currency images go into: `backend/dataset_downloads/preetrank/real/`

3. **Supported image formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`

4. **Recommended dataset size** (as per project research):
   - Real images: ~4,937
   - Fake images: ~581
   - Balance ratio: 8.5:1 (handled automatically with weighted loss)

### Training the Model:

```bash
cd backend

# Train with default settings
python cnn_classifier.py --epochs 15 --batch-size 8

# Or specify custom dataset path
python cnn_classifier.py --train-dir dataset_downloads/preetrank --val-dir dataset_downloads/preetrank --epochs 15
```

**After training completes**, the model files will be saved to:
- `backend/models/cnn_pytorch_best.pth` — Best checkpoint (auto-saved during training)
- `backend/models/cnn_pytorch_final.pth` — Final model after all epochs
- `backend/models/training_history_pytorch.json` — Training metrics log

The system will automatically load `cnn_pytorch_best.pth` on startup when present.

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
Input Image → Preprocessing → PyTorch CNN (MobileNetV3-Large + TTA)
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
| Real | 4,937 | 66.3% |
| Fake | 2,506 | 33.7% |
| **Total** | **7,443** | **100%** |

**Balance Ratio**: 2:1 (real:fake) — handled with weighted BCELoss (1.48x for fake)

**Denominations**: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000 (organized in subfolders)

### Training Methodology

1. **Phase 1** (Epochs 1-5): Train classification head with frozen backbone
2. **Phase 2** (Epochs 6-10): Fine-tune top 30% of backbone
3. **Class Balancing**: Weighted BCELoss (1.48x fake, 0.76x real)
4. **Augmentation**: RandomResizedCrop (85-100%), horizontal flip, color jitter
5. **TTA**: 7 augmented predictions averaged at inference (flip, rotation, brightness, zoom)
6. **Temperature Scaling**: Confidence calibration with T=1.5

### Validation Results (25% subset, 10 epochs)

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | 96-98% |
| **Model Architecture** | MobileNetV3-Large (3.25M params) |
| **Model Size** | 13.2 MB |
| **Training Time** | ~14 min/epoch (GTX 1050) |
| **Inference Time** | 0.5-1.5s (GPU + TTA) |

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

**Version**: 5.0 (MobileNetV3-Large PyTorch CPU/GPU Edition)  
**Last Updated**: April 14, 2026  
**Framework**: PyTorch 2.7.1+cu118 + FastAPI + React  
**Hardware**: CPU-only or NVIDIA GPU with CUDA 11.8+ (both fully supported)  
**Model**: MobileNetV3-Large (3.25M params, 13.2 MB checkpoint)  
**Dataset**: 7,443 images (2,506 fake, 4,937 real) across 7 denominations
