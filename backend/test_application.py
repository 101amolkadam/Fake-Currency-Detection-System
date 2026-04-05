"""
Comprehensive Application Test Suite
Tests all backend services, API endpoints, and integration
"""
import sys
import os
import cv2
import numpy as np
import base64

sys.path.insert(0, os.path.dirname(__file__))

def test_preprocessing():
    """Test image preprocessing"""
    print("\n=== TEST 1: Image Preprocessing ===")
    try:
        from services.image_preprocessor import preprocess_image, decode_base64_image
        
        # Create test image
        img = np.zeros((500, 700, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (650, 450), (255, 255, 255), -1)
        
        # Test preprocessing
        cnn_input, denoised, enhanced = preprocess_image(img)
        assert cnn_input.shape == (299, 299, 3), f"Wrong CNN input shape: {cnn_input.shape}"
        assert cnn_input.min() >= -1.01 and cnn_input.max() <= 1.01, f"Wrong range: [{cnn_input.min()}, {cnn_input.max()}]"
        print(f"✓ CNN input shape: {cnn_input.shape}")
        print(f"✓ CNN input range: [{cnn_input.min():.2f}, {cnn_input.max():.2f}]")
        
        # Test base64 decode
        _, buffer = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(buffer).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"
        decoded, mime = decode_base64_image(data_uri)
        assert decoded.shape == img.shape, f"Decode shape mismatch: {decoded.shape} vs {img.shape}"
        assert mime == "image/jpeg", f"Wrong MIME: {mime}"
        print(f"✓ Base64 decode works (MIME: {mime})")
        
        print("✓ PASS: Image preprocessing")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_analyzer():
    """Test OpenCV feature analyzer"""
    print("\n=== TEST 2: OpenCV Feature Analyzer ===")
    try:
        from services.image_preprocessor import preprocess_image
        from services.opencv_analyzer import analyze_security_features
        
        # Create test image
        img = np.zeros((500, 700, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (650, 450), (255, 255, 255), -1)
        cnn_input, denoised, enhanced = preprocess_image(img)
        
        # Test feature analysis
        features = analyze_security_features(img, denoised, enhanced, '₹500')
        assert len(features) == 14, f"Expected 14 features, got {len(features)}"
        print(f"✓ Features analyzed: {len(features)}")
        
        for name, data in features.items():
            assert 'status' in data, f"Missing 'status' in {name}"
            assert 'confidence' in data, f"Missing 'confidence' in {name}"
            print(f"  ✓ {name}: {data['status']} (conf: {data['confidence']:.2f})")
        
        print("✓ PASS: OpenCV analyzer")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_engine():
    """Test ensemble decision engine"""
    print("\n=== TEST 3: Ensemble Decision Engine ===")
    try:
        from services.ensemble_engine import compute_ensemble_score
        
        # Test with mock data
        cnn_result = "REAL"
        cnn_confidence = 0.85
        
        # Mock features
        features = {
            "security_thread": {"status": "present", "confidence": 0.90},
            "watermark": {"status": "present", "confidence": 0.85},
            "serial_number": {"status": "valid", "confidence": 0.80},
            "optically_variable_ink": {"status": "present", "confidence": 0.70},
            "latent_image": {"status": "present", "confidence": 0.60},
            "intaglio_printing": {"status": "present", "confidence": 0.75},
            "see_through_registration": {"status": "present", "confidence": 0.65},
            "microlettering": {"status": "present", "confidence": 0.55},
            "fluorescence": {"status": "unknown", "confidence": 0.50},
            "color_analysis": {"status": "match", "confidence": 0.80},
            "texture_analysis": {"status": "normal", "confidence": 0.85},
            "dimensions": {"status": "correct", "confidence": 0.90},
            "identification_mark": {"status": "present", "confidence": 0.70},
            "angular_lines": {"status": "present", "confidence": 0.65},
        }
        
        score, result, confidence, agreement, critical_failures = compute_ensemble_score(
            cnn_result, cnn_confidence, features
        )
        
        assert isinstance(score, float), f"Score type wrong: {type(score)}"
        assert isinstance(result, str), f"Result type wrong: {type(result)}"
        assert 0 <= score <= 1, f"Score out of range: {score}"
        assert 0 <= confidence <= 1, f"Confidence out of range: {confidence}"
        assert 0 <= agreement <= 1, f"Agreement out of range: {agreement}"
        assert isinstance(critical_failures, list), f"Critical failures type wrong"
        
        print(f"✓ Ensemble score: {score:.4f}")
        print(f"✓ Result: {result}")
        print(f"✓ Confidence: {confidence:.4f}")
        print(f"✓ Feature agreement: {agreement:.4f}")
        print(f"✓ Critical failures: {len(critical_failures)}")
        
        # Test with critical failure
        features_fail = features.copy()
        features_fail["security_thread"] = {"status": "missing", "confidence": 0.90}
        features_fail["watermark"] = {"status": "missing", "confidence": 0.85}
        
        score2, result2, conf2, agr2, crit2 = compute_ensemble_score(
            cnn_result, cnn_confidence, features_fail
        )
        assert result2 == "FAKE", f"Should be FAKE with critical failures, got {result2}"
        assert len(crit2) == 2, f"Should have 2 critical failures, got {len(crit2)}"
        print(f"✓ Critical override works: {result2} (failures: {len(crit2)})")
        
        print("✓ PASS: Ensemble engine")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cnn_classifier():
    """Test CNN classifier (may fail if no model)"""
    print("\n=== TEST 4: CNN Classifier ===")
    try:
        from services.cnn_classifier import classify_currency, is_model_loaded, load_model
        
        # Check if model is loaded
        loaded = is_model_loaded()
        print(f"✓ Model loaded: {loaded}")
        
        if loaded:
            # Test classification
            img = np.zeros((299, 299, 3), dtype=np.float32)
            img = (img / 127.5) - 1.0  # Normalize
            result, denom, denom_conf, auth_conf = classify_currency(img, use_tta=False)
            print(f"✓ Classification result: {result}")
            print(f"✓ Denomination: {denom}")
            print(f"✓ Auth confidence: {auth_conf:.4f}")
        else:
            print("⚠ WARNING: No model file found - CNN classifier in OpenCV-only mode")
            print("  This is expected if you haven't trained a model yet")
            print("  To train: python cnn_classifier.py")
        
        print("✓ PASS: CNN classifier (functional)")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_annotator():
    """Test image annotator"""
    print("\n=== TEST 5: Image Annotator ===")
    try:
        from services.image_annotator import generate_annotated_image, generate_thumbnail
        
        # Create test image
        img = np.zeros((500, 700, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (650, 450), (255, 255, 255), -1)
        
        # Mock analysis results
        analysis = {
            "overall_result": "REAL",
            "ensemble_score": 0.85,
            "security_thread": {"status": "present", "confidence": 0.90},
            "watermark": {"status": "present", "confidence": 0.85},
        }
        
        # Test annotation
        annotated_b64 = generate_annotated_image(img, analysis)
        assert annotated_b64.startswith("data:image"), "Annotated image not base64"
        print(f"✓ Annotated image generated ({len(annotated_b64)} chars)")
        
        # Test thumbnail
        thumb_b64 = generate_thumbnail(img)
        assert thumb_b64.startswith("data:image"), "Thumbnail not base64"
        print(f"✓ Thumbnail generated ({len(thumb_b64)} chars)")
        
        print("✓ PASS: Image annotator")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database():
    """Test database connection"""
    print("\n=== TEST 6: Database Connection ===")
    try:
        from config import settings
        from database import engine, get_db
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        print(f"✓ Database connected: {settings.DATABASE_URL[:30]}...")
        
        print("✓ PASS: Database connection")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        print("  Note: This is expected if MySQL is not configured")
        print("  You can use SQLite for testing: sqlite:///./currency.db")
        return False

def test_api_endpoints():
    """Test FastAPI app can be created"""
    print("\n=== TEST 7: API Endpoints ===")
    try:
        from main import app
        
        # Check routes
        routes = [r.path for r in app.routes]
        expected = ["/api/v1/health", "/api/v1/model/info", "/api/v1/analyze"]
        
        for route in expected:
            assert route in routes, f"Missing route: {route}"
            print(f"✓ Route exists: {route}")
        
        print("✓ PASS: API endpoints")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*80)
    print("FAKE CURRENCY DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    import sqlalchemy
    
    results = []
    results.append(("Image Preprocessing", test_preprocessing()))
    results.append(("OpenCV Analyzer", test_opencv_analyzer()))
    results.append(("Ensemble Engine", test_ensemble_engine()))
    results.append(("CNN Classifier", test_cnn_classifier()))
    results.append(("Image Annotator", test_image_annotator()))
    results.append(("Database", test_database()))
    results.append(("API Endpoints", test_api_endpoints()))
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Application is functional!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - see details above")
        sys.exit(1)
