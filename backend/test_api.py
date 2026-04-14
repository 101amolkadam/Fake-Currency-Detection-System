"""
Test API endpoints using FastAPI TestClient
Tests actual HTTP requests to the backend
"""
import sys
import os
import cv2
import numpy as np
import base64
import json

sys.path.insert(0, os.path.dirname(__file__))

def create_test_image_base64():
    """Create a test currency image and encode as base64"""
    img = np.zeros((500, 700, 3), dtype=np.uint8)
    # Draw white rectangle (simulating currency note)
    cv2.rectangle(img, (50, 50), (650, 450), (255, 255, 255), -1)
    # Add some text
    cv2.putText(img, "TEST NOTE", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{b64}"

def test_health_endpoint():
    """Test /api/v1/health endpoint"""
    print("\n=== TEST 1: Health Endpoint ===")
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    response = client.get("/api/v1/health")

    assert response.status_code == 200, f"Status: {response.status_code}"
    data = response.json()
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Response keys: {list(data.keys())}")
    print(f"  Model loaded: {data.get('model_loaded', False)}")
    print(f"  DB connected: {data.get('db_connected', False)}")

    print("✓ PASS: Health endpoint")

def test_model_info_endpoint():
    """Test /api/v1/model/info endpoint"""
    print("\n=== TEST 2: Model Info Endpoint ===")
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    response = client.get("/api/v1/model/info")

    assert response.status_code == 200, f"Status: {response.status_code}"
    data = response.json()
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Model info: {list(data.keys())}")

    print("✓ PASS: Model info endpoint")

def test_analyze_endpoint():
    """Test /api/v1/analyze endpoint"""
    print("\n=== TEST 3: Analyze Endpoint ===")
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # Create test image
    test_image = create_test_image_base64()
    print(f"✓ Test image created ({len(test_image)} chars)")

    # Send request
    response = client.post(
        "/api/v1/analyze",
        json={
            "image": test_image,
            "source": "upload"  # Fixed: must be "upload" or "camera"
        }
    )

    assert response.status_code == 200, f"Status: {response.status_code}\nResponse: {response.text[:500]}"
    data = response.json()

    # Verify response structure
    assert "result" in data, "Missing 'result'"
    assert "confidence" in data, "Missing 'confidence'"
    assert "analysis" in data, "Missing 'analysis'"
    assert "ensemble_score" in data, "Missing 'ensemble_score'"
    assert "processing_time_ms" in data, "Missing 'processing_time_ms'"

    print(f"✓ Status: {response.status_code}")
    print(f"✓ Result: {data['result']}")
    print(f"✓ Confidence: {data['confidence']:.4f}")
    print(f"✓ Ensemble score: {data['ensemble_score']:.4f}")
    print(f"✓ Processing time: {data['processing_time_ms']}ms")

    # Verify analysis has all features
    analysis = data['analysis']
    expected_features = [
        'cnn_classification', 'watermark', 'security_thread',
        'color_analysis', 'texture_analysis', 'serial_number',
        'dimensions', 'intaglio_printing', 'latent_image',
        'optically_variable_ink', 'microlettering',
        'identification_mark', 'angular_lines', 'fluorescence',
        'see_through_registration'
    ]

    for feature in expected_features:
        assert feature in analysis, f"Missing feature: {feature}"
        print(f"  ✓ {feature}: {analysis[feature].get('status', 'N/A')}")

    print("✓ PASS: Analyze endpoint")

def test_invalid_image():
    """Test analyze endpoint with invalid image"""
    print("\n=== TEST 4: Invalid Image Handling ===")
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # Send invalid base64
    response = client.post(
        "/api/v1/analyze",
        json={
            "image": "data:image/jpeg;base64,invalid_base64_data_here!!!",
            "source": "upload"
        }
    )

    # Should return 400 or 422 or 500 (expected error for invalid data)
    assert response.status_code in [400, 422, 500], f"Expected error, got {response.status_code}"
    print(f"✓ Correctly rejected invalid image (status: {response.status_code})")

    print("✓ PASS: Invalid image handling")

def test_history_endpoint():
    """Test history endpoints (requires DB)"""
    print("\n=== TEST 5: History Endpoints ===")
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # Get history (should work even if empty)
    response = client.get("/api/v1/analyze/history")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ History endpoint works")
        print(f"  Items: {data.get('total', 0)}")
        print("✓ PASS: History endpoints")
    else:
        print(f"⚠ History endpoint returned {response.status_code}")
        print("  This is expected if DB is not configured")

if __name__ == "__main__":
    print("="*80)
    print("FAKE CURRENCY DETECTION SYSTEM - API ENDPOINT TESTS")
    print("="*80)
    
    results = []
    results.append(("Health Endpoint", test_health_endpoint()))
    results.append(("Model Info Endpoint", test_model_info_endpoint()))
    results.append(("Analyze Endpoint", test_analyze_endpoint()))
    results.append(("Invalid Image Handling", test_invalid_image()))
    results.append(("History Endpoints", test_history_endpoint()))
    
    print("\n" + "="*80)
    print("API TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} API tests passed")
    
    if passed == total:
        print("\n✓ ALL API TESTS PASSED!")
    else:
        print(f"\n⚠ {total - passed} API test(s) failed")
        sys.exit(1)
