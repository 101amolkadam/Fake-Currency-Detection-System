"""Quick CORS test script - run this against your deployed backend."""
import requests

# Test URLs
BACKEND_URL = "https://validcash.duckdns.org"
TEST_ORIGINS = [
    "http://localhost:5173",
    "https://validcash.netlify.app",
]

def test_cors_preflight():
    """Test CORS preflight (OPTIONS) request."""
    print("\n" + "="*60)
    print("Testing CORS Preflight (OPTIONS)")
    print("="*60)
    
    for origin in TEST_ORIGINS:
        print(f"\nTesting origin: {origin}")
        try:
            response = requests.options(
                f"{BACKEND_URL}/api/v1/analyze",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                },
                timeout=10
            )
            
            # Check CORS headers
            allow_origin = response.headers.get("Access-Control-Allow-Origin", "MISSING")
            allow_methods = response.headers.get("Access-Control-Allow-Methods", "MISSING")
            allow_headers = response.headers.get("Access-Control-Allow-Headers", "MISSING")
            
            print(f"  ✓ Status: {response.status_code}")
            print(f"  ✓ Allow-Origin: {allow_origin}")
            print(f"  ✓ Allow-Methods: {allow_methods}")
            print(f"  ✓ Allow-Headers: {allow_headers}")
            
            if allow_origin == origin or allow_origin == "*":
                print(f"  ✅ CORS OK for {origin}")
            else:
                print(f"  ❌ CORS FAILED for {origin}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_health_endpoint():
    """Test if backend is reachable."""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=10)
        print(f"\n✓ Status: {response.status_code}")
        print(f"✓ Response: {response.json()}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def test_actual_request():
    """Test actual POST request with CORS."""
    print("\n" + "="*60)
    print("Testing Actual POST Request")
    print("="*60)
    
    # Minimal test payload (invalid base64, should return 400 not CORS error)
    test_payload = {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",  # Minimal valid JPEG
        "source": "upload"
    }
    
    for origin in TEST_ORIGINS:
        print(f"\nTesting origin: {origin}")
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/v1/analyze",
                json=test_payload,
                headers={"Origin": origin},
                timeout=30
            )
            
            allow_origin = response.headers.get("Access-Control-Allow-Origin", "MISSING")
            
            print(f"  ✓ Status: {response.status_code}")
            print(f"  ✓ Allow-Origin: {allow_origin}")
            
            if response.status_code == 400:
                print(f"  ✅ Request reached backend (validation error is expected)")
            elif response.status_code == 200:
                print(f"  ✅ Success!")
            else:
                print(f"  ⚠️  Unexpected status: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("CORS Test Suite for Fake Currency Detection Backend")
    print(f"Backend URL: {BACKEND_URL}")
    
    test_health_endpoint()
    test_cors_preflight()
    test_actual_request()
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)
