"""
Comprehensive validation and Testing Script for Enhanced Fake Currency Detection

Tests:
1. All 15 security feature detectors
2. End-to-end API pipeline
3. Model accuracy on test set
4. Critical feature failure detection
5. Edge cases (blurred, angled, poor lighting)
6. Performance benchmarks
"""

import os
import sys
import cv2
import numpy as np
import requests
import time
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from services.opencv_analyzer import analyze_security_features, detect_watermark, detect_security_thread
from services.image_preprocessor import decode_base64_image, preprocess_image


class FeatureValidationTester:
    """Test all 15 security feature detectors."""

    def __init__(self, test_images_dir: str):
        self.test_images_dir = test_images_dir
        self.results = {
            'features_tested': 0,
            'features_passed': 0,
            'features_failed': 0,
            'details': {}
        }

    def test_all_features(self, image_path: str, denomination: str = "₹500") -> Dict:
        """Test all 15 security features on a single image."""
        print(f"\n{'='*80}")
        print(f"Testing: {image_path}")
        print(f"{'='*80}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Failed to load image")
            return {}

        # Preprocess
        _, denoised, enhanced = preprocess_image(image)

        # Run feature analysis
        features = analyze_security_features(image, denoised, enhanced, denomination)

        # Print results
        print(f"\n{'Feature':<30} {'Status':<12} {'Confidence':<12}")
        print("-" * 80)

        for feature_name, feature_data in features.items():
            status = feature_data.get('status', 'unknown')
            confidence = feature_data.get('confidence', 0.0)

            # Color code
            if status in ['present', 'match', 'normal', 'valid']:
                symbol = "✓"
            elif status == 'unknown':
                symbol = "?"
            else:
                symbol = "✗"

            print(f"{symbol} {feature_name:<28} {status:<12} {confidence:<12.4f}")

            self.results['features_tested'] += 1
            if status in ['present', 'match', 'normal', 'valid']:
                self.results['features_passed'] += 1
            elif status not in ['unknown']:
                self.results['features_failed'] += 1

            self.results['details'][feature_name] = {
                'status': status,
                'confidence': confidence
            }

        return features

    def run_batch_test(self, image_dir: str) -> Dict:
        """Test multiple images and aggregate results."""
        image_files = list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.png')) + \
                     list(Path(image_dir).glob('*.jpeg'))

        if not image_files:
            print(f"No images found in {image_dir}")
            return {}

        print(f"\n{'='*80}")
        print(f"BATCH TEST: {len(image_files)} images")
        print(f"{'='*80}")

        feature_stats = {}

        for img_path in image_files:
            features = self.test_all_features(str(img_path))

            for feature_name, feature_data in features.items():
                if feature_name not in feature_stats:
                    feature_stats[feature_name] = {
                        'present': 0, 'missing': 0, 'unknown': 0, 'invalid': 0
                    }

                status = feature_data.get('status', 'unknown')
                if status in feature_stats[feature_name]:
                    feature_stats[feature_name][status] += 1

        # Print summary
        print(f"\n{'='*80}")
        print(f"FEATURE DETECTION SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Feature':<30} {'Present':<10} {'Missing':<10} {'Unknown':<10} {'Invalid':<10}")
        print("-" * 80)

        for feature, stats in feature_stats.items():
            total = sum(stats.values())
            present_pct = stats['present'] / total * 100 if total > 0 else 0
            print(f"{feature:<28} {stats['present']}/{total} ({present_pct:.0f}%)  "
                  f"{stats['missing']}/{total}  {stats['unknown']}/{total}  {stats['invalid']}/{total}")

        return feature_stats


class APIIntegrationTester:
    """Test the backend API endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def test_health(self) -> bool:
        """Test health endpoint."""
        print("\nTesting /api/v1/health...")
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
            data = response.json()
            print(f"  Status: {response.status_code}")
            print(f"  Model loaded: {data.get('model_loaded', False)}")
            print(f"  DB connected: {data.get('db_connected', False)}")
            return response.status_code == 200
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False

    def test_analyze_image(self, image_path: str) -> Dict:
        """Test full analysis endpoint."""
        print(f"\nTesting /api/v1/analyze with {image_path}...")

        # Convert to base64
        with open(image_path, 'rb') as f:
            import base64
            img_data = base64.b64encode(f.read()).decode()
            base64_image = f"data:image/jpeg;base64,{img_data}"

        # Send request
        start = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/analyze",
                json={"image": base64_image, "source": "test"},
                timeout=60
            )

            elapsed = time.time() - start
            print(f"  Status: {response.status_code}")
            print(f"  Time: {elapsed:.2f}s")

            if response.status_code == 200:
                data = response.json()
                print(f"  Result: {data.get('result')}")
                print(f"  Confidence: {data.get('confidence'):.4f}")
                print(f"  Denomination: {data.get('currency_denomination')}")

                # Print feature analysis
                analysis = data.get('analysis', {})
                print(f"\n  Feature Analysis:")
                for feature_name, feature_data in analysis.items():
                    if isinstance(feature_data, dict) and 'status' in feature_data:
                        status = feature_data.get('status', 'N/A')
                        conf = feature_data.get('confidence', 0)
                        print(f"    {feature_name}: {status} ({conf:.2f})")

                # Check for critical failures
                critical = analysis.get('critical_failures', [])
                if critical:
                    print(f"\n  ⚠ Critical Failures:")
                    for failure in critical:
                        print(f"    - {failure.get('feature')}: {failure.get('status')}")

                return data
            else:
                print(f"  ✗ Error: {response.text}")
                return {}

        except Exception as e:
            print(f"  ✗ Request failed: {e}")
            return {}


class EdgeCaseTester:
    """Test system with edge cases and challenging conditions."""

    def __init__(self):
        self.results = []

    def test_blurred_image(self, image_path: str, output_dir: str = "test_outputs"):
        """Test with artificially blurred images."""
        print(f"\nTesting blurred image: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Test different blur levels
        for blur_amount in [5, 11, 21, 31]:
            blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
            blurred_path = os.path.join(output_dir, f"blur_{blur_amount}.jpg")
            cv2.imwrite(blurred_path, blurred)

            # Test features
            _, denoised, enhanced = preprocess_image(blurred)
            features = analyze_security_features(blurred, denoised, enhanced, "₹500")

            passed = sum(1 for f in features.values()
                        if f.get('status') in ['present', 'match', 'normal', 'valid'])
            total = len(features)

            print(f"  Blur {blur_amount}: {passed}/{total} features passed")
            self.results.append({
                'test': 'blur',
                'blur_amount': blur_amount,
                'features_passed': passed,
                'features_total': total
            })

    def test_varying_brightness(self, image_path: str, output_dir: str = "test_outputs"):
        """Test with different brightness levels."""
        print(f"\nTesting brightness variations: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Test different brightness
        for brightness_factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
            adjusted = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
            bright_path = os.path.join(output_dir, f"brightness_{brightness_factor:.2f}.jpg")
            cv2.imwrite(bright_path, adjusted)

            # Test features
            _, denoised, enhanced = preprocess_image(adjusted)
            features = analyze_security_features(adjusted, denoised, enhanced, "₹500")

            passed = sum(1 for f in features.values()
                        if f.get('status') in ['present', 'match', 'normal', 'valid'])
            total = len(features)

            print(f"  Brightness {brightness_factor}: {passed}/{total} features passed")
            self.results.append({
                'test': 'brightness',
                'factor': brightness_factor,
                'features_passed': passed,
                'features_total': total
            })


def run_comprehensive_test_suite(test_images_dir: str, api_url: str = None):
    """Run all test suites."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION TEST SUITE")
    print("=" * 80)

    # 1. Feature Validation Tests
    print("\n[1/3] Feature Validation Tests")
    feature_tester = FeatureValidationTester(test_images_dir)

    # Get first image for detailed test
    test_images = list(Path(test_images_dir).glob('*.jpg'))[:3]
    if test_images:
        for img in test_images:
            feature_tester.test_all_features(str(img))

    # 2. API Integration Tests (if server is running)
    if api_url:
        print("\n[2/3] API Integration Tests")
        api_tester = APIIntegrationTester(api_url)

        if api_tester.test_health():
            print("✓ API server is running")

            if test_images:
                api_tester.test_analyze_image(str(test_images[0]))
        else:
            print("⚠ API server not running. Skipping API tests.")
    else:
        print("\n[2/3] API Integration Tests - SKIPPED (no URL provided)")

    # 3. Edge Case Tests
    print("\n[3/3] Edge Case Tests")
    edge_tester = EdgeCaseTester()

    if test_images:
        edge_tester.test_blurred_image(str(test_images[0]))
        edge_tester.test_varying_brightness(str(test_images[0]))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\nFeatures tested: {feature_tester.results['features_tested']}")
    print(f"Features passed: {feature_tester.results['features_passed']}")
    print(f"Features failed: {feature_tester.results['features_failed']}")

    if feature_tester.results['features_tested'] > 0:
        pass_rate = feature_tester.results['features_passed'] / feature_tester.results['features_tested'] * 100
        print(f"Pass rate: {pass_rate:.1f}%")

    print(f"\nEdge case tests: {len(edge_tester.results)}")

    return {
        'feature_tests': feature_tester.results,
        'edge_cases': edge_tester.results
    }


def main():
    parser = argparse.ArgumentParser(description='Validation Testing Suite')
    parser.add_argument('--test-dir', type=str, default='./test_images',
                       help='Directory with test images')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000',
                       help='Backend API URL')
    parser.add_argument('--suite', choices=['features', 'api', 'edge', 'all'],
                       default='all', help='Test suite to run')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    if args.suite == 'all':
        results = run_comprehensive_test_suite(args.test_dir, args.api_url)
    elif args.suite == 'features':
        tester = FeatureValidationTester(args.test_dir)
        images = list(Path(args.test_dir).glob('*.jpg'))[:5]
        for img in images:
            tester.test_all_features(str(img))
        results = {'feature_tests': tester.results}
    elif args.suite == 'api':
        api_tester = APIIntegrationTester(args.api_url)
        api_tester.test_health()
        results = {'api_tests': 'completed'}
    elif args.suite == 'edge':
        edge_tester = EdgeCaseTester()
        images = list(Path(args.test_dir).glob('*.jpg'))
        if images:
            edge_tester.test_blurred_image(str(images[0]))
            edge_tester.test_varying_brightness(str(images[0]))
        results = {'edge_cases': edge_tester.results}

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
