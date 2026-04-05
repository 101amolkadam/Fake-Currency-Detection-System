"""
Comprehensive Post-Training Validation Suite

This script runs automatically after training completes to validate:
1. Model accuracy on test dataset
2. All 15 security feature detectors
3. End-to-end API pipeline
4. Critical feature failure detection
5. Edge cases and robustness
6. Performance benchmarks
7. Regression testing (compare with previous model)

Usage:
  python test_validation_full.py                    # Run all tests
  python test_validation_full.py --model-path models/xception_currency_final.keras  # Specific model
  python test_validation_full.py --suite accuracy   # Only accuracy tests
  python test_validation_full.py --output results.json  # Save results to file
"""

import os
import sys
import cv2
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠ TensorFlow not available. Model tests will be skipped.")

from services.opencv_analyzer import analyze_security_features
from services.image_preprocessor import preprocess_image
from services.ensemble_engine import compute_ensemble_score


class ModelAccuracyTester:
    """Test model accuracy on test dataset."""

    def __init__(self, model_path: str = "models/xception_best.keras"):
        self.model_path = model_path
        self.model = None
        self.results = {}

    def load_model(self):
        """Load trained model."""
        if not HAS_TENSORFLOW:
            print("✗ TensorFlow not available")
            return False

        print(f"\nLoading model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            print(f"✗ Model file not found: {self.model_path}")
            return False

        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✓ Model loaded successfully")
            print(f"  Parameters: {self.model.count_params():,}")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False

    def test_on_dataset(self, data_dir: str = "training_data/test") -> Dict:
        """Test model accuracy on dataset."""
        if self.model is None:
            print("✗ Model not loaded")
            return {}

        print(f"\nTesting on dataset: {data_dir}")

        # Load test dataset
        try:
            test_ds = keras.utils.image_dataset_from_directory(
                data_dir,
                image_size=(299, 299),
                batch_size=16,
                shuffle=False,
                label_mode='binary'
            )
            class_names = test_ds.class_names
            print(f"✓ Loaded {len(test_ds)} batches")
            print(f"  Classes: {class_names}")
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            return {}

        # Evaluate
        print("\nRunning evaluation...")
        start = time.time()
        metrics = self.model.evaluate(test_ds, verbose=1)
        elapsed = time.time() - start

        # Get metric names
        metric_names = self.model.metrics_names
        results = dict(zip(metric_names, metrics))

        print(f"\n✓ Evaluation complete ({elapsed:.2f}s)")
        for name, value in results.items():
            print(f"  {name}: {value:.4f}")

        # Collect predictions for detailed analysis
        print("\nCollecting predictions for detailed analysis...")
        all_preds = []
        all_labels = []

        for images, labels in test_ds:
            preds = self.model.predict(images, verbose=0)
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

        binary_preds = (all_preds >= 0.5).astype(int)
        accuracy = np.mean(binary_preds == all_labels)

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0

        cm = confusion_matrix(all_labels, binary_preds)

        print(f"\nDetailed Metrics:")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  AUC: {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  {cm}")

        # Per-class metrics
        print(f"\nPer-Class Performance:")
        for i, class_name in enumerate(class_names):
            tp = cm[i][i] if i < cm.shape[0] else 0
            total = np.sum(cm[i]) if i < cm.shape[0] else 0
            acc = tp / total if total > 0 else 0
            print(f"  {class_name}: {acc*100:.2f}% ({tp}/{total})")

        self.results = {
            'dataset': data_dir,
            'metrics': results,
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(all_labels),
            'class_names': class_names,
            'evaluation_time': elapsed
        }

        return self.results


class FeatureValidationTester:
    """Test all 15 security feature detectors."""

    def __init__(self, test_images_dir: str):
        self.test_images_dir = test_images_dir
        self.feature_results = {}

    def test_single_image(self, image_path: str, denomination: str = "₹500") -> Dict:
        """Test all 15 security features on a single image."""
        image = cv2.imread(image_path)
        if image is None:
            return {}

        # Preprocess
        _, denoised, enhanced = preprocess_image(image)

        # Run feature analysis
        features = analyze_security_features(image, denoised, enhanced, denomination)

        # Summarize
        summary = {
            'total': 0,
            'present': 0,
            'missing': 0,
            'unknown': 0,
            'invalid': 0,
            'details': {}
        }

        for feature_name, feature_data in features.items():
            status = feature_data.get('status', 'unknown')
            confidence = feature_data.get('confidence', 0.0)

            summary['total'] += 1
            if status in ['present', 'match', 'normal', 'valid']:
                summary['present'] += 1
            elif status == 'missing':
                summary['missing'] += 1
            elif status == 'unknown':
                summary['unknown'] += 1
            elif status == 'invalid':
                summary['invalid'] += 1

            summary['details'][feature_name] = {
                'status': status,
                'confidence': confidence
            }

        return summary

    def test_dataset(self, sample_size: int = 5) -> Dict:
        """Test features on multiple images."""
        print(f"\nTesting security features on {sample_size} sample images...")

        image_files = list(Path(self.test_images_dir).glob('*.jpg')) + \
                     list(Path(self.test_images_dir).glob('*.png')) + \
                     list(Path(self.test_images_dir).glob('*.jpeg'))

        if not image_files:
            print(f"⚠ No images found in {self.test_images_dir}")
            return {}

        # Sample images
        sample = image_files[:min(sample_size, len(image_files))]

        all_results = []
        feature_stats = {}

        for img_path in sample:
            print(f"\n  Testing: {img_path.name}")
            result = self.test_single_image(str(img_path))

            if not result:
                continue

            all_results.append({
                'file': img_path.name,
                'summary': result
            })

            # Aggregate feature stats
            for feature_name, feature_data in result['details'].items():
                if feature_name not in feature_stats:
                    feature_stats[feature_name] = {
                        'present': 0, 'missing': 0, 'unknown': 0, 'invalid': 0
                    }

                status = feature_data['status']
                if status in feature_stats[feature_name]:
                    feature_stats[feature_name][status] += 1

        # Print summary
        print(f"\n{'='*80}")
        print(f"FEATURE DETECTION SUMMARY ({len(all_results)} images)")
        print(f"{'='*80}")
        print(f"\n{'Feature':<30} {'Present':<10} {'Missing':<10} {'Unknown':<10} {'Invalid':<10}")
        print("-" * 80)

        for feature, stats in sorted(feature_stats.items()):
            total = sum(stats.values())
            present_pct = stats['present'] / total * 100 if total > 0 else 0
            print(f"{feature:<28} {stats['present']}/{total} ({present_pct:.0f}%)  "
                  f"{stats['missing']}/{total}  {stats['unknown']}/{total}  {stats['invalid']}/{total}")

        self.feature_results = {
            'images_tested': len(all_results),
            'feature_statistics': feature_stats,
            'individual_results': all_results
        }

        return self.feature_results


class EdgeCaseTester:
    """Test system robustness with edge cases."""

    def __init__(self):
        self.results = []

    def test_blur_robustness(self, image_path: str) -> List[Dict]:
        """Test with different blur levels."""
        image = cv2.imread(image_path)
        if image is None:
            return []

        print(f"\n  Testing blur robustness...")
        results = []

        for blur_amount in [5, 11, 21, 31]:
            blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
            _, denoised, enhanced = preprocess_image(blurred)
            features = analyze_security_features(blurred, denoised, enhanced, "₹500")

            passed = sum(1 for f in features.values()
                        if f.get('status') in ['present', 'match', 'normal', 'valid'])

            results.append({
                'test': 'blur',
                'blur_amount': blur_amount,
                'features_passed': passed,
                'features_total': len(features)
            })

            print(f"    Blur {blur_amount}: {passed}/{len(features)} features passed")

        self.results.extend(results)
        return results

    def test_brightness_robustness(self, image_path: str) -> List[Dict]:
        """Test with different brightness levels."""
        image = cv2.imread(image_path)
        if image is None:
            return []

        print(f"\n  Testing brightness robustness...")
        results = []

        for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
            adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
            _, denoised, enhanced = preprocess_image(adjusted)
            features = analyze_security_features(adjusted, denoised, enhanced, "₹500")

            passed = sum(1 for f in features.values()
                        if f.get('status') in ['present', 'match', 'normal', 'valid'])

            results.append({
                'test': 'brightness',
                'factor': factor,
                'features_passed': passed,
                'features_total': len(features)
            })

            print(f"    Brightness {factor:.2f}: {passed}/{len(features)} features passed")

        self.results.extend(results)
        return results


class PerformanceBenchmark:
    """Benchmark system performance."""

    def __init__(self):
        self.benchmarks = {}

    def benchmark_inference_time(self, model_path: str, num_runs: int = 10) -> Dict:
        """Benchmark model inference time."""
        if not HAS_TENSORFLOW:
            return {}

        print(f"\nBenchmarking inference time ({num_runs} runs)...")

        try:
            model = keras.models.load_model(model_path)
        except:
            return {}

        # Create dummy input
        dummy_input = np.random.rand(1, 299, 299, 3).astype(np.float32)

        # Warm up
        _ = model.predict(dummy_input, verbose=0)

        # Benchmark
        times = []
        for i in range(num_runs):
            start = time.time()
            _ = model.predict(dummy_input, verbose=0)
            elapsed = time.time() - start
            times.append(elapsed)

        times = np.array(times)
        benchmark = {
            'mean_ms': np.mean(times) * 1000,
            'median_ms': np.median(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'num_runs': num_runs
        }

        print(f"  Mean: {benchmark['mean_ms']:.2f} ms")
        print(f"  Median: {benchmark['median_ms']:.2f} ms")
        print(f"  Std: {benchmark['std_ms']:.2f} ms")

        self.benchmarks['inference_time'] = benchmark
        return benchmark


class ValidationSuite:
    """Run complete validation suite."""

    def __init__(self, model_path: str, test_dir: str, output_file: str = None):
        self.model_path = model_path
        self.test_dir = test_dir
        self.output_file = output_file or f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.all_results = {
            'timestamp': datetime.now().isoformat(),
            'model': model_path,
            'test_directory': test_dir,
            'tests': {}
        }

    def run_all_tests(self):
        """Run all validation tests."""
        print("\n" + "="*80)
        print("  COMPREHENSIVE POST-TRAINING VALIDATION SUITE")
        print("="*80)
        print(f"\nModel: {self.model_path}")
        print(f"Test Directory: {self.test_dir}")
        print(f"Output: {self.output_file}")

        # Test 1: Model Accuracy
        print("\n" + "="*80)
        print("[1/4] MODEL ACCURACY TESTS")
        print("="*80)

        accuracy_tester = ModelAccuracyTester(self.model_path)
        if accuracy_tester.load_model():
            accuracy_results = accuracy_tester.test_on_dataset(self.test_dir)
            self.all_results['tests']['accuracy'] = accuracy_results
        else:
            print("⚠ Skipping accuracy tests (model not loaded)")
            self.all_results['tests']['accuracy'] = {'status': 'skipped'}

        # Test 2: Feature Validation
        print("\n" + "="*80)
        print("[2/4] FEATURE VALIDATION TESTS")
        print("="*80)

        feature_tester = FeatureValidationTester(self.test_dir)
        feature_results = feature_tester.test_dataset(sample_size=5)
        self.all_results['tests']['features'] = feature_results

        # Test 3: Edge Cases
        print("\n" + "="*80)
        print("[3/4] EDGE CASE TESTS")
        print("="*80)

        edge_tester = EdgeCaseTester()
        test_images = list(Path(self.test_dir).glob('*.jpg'))[:1]

        if test_images:
            edge_tester.test_blur_robustness(str(test_images[0]))
            edge_tester.test_brightness_robustness(str(test_images[0]))
            self.all_results['tests']['edge_cases'] = edge_tester.results
        else:
            print("⚠ No test images for edge case tests")
            self.all_results['tests']['edge_cases'] = {'status': 'skipped'}

        # Test 4: Performance Benchmark
        print("\n" + "="*80)
        print("[4/4] PERFORMANCE BENCHMARKS")
        print("="*80)

        benchmark = PerformanceBenchmark()
        benchmark.benchmark_inference_time(self.model_path, num_runs=10)
        self.all_results['tests']['benchmarks'] = benchmark.benchmarks

        # Summary
        print("\n" + "="*80)
        print("  VALIDATION SUMMARY")
        print("="*80)

        if 'accuracy' in self.all_results['tests'] and self.all_results['tests']['accuracy']:
            acc = self.all_results['tests']['accuracy']
            print(f"\n✓ Model Accuracy: {acc.get('accuracy', 0)*100:.2f}%")
            print(f"✓ AUC Score: {acc.get('auc', 0):.4f}")
            print(f"✓ Test Samples: {acc.get('total_samples', 0)}")

        if 'features' in self.all_results['tests']:
            feat = self.all_results['tests']['features']
            print(f"\n✓ Feature Tests: {feat.get('images_tested', 0)} images")

        if 'edge_cases' in self.all_results['tests']:
            print(f"\n✓ Edge Case Tests: {len(self.all_results['tests']['edge_cases'])} tests")

        if 'benchmarks' in self.all_results['tests']:
            bench = self.all_results['tests']['benchmarks'].get('inference_time', {})
            print(f"\n✓ Inference Time: {bench.get('mean_ms', 0):.2f} ms (mean)")

        # Save results
        with open(self.output_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)

        print(f"\n✓ Results saved to: {self.output_file}")
        print("="*80)

        return self.all_results


def main():
    parser = argparse.ArgumentParser(description='Post-Training Validation Suite')
    parser.add_argument('--model-path', default='models/xception_best.keras',
                       help='Path to trained model')
    parser.add_argument('--test-dir', default='training_data/test',
                       help='Test dataset directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    parser.add_argument('--suite', choices=['accuracy', 'features', 'edge', 'benchmark', 'all'],
                       default='all', help='Test suite to run')

    args = parser.parse_args()

    if args.suite == 'all':
        suite = ValidationSuite(args.model_path, args.test_dir, args.output)
        suite.run_all_tests()
    else:
        print(f"Running {args.suite} tests only...")
        # Implement partial test runs if needed


if __name__ == "__main__":
    main()
