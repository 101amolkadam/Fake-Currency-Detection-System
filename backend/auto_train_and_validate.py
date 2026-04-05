"""
Auto Training Runner - Monitors training and runs validation when complete

This script:
1. Monitors training process (train_quick.py)
2. Detects when training completes
3. Automatically runs the full validation suite
4. Generates a comprehensive report
5. Optionally restarts the backend server

Usage:
  python auto_train_and_validate.py              # Monitor and auto-validate
  python auto_train_and_validate.py --train       # Start training first, then monitor
  python auto_train_and_validate.py --validate-only  # Just run validation now
  python auto_train_and_validate.py --restart-backend  # Restart backend after validation
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import argparse

# Windows process management
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠ psutil not installed. Install with: pip install psutil")
    HAS_PSUTIL = False


class TrainingMonitor:
    """Monitor training process for completion."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.last_checkpoint_time = None
        self.checkpoints = []

    def get_latest_checkpoint_age(self) -> Optional[float]:
        """Get age of latest model checkpoint in seconds."""
        best_model = self.models_dir / "xception_best.keras"

        if not best_model.exists():
            return None

        age = time.time() - best_model.stat().st_mtime
        return age

    def is_training_running(self) -> bool:
        """Check if training process is still running."""
        if not HAS_PSUTIL:
            # Fallback: check with tasklist
            try:
                result = subprocess.run(
                    ['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                    capture_output=True, text=True
                )
                # Look for train in command line would need wmic, simplified here
                return 'python.exe' in result.stdout
            except:
                return False

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'python' in proc.info['name'].lower() and 'train' in cmdline.lower():
                    return True
            except:
                pass

        return False

    def wait_for_completion(self, check_interval: int = 30, stale_threshold: int = 300) -> bool:
        """
        Wait for training to complete.

        Args:
            check_interval: Seconds between checks
            stale_threshold: Seconds without checkpoint update = training stalled

        Returns:
            True if training completed, False if timeout/stalled
        """
        print("\n" + "="*80)
        print("  MONITORING TRAINING PROGRESS")
        print("="*80)
        print(f"\nChecking every {check_interval} seconds")
        print(f"Stale threshold: {stale_threshold} seconds")

        iteration = 0
        last_checkpoint_age = self.get_latest_checkpoint_age()

        while True:
            iteration += 1
            current_age = self.get_latest_checkpoint_age()

            # Get timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')

            if self.is_training_running():
                status = "🔄 RUNNING"
                if current_age is not None:
                    age_min = current_age / 60
                    print(f"[{timestamp}] {status} | Last checkpoint: {age_min:.1f}m ago")
                else:
                    print(f"[{timestamp}] {status} | No checkpoints yet")
            else:
                # Training process not found
                if current_age is not None and current_age < stale_threshold:
                    print(f"\n{'='*80}")
                    print("  ✓ TRAINING COMPLETED!")
                    print(f"{'='*80}")
                    print(f"  Last checkpoint: {current_age/60:.1f} minutes ago")
                    print(f"  Model file: {self.models_dir / 'xception_best.keras'}")
                    return True
                elif current_age is not None:
                    print(f"\n{'='*80}")
                    print("  ⚠ TRAINING MAY HAVE CRASHED")
                    print(f"{'='*80}")
                    print(f"  Last checkpoint: {current_age/60:.1f} minutes ago (stale)")
                    return False
                else:
                    print(f"[{timestamp}] ✗ NOT RUNNING | No checkpoints")

            # Check for stale training
            if current_age is not None and current_age > stale_threshold:
                print(f"\n{'='*80}")
                print("  ⚠ TRAINING STALLED")
                print(f"{'='*80}")
                print(f"  No checkpoint updates for {current_age/60:.1f} minutes")
                return False

            time.sleep(check_interval)


class ValidationRunner:
    """Run validation suite and generate reports."""

    def __init__(self, model_path: str, test_dir: str):
        self.model_path = model_path
        self.test_dir = test_dir
        self.results_file = None

    def run_validation(self) -> Dict:
        """Run the full validation suite."""
        print("\n" + "="*80)
        print("  RUNNING COMPREHENSIVE VALIDATION SUITE")
        print("="*80)

        # Import and run validation
        sys.path.insert(0, str(Path(__file__).parent))

        try:
            from test_validation_full import ValidationSuite

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"validation_results_{timestamp}.json"

            suite = ValidationSuite(
                model_path=self.model_path,
                test_dir=self.test_dir,
                output_file=output_file
            )

            results = suite.run_all_tests()
            self.results_file = output_file

            return results

        except Exception as e:
            print(f"\n✗ Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def generate_summary_report(self, results: Dict) -> str:
        """Generate human-readable summary report."""
        report = []
        report.append("\n" + "="*80)
        report.append("  VALIDATION SUMMARY REPORT")
        report.append("="*80)
        report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  Model: {self.model_path}")
        report.append("="*80)

        # Accuracy
        if 'accuracy' in results.get('tests', {}):
            acc_data = results['tests']['accuracy']
            if acc_data.get('status') != 'skipped':
                report.append(f"\n[MODEL ACCURACY]")
                report.append(f"  Test Accuracy: {acc_data.get('accuracy', 0)*100:.2f}%")
                report.append(f"  AUC Score: {acc_data.get('auc', 0):.4f}")
                report.append(f"  Test Samples: {acc_data.get('total_samples', 0)}")

                # Confusion matrix
                cm = acc_data.get('confusion_matrix', [])
                if cm:
                    report.append(f"\n  Confusion Matrix:")
                    for row in cm:
                        report.append(f"    {row}")

        # Features
        if 'features' in results.get('tests', {}):
            feat_data = results['tests']['features']
            report.append(f"\n[FEATURE DETECTION]")
            report.append(f"  Images Tested: {feat_data.get('images_tested', 0)}")

            # Feature statistics
            stats = feat_data.get('feature_statistics', {})
            if stats:
                report.append(f"\n  {'Feature':<30} {'Present':<10} {'Missing':<10}")
                report.append("  " + "-"*50)
                for feature, feature_stats in sorted(stats.items()):
                    total = sum(feature_stats.values())
                    present = feature_stats.get('present', 0)
                    report.append(f"  {feature:<28} {present}/{total:<10}")

        # Edge cases
        if 'edge_cases' in results.get('tests', {}):
            edge_data = results['tests']['edge_cases']
            if isinstance(edge_data, list):
                report.append(f"\n[EDGE CASE TESTS]")
                report.append(f"  Tests Run: {len(edge_data)}")

                # Group by test type
                by_type = {}
                for test in edge_data:
                    test_type = test.get('test', 'unknown')
                    if test_type not in by_type:
                        by_type[test_type] = []
                    by_type[test_type].append(test)

                for test_type, tests in by_type.items():
                    report.append(f"\n  {test_type.upper()}:")
                    for test in tests:
                        passed = test.get('features_passed', 0)
                        total = test.get('features_total', 0)
                        report.append(f"    {passed}/{total} features passed")

        # Benchmarks
        if 'benchmarks' in results.get('tests', {}):
            bench_data = results['tests']['benchmarks']
            if 'inference_time' in bench_data:
                bench = bench_data['inference_time']
                report.append(f"\n[PERFORMANCE BENCHMARKS]")
                report.append(f"  Inference Time:")
                report.append(f"    Mean: {bench.get('mean_ms', 0):.2f} ms")
                report.append(f"    Median: {bench.get('median_ms', 0):.2f} ms")
                report.append(f"    Min: {bench.get('min_ms', 0):.2f} ms")
                report.append(f"    Max: {bench.get('max_ms', 0):.2f} ms")

        report.append("\n" + "="*80)

        report_text = '\n'.join(report)
        print(report_text)

        # Save report
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)

        print(f"\n✓ Report saved to: {report_file}")
        return report_file


class BackendRestarter:
    """Restart backend server to load new model."""

    @staticmethod
    def kill_existing_backend():
        """Kill any running backend servers."""
        print("\nKilling existing backend processes...")

        try:
            if HAS_PSUTIL:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if 'uvicorn' in cmdline.lower() and 'main:app' in cmdline:
                            print(f"  Killing PID: {proc.info['pid']}")
                            proc.kill()
                    except:
                        pass
            else:
                # Use taskkill
                subprocess.run(
                    ['taskkill', '/F', '/IM', 'python.exe'],
                    capture_output=True
                )
                print("  Killed Python processes")

            print("✓ Done")
        except Exception as e:
            print(f"✗ Error: {e}")

    @staticmethod
    def start_backend():
        """Start backend server."""
        print("\nStarting backend server...")
        print("  Command: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        print("  Note: Server will run in background")

        # Don't actually start it in background from this script
        # Just print instructions
        print("\n✓ To start backend manually:")
        print("  cd backend")
        print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")


def run_auto_training_and_validate(
    start_training: bool = False,
    check_interval: int = 30,
    stale_threshold: int = 300,
    restart_backend: bool = False
):
    """
    Complete workflow:
    1. Optionally start training
    2. Monitor for completion
    3. Run validation suite
    4. Generate report
    5. Optionally restart backend
    """
    print("\n" + "="*80)
    print("  AUTO TRAINING & VALIDATION WORKFLOW")
    print("="*80)
    print(f"  Start Training: {start_training}")
    print(f"  Check Interval: {check_interval}s")
    print(f"  Stale Threshold: {stale_threshold}s")
    print(f"  Restart Backend: {restart_backend}")
    print("="*80)

    # Step 1: Start training if requested
    if start_training:
        print("\n[1/5] STARTING TRAINING")
        print("-" * 80)

        # Start training in background
        train_script = Path(__file__).parent / "train_quick.py"
        if not train_script.exists():
            print("✗ train_quick.py not found!")
            return

        print("Starting: python train_quick.py --epochs 15 --batch-size 16")
        subprocess.Popen(
            [sys.executable, str(train_script), "--epochs", "15", "--batch-size", "16"],
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )

        print("✓ Training started in new window")
        print("  Waiting 60 seconds for first checkpoint...")
        time.sleep(60)

    # Step 2: Monitor training
    print("\n[2/5] MONITORING TRAINING")
    print("-" * 80)

    monitor = TrainingMonitor()
    training_completed = monitor.wait_for_completion(
        check_interval=check_interval,
        stale_threshold=stale_threshold
    )

    if not training_completed:
        print("\n✗ Training did not complete successfully")
        return

    # Step 3: Run validation
    print("\n[3/5] RUNNING VALIDATION SUITE")
    print("-" * 80)

    model_path = "models/xception_best.keras"
    test_dir = "training_data/test"

    validator = ValidationRunner(model_path, test_dir)
    results = validator.run_validation()

    if not results:
        print("\n✗ Validation failed")
        return

    # Step 4: Generate report
    print("\n[4/5] GENERATING REPORT")
    print("-" * 80)

    report_file = validator.generate_summary_report(results)

    # Step 5: Restart backend if requested
    if restart_backend:
        print("\n[5/5] RESTARTING BACKEND")
        print("-" * 80)

        restarter = BackendRestarter()
        restarter.kill_existing_backend()
        # Don't auto-start, just print instructions
        restarter.start_backend()
    else:
        print("\n[5/5] BACKEND RESTART SKIPPED")
        print("-" * 80)
        print("  To restart backend manually:")
        print("    cd backend")
        print("    uvicorn main:app --reload --host 0.0.0.0 --port 8000")

    # Final summary
    print("\n" + "="*80)
    print("  WORKFLOW COMPLETE")
    print("="*80)
    print(f"\n✓ Training: {'Completed' if training_completed else 'Failed'}")
    print(f"✓ Validation: Results saved")
    print(f"✓ Report: {report_file}")
    print(f"✓ Results: {validator.results_file}")
    print("\nNext steps:")
    print("  1. Review validation report")
    print("  2. Start/restart backend server")
    print("  3. Test with web interface")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Auto Training & Validation Runner')
    parser.add_argument('--train', action='store_true', help='Start training first')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation')
    parser.add_argument('--restart-backend', action='store_true', help='Restart backend after validation')
    parser.add_argument('--check-interval', type=int, default=30, help='Monitor check interval (seconds)')
    parser.add_argument('--stale-threshold', type=int, default=300, help='Stale training threshold (seconds)')
    parser.add_argument('--model-path', default='models/xception_best.keras', help='Model path for validation')
    parser.add_argument('--test-dir', default='training_data/test', help='Test directory')

    args = parser.parse_args()

    if args.validate_only:
        # Just run validation now
        validator = ValidationRunner(args.model_path, args.test_dir)
        results = validator.run_validation()
        if results:
            validator.generate_summary_report(results)
    else:
        # Full workflow
        run_auto_training_and_validate(
            start_training=args.train,
            check_interval=args.check_interval,
            stale_threshold=args.stale_threshold,
            restart_backend=args.restart_backend
        )


if __name__ == "__main__":
    main()
