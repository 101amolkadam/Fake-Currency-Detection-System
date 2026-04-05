"""
Training Monitor - Real-time Training Dashboard
Monitors model training progress, system resources, and file updates.

Usage:
  python monitor_training.py              # Continuous monitoring
  python monitor_training.py --once       # Single snapshot
  python monitor_training.py --interval 5 # Check every 5 seconds
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Windows-specific imports
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Install with: pip install psutil")


class TrainingMonitor:
    """Monitor model training progress in real-time."""

    def __init__(self, models_dir: str = "models", data_dir: str = "training_data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.checkpoints = []
        self.start_time = datetime.now()

    def get_training_process(self) -> Optional[Dict]:
        """Find running Python training processes."""
        if not HAS_PSUTIL:
            return None

        training_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'train' in cmdline.lower():
                        training_processes.append({
                            'pid': proc.info['pid'],
                            'cpu_percent': proc.cpu_percent(),
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0,
                            'cmdline': cmdline,
                            'status': proc.status(),
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return training_processes[0] if training_processes else None

    def get_model_files(self) -> Dict:
        """Get information about model files."""
        model_files = {}

        if not self.models_dir.exists():
            return model_files

        for model_file in self.models_dir.glob('*'):
            if model_file.suffix in ['.keras', '.h5', '.json', '.png']:
                stat = model_file.stat()
                model_files[model_file.name] = {
                    'size_mb': stat.st_size / 1024 / 1024,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'age_minutes': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 60
                }

        return model_files

    def get_training_history(self) -> Optional[Dict]:
        """Load training history if available."""
        history_file = self.models_dir / 'training_history.json'

        if not history_file.exists():
            return None

        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except:
            return None

    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'train_real': 0,
            'train_fake': 0,
            'val_real': 0,
            'val_fake': 0,
            'test_real': 0,
            'test_fake': 0,
            'total_images': 0,
        }

        splits = ['train', 'val', 'test']
        categories = ['real', 'fake']

        for split in splits:
            for category in categories:
                split_dir = self.data_dir / split / category
                if split_dir.exists():
                    count = len(list(split_dir.glob('*.jpg'))) + \
                           len(list(split_dir.glob('*.png'))) + \
                           len(list(split_dir.glob('*.jpeg')))
                    stats[f'{split}_{category}'] = count
                    stats['total_images'] += count

        return stats

    def get_system_resources(self) -> Dict:
        """Get system resource usage."""
        if not HAS_PSUTIL:
            return {'error': 'psutil not installed'}

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.models_dir))

        return {
            'cpu_percent': cpu_percent,
            'memory_total_gb': memory.total / 1024 / 1024 / 1024,
            'memory_used_gb': memory.used / 1024 / 1024 / 1024,
            'memory_percent': memory.percent,
            'disk_total_gb': disk.total / 1024 / 1024 / 1024,
            'disk_used_gb': disk.used / 1024 / 1024 / 1024,
            'disk_percent': disk.percent,
        }

    def get_training_phase(self, history: Optional[Dict]) -> str:
        """Determine current training phase."""
        if not history:
            return "Not started"

        epochs = history.get('epochs', 0)

        if epochs == 0:
            return "Initializing..."
        elif epochs <= 10:
            return f"Phase 1: Training Head (Epoch {epochs}/10)"
        elif epochs <= 15:
            return f"Phase 2: Fine-Tuning (Epoch {epochs}/15)"
        else:
            return f"Complete ({epochs} epochs)"

    def display_dashboard(self):
        """Display full training dashboard."""
        # Clear screen (Windows)
        os.system('cls')

        print("\n" + "="*80)
        print("  FAKE CURRENCY DETECTION - TRAINING MONITOR")
        print("="*80)
        print(f"  Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Monitor Running: {(datetime.now() - self.start_time).total_seconds()/60:.1f} minutes")
        print("="*80)

        # 1. Training Process Status
        print("\n[1/5] TRAINING PROCESS")
        print("-" * 80)
        process = self.get_training_process()

        if process:
            print(f"  Status:       ✓ RUNNING")
            print(f"  PID:          {process['pid']}")
            print(f"  CPU Usage:    {process['cpu_percent']:.1f}%")
            print(f"  Memory:       {process['memory_mb']:.0f} MB")
            print(f"  State:        {process['status']}")
        else:
            print(f"  Status:       ✗ NOT RUNNING")
            print(f"  Note:         Training may have completed or not started")

        # 2. Dataset Statistics
        print("\n[2/5] DATASET")
        print("-" * 80)
        dataset = self.get_dataset_stats()

        print(f"  Training:")
        print(f"    Real:  {dataset['train_real']:>4} images")
        print(f"    Fake:  {dataset['train_fake']:>4} images")
        print(f"    Total: {dataset['train_real'] + dataset['train_fake']:>4} images")
        print(f"\n  Validation:")
        print(f"    Real:  {dataset['val_real']:>4} images")
        print(f"    Fake:  {dataset['val_fake']:>4} images")
        print(f"    Total: {dataset['val_real'] + dataset['val_fake']:>4} images")
        print(f"\n  Testing:")
        print(f"    Real:  {dataset['test_real']:>4} images")
        print(f"    Fake:  {dataset['test_fake']:>4} images")
        print(f"    Total: {dataset['test_real'] + dataset['test_fake']:>4} images")
        print(f"\n  Grand Total: {dataset['total_images']} images")

        balance = dataset['train_real'] / max(dataset['train_fake'], 1)
        print(f"  Balance Ratio: {balance:.1f}:1 (real:fake)")
        if balance > 5:
            print(f"  ⚠ Warning: Severely imbalanced! Consider downloading more fake samples.")

        # 3. Model Files
        print("\n[3/5] MODEL FILES")
        print("-" * 80)
        models = self.get_model_files()

        if not models:
            print("  No model files found yet")
        else:
            for name, info in sorted(models.items(), key=lambda x: x[1]['modified'], reverse=True):
                age = info['age_minutes']
                if age < 1:
                    status = "🔄 UPDATING NOW"
                elif age < 5:
                    status = "✓ Just saved"
                elif age < 30:
                    status = "✓ Recent"
                else:
                    status = "○ Old"

                print(f"  {status}  {name:<45} {info['size_mb']:>8.1f} MB  ({age:.1f}m ago)")

        # 4. Training History
        print("\n[4/5] TRAINING PROGRESS")
        print("-" * 80)
        history = self.get_training_history()

        if history:
            phase = self.get_training_phase(history)
            print(f"  Phase:          {phase}")
            print(f"  Epochs:         {history.get('epochs', 'N/A')}")

            if 'final_accuracy' in history:
                print(f"  Accuracy:       {history['final_accuracy']*100:.2f}%")
            if 'final_val_accuracy' in history:
                print(f"  Val Accuracy:   {history['final_val_accuracy']*100:.2f}%")
            if 'best_val_accuracy' in history:
                print(f"  Best Val Acc:   {history['best_val_accuracy']*100:.2f}%")

            # Estimate progress
            epochs = history.get('epochs', 0)
            progress = min(100, (epochs / 15) * 100)
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\n  Progress: [{bar}] {progress:.1f}%")
        else:
            print("  Training history not yet available")
            print("  Waiting for first checkpoint...")

        # 5. System Resources
        print("\n[5/5] SYSTEM RESOURCES")
        print("-" * 80)
        resources = self.get_system_resources()

        if 'error' not in resources:
            print(f"  CPU Usage:      {resources['cpu_percent']:.1f}%")
            print(f"  Memory:         {resources['memory_used_gb']:.1f}/{resources['memory_total_gb']:.1f} GB ({resources['memory_percent']}%)")
            print(f"  Disk:           {resources['disk_used_gb']:.1f}/{resources['disk_total_gb']:.1f} GB ({resources['disk_percent']}%)")

            # Warnings
            if resources['memory_percent'] > 90:
                print(f"  ⚠ Warning: High memory usage!")
            if resources['disk_percent'] > 90:
                print(f"  ⚠ Warning: Low disk space!")

        print("\n" + "="*80)

        return {
            'process': process is not None,
            'models': len(models),
            'history': history is not None,
            'dataset_total': dataset['total_images'],
        }

    def monitor_continuous(self, interval: int = 10):
        """Monitor training continuously."""
        print(f"\nMonitoring training every {interval} seconds...")
        print("Press CTRL+C to stop monitoring\n")

        try:
            iteration = 0
            while True:
                iteration += 1
                stats = self.display_dashboard()

                # Check if training completed
                if not stats['process'] and stats['history']:
                    print("\n✓ Training appears to have completed!")
                    print("Press CTRL+C to exit monitor")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user")
            print(f"Monitored for: {(datetime.now() - self.start_time).total_seconds()/60:.1f} minutes")
            print(f"Checks performed: {iteration}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Training Monitor Dashboard')
    parser.add_argument('--models-dir', default='models', help='Models directory')
    parser.add_argument('--data-dir', default='training_data', help='Training data directory')
    parser.add_argument('--interval', type=int, default=10, help='Check interval in seconds')
    parser.add_argument('--once', action='store_true', help='Single check instead of continuous')

    args = parser.parse_args()

    monitor = TrainingMonitor(args.models_dir, args.data_dir)

    if args.once:
        monitor.display_dashboard()
    else:
        monitor.monitor_continuous(args.interval)


if __name__ == "__main__":
    main()
