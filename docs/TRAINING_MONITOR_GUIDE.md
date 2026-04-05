# Training Monitoring Dashboard - User Guide

## Overview

The Training Monitoring Dashboard provides real-time visibility into your model training process. It tracks:
- Training process status (running/completed/failed)
- Dataset statistics and class balance
- Model file updates and checkpoints
- System resource usage (CPU, memory, disk)
- Training progress and performance metrics

---

## Quick Start

### Installation

```bash
cd backend
pip install psutil
```

### Basic Usage

#### Option 1: Single Snapshot
Get a one-time view of the current training status:

```bash
python monitor_training.py --once
```

**Output**: Displays complete dashboard with all 5 sections.

#### Option 2: Continuous Monitoring
Monitor training in real-time with automatic updates:

```bash
# Check every 10 seconds (default)
python monitor_training.py

# Check every 5 seconds
python monitor_training.py --interval 5

# Check every 30 seconds
python monitor_training.py --interval 30
```

**Output**: Refreshes dashboard at specified interval. Press `CTRL+C` to stop.

---

## Dashboard Sections

### [1/5] TRAINING PROCESS
Shows the status of the training Python process:

```
[1/5] TRAINING PROCESS
--------------------------------------------------------------------------------
  Status:       ✓ RUNNING
  PID:          26196
  CPU Usage:    85.3%
  Memory:       331 MB
  State:        running
```

**Key Metrics:**
- **Status**: ✓ RUNNING (training active) or ✗ NOT RUNNING (completed/not started)
- **PID**: Process ID (use `taskkill /F /PID <pid>` to stop)
- **CPU Usage**: Percentage of CPU being used
- **Memory**: RAM usage in MB
- **State**: running, sleeping, zombie, etc.

**Troubleshooting:**
- If Status shows ✗ NOT RUNNING but training should be active, check if it crashed
- If Memory is very high (>500 MB), consider reducing batch size

---

### [2/5] DATASET
Shows dataset statistics and class balance:

```
[2/5] DATASET
--------------------------------------------------------------------------------
  Training:
    Real:    95 images
    Fake:     8 images
    Total:  103 images

  Validation:
    Real:    20 images
    Fake:     1 images
    Total:   21 images

  Testing:
    Real:    21 images
    Fake:     3 images
    Total:   24 images

  Grand Total: 148 images
  Balance Ratio: 11.9:1 (real:fake)
  ⚠ Warning: Severely imbalanced! Consider downloading more fake samples.
```

**Key Metrics:**
- **Balance Ratio**: Ideal is 1:1 (equal real/fake). >5:1 is problematic.
- **Warning**: Shows if dataset is severely imbalanced

**Recommendations:**
- If imbalanced, download more fake samples from Kaggle
- See `backend/DATASET_DOWNLOAD.md` for dataset download instructions
- After downloading, run: `python collect_datasets.py prepare`

---

### [3/5] MODEL FILES
Shows all model files and when they were last updated:

```
[3/5] MODEL FILES
--------------------------------------------------------------------------------
  🔄 UPDATING NOW  xception_best.keras                     94.1 MB  (0.3m ago)
  ✓ Recent         xception_currency_final.keras           92.9 MB  (11.6m ago)
  ○ Old            xception_currency_final.h5              92.8 MB  (23.9h ago)
```

**Status Indicators:**
- 🔄 **UPDATING NOW**: File modified in last 1 minute
- ✓ **Just saved**: File modified in last 5 minutes
- ✓ **Recent**: File modified in last 30 minutes
- ○ **Old**: File not updated recently

**File Descriptions:**
- `xception_best.keras`: Best model checkpoint (updated during training)
- `xception_currency_final.keras`: Final model (saved at end of training)
- `xception_currency_final.h5`: Legacy format model (for compatibility)

**Monitoring Tip**: If the best model file keeps updating (age < 5 min), training is progressing well.

---

### [4/5] TRAINING PROGRESS
Shows training metrics and progress bar:

```
[4/5] TRAINING PROGRESS
--------------------------------------------------------------------------------
  Phase:          Phase 1: Training Head (Epoch 5/10)
  Epochs:         5
  Accuracy:       94.23%
  Val Accuracy:   90.48%
  Best Val Acc:   91.67%

  Progress: [████████████████████░░░░░░░░░░░░░░░░░░░░] 33.3%
```

**Key Metrics:**
- **Phase**: Current training phase (Phase 1: Head Training, Phase 2: Fine-Tuning, Complete)
- **Epochs**: Number of epochs completed
- **Accuracy**: Training accuracy
- **Val Accuracy**: Validation accuracy (most important metric)
- **Best Val Acc**: Best validation accuracy achieved so far
- **Progress Bar**: Visual progress toward 15 epochs

**Interpreting Results:**
- If Val Accuracy < Training Accuracy by >5%, model may be overfitting
- If both are low (<70%), model may need more training data or epochs
- If Val Accuracy is >90%, model is performing well

---

### [5/5] SYSTEM RESOURCES
Shows system resource usage:

```
[5/5] SYSTEM RESOURCES
--------------------------------------------------------------------------------
  CPU Usage:      72.3%
  Memory:         7.9/7.9 GB (99.9%)
  Disk:           328.5/495.7 GB (66.3%)
  ⚠ Warning: High memory usage!
```

**Key Metrics:**
- **CPU Usage**: Overall system CPU usage
- **Memory**: RAM usage (total/available)
- **Disk**: Disk space usage

**Warnings:**
- **High memory usage** (>90%): Training may slow down or crash. Consider:
  - Reducing batch size: `python train_quick.py --batch-size 8`
  - Closing other applications
- **Low disk space** (>90%): May cause training to fail. Free up disk space.

---

## Common Use Cases

### Use Case 1: Quick Status Check

```bash
python monitor_training.py --once
```

**When to use**: Before starting other work, to verify training is running.

**What to look for**:
- Training Process: ✓ RUNNING
- Model Files: Recent checkpoint (< 5 min old)
- No critical warnings

---

### Use Case 2: Monitor Throughout Training

```bash
python monitor_training.py --interval 30
```

**When to use**: While training is running, to monitor progress.

**What to look for**:
- Model checkpoint age decreasing (means new checkpoint saved)
- Progress bar advancing
- No resource warnings

---

### Use Case 3: Check if Training Completed

```bash
python monitor_training.py --once
```

**When to use**: After expected training time has passed.

**What to look for**:
- Training Process: ✗ NOT RUNNING (training process exited)
- Training Progress: Phase: Complete
- Model Files: Recent final model (< 5 min old)

**Next steps after completion**:
1. Restart backend server: `uvicorn main:app --reload`
2. Test with validation: `python test_validation.py --test-dir ../test_images`
3. Test in web interface: Open http://localhost:5173

---

### Use Case 4: Debug Training Issues

```bash
# Check training status
python monitor_training.py --once

# Check if process is running
tasklist | findstr python

# Check model files
cd models
dir /o-d
```

**Common Issues:**

**Issue 1: Training not running**
- Status shows ✗ NOT RUNNING
- Solution: Start training: `python train_quick.py --epochs 15`

**Issue 2: No checkpoints**
- Model Files section empty
- Solution: Wait longer (first checkpoint takes 5-10 min) or check for errors

**Issue 3: High memory**
- Warning: High memory usage!
- Solution: Reduce batch size, restart training with smaller batch

**Issue 4: Imbalanced dataset**
- Warning: Severely imbalanced!
- Solution: Download more fake samples, see `DATASET_DOWNLOAD.md`

---

## Advanced Usage

### Custom Directories

If your models or data are in different locations:

```bash
python monitor_training.py --models-dir /path/to/models --data-dir /path/to/data
```

### Continuous Monitoring with Logging

Monitor and save logs to file:

```bash
python monitor_training.py --interval 60 > training_monitor.log 2>&1
```

### Check Specific Aspect Only

For quick checks, use command-line tools instead of full dashboard:

```bash
# Check if training is running
tasklist | findstr python

# Check latest model
cd models
dir /o-d *.keras | head -5

# Check training history
cat models/training_history.json
```

---

## Integration with Training

### Start Training + Monitoring

```bash
# Terminal 1: Start training
cd backend
python train_quick.py --epochs 15 --batch-size 16

# Terminal 2: Monitor training
cd backend
python monitor_training.py --interval 30
```

### Automated Monitoring Script

Create a batch file `start_training_with_monitor.bat`:

```batch
@echo off
echo Starting training...
start cmd /k "cd backend && python train_quick.py --epochs 15"

echo Waiting 30 seconds for first checkpoint...
timeout /t 30 /nobreak

echo Starting monitor...
cd backend
python monitor_training.py --interval 30
```

---

## Troubleshooting

### "psutil not installed" Error

**Solution**:
```bash
pip install psutil
```

### Dashboard Shows Incorrect Data

**Cause**: Data directories not matching.

**Solution**: Specify correct directories:
```bash
python monitor_training.py --models-dir ./models --data-dir ./training_data
```

### Monitor Crashes or Freezes

**Cause**: Python process issues on Windows.

**Solution**: 
- Run with `--once` flag for single snapshot
- Reduce monitoring frequency (increase `--interval`)
- Restart Python interpreter

### Model Files Not Updating

**Possible causes**:
1. Training crashed
2. Training not started
3. First epoch still in progress (takes 5-10 min)

**Solution**:
1. Check Training Process section for RUNNING status
2. Check for error messages in training terminal
3. Wait for first checkpoint (10-15 min)

---

## Performance Tips

### Reduce Memory Usage

If you see "High memory usage" warning:

```bash
# Reduce batch size
python train_quick.py --batch-size 8

# Or even smaller
python train_quick.py --batch-size 4
```

### Speed Up Training

```bash
# Use GPU if available (requires TensorFlow GPU setup)
# Training will be 5-10x faster

# Reduce epochs for quick test
python train_quick.py --epochs 5
```

### Manage Disk Space

Model files are large (~95 MB each). Clean up old models:

```bash
cd models
# Keep only latest 3 files
dir /o-d *.keras
# Delete old files manually
```

---

## Next Steps After Training

Once training completes (monitor shows "Complete" phase):

1. **Verify model saved**:
   ```bash
   python monitor_training.py --once
   # Check: Model Files section shows recent final model
   ```

2. **Restart backend**:
   ```bash
   # Kill existing backend
   taskkill /F /IM python.exe
   
   # Start new backend
   cd backend
   uvicorn main:app --reload
   ```

3. **Test new model**:
   ```bash
   python test_validation.py --test-dir ../test_images --suite all
   ```

4. **Commit to Git**:
   ```bash
   git add -A
   git commit -m "feat: retrain model with improved accuracy"
   git push origin main
   ```

---

## Summary

| Command | Description | Use Case |
|---------|-------------|----------|
| `python monitor_training.py` | Continuous monitoring (10s interval) | Watch training progress |
| `python monitor_training.py --once` | Single snapshot | Quick status check |
| `python monitor_training.py --interval 5` | Fast monitoring (5s) | Detailed tracking |
| `python monitor_training.py --interval 60` | Slow monitoring (60s) | Long-running training |

**Dashboard Shows**:
1. ✓ Training Process (running/completed)
2. 📊 Dataset Statistics (balance, size)
3. 💾 Model Files (checkpoints, age)
4. 📈 Training Progress (accuracy, phase)
5. 🔧 System Resources (CPU, memory, disk)

**Key Files**:
- `monitor_training.py`: Monitoring script
- `train_quick.py`: Training script
- `models/xception_best.keras`: Best checkpoint
- `models/training_history.json`: Training metrics

---

**Version**: 1.0  
**Last Updated**: 5 April 2026  
**Status**: Production-Ready
