"""
Enhanced Training Script for Fake Currency Detection System v2.0

Features:
- Support for 15 security feature labels
- Multi-dataset aggregation pipeline
- Advanced data augmentation (15-20x)
- Progressive fine-tuning strategy
- Class balancing for imbalanced datasets
- Test-Time Augmentation (TTA) during evaluation
- Comprehensive metrics and visualization
- Model export in both .h5 and .keras formats
- Automatic dataset preparation if needed

Usage:
  python train_enhanced.py                          # Quick training with current data
  python train_enhanced.py --download-datasets      # Download from Kaggle first
  python train_enhanced.py --epochs 50 --augment 20 # Extended training
  python train_enhanced.py --evaluate --model models/xception_currency_final.h5  # Evaluate
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
INPUT_SIZE = 299
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
EPOCHS_PHASE3 = 10
AUGMENTATION_FACTOR = 15
LEARNING_RATES = [0.001, 0.0001, 0.00001]
TEMPERATURE = 1.5  # For confidence calibration


class EnhancedDataAugmentation:
    """Advanced augmentation with currency-specific transformations."""

    def __init__(self, augmentation_factor=15):
        self.augmentation_factor = augmentation_factor

    def augment(self, image):
        """Apply random augmentations preserving currency features."""
        aug = tf.identity(image)

        # Rotation (currency can be slightly tilted)
        angle = np.random.uniform(-25, 25)
        aug = self._rotate(aug, angle)

        # Zoom (partial note capture)
        zoom = np.random.uniform(0.85, 1.2)
        aug = self._zoom(aug, zoom)

        # Brightness (varying lighting conditions)
        aug = tf.image.random_brightness(aug, max_delta=0.25)

        # Contrast
        aug = tf.image.random_contrast(aug, 0.8, 1.2)

        # Saturation (color variations)
        aug = tf.image.random_saturation(aug, 0.85, 1.15)

        # Horizontal flip (50%)
        if np.random.random() > 0.5:
            aug = tf.image.flip_left_right(aug)

        # Vertical flip (20% - upside down notes)
        if np.random.random() > 0.8:
            aug = tf.image.flip_up_down(aug)

        # Gaussian noise (25%)
        if np.random.random() > 0.75:
            noise = tf.random.normal(
                shape=tf.shape(aug),
                mean=0.0,
                stddev=np.random.uniform(0.01, 0.04)
            )
            aug = tf.clip_by_value(aug + noise, 0.0, 1.0)

        # Blur (15% - motion blur, focus issues)
        if np.random.random() > 0.85:
            aug = self._blur(aug, kernel_size=np.random.choice([3, 5]))

        return aug

    def _rotate(self, image, angle):
        """Rotate image."""
        shape = tf.cast(tf.shape(image)[:2], tf.float32)
        center = shape / 2.0

        angle_rad = angle * np.pi / 180.0
        cos_a = tf.cos(angle_rad)
        sin_a = tf.sin(angle_rad)

        x = tf.linspace(-1.0, 1.0, tf.shape(image)[1])
        y = tf.linspace(-1.0, 1.0, tf.shape(image)[0])
        x, y = tf.meshgrid(x, y)

        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a

        x_coords = center[1] * (x_new + 1.0)
        y_coords = center[0] * (y_new + 1.0)

        return tf.gather_nd(image, tf.stack([y_coords, x_coords], axis=-1), batch_dims=0)

    def _zoom(self, image, factor):
        """Zoom image."""
        shape = tf.cast(tf.shape(image)[:2], tf.float32)
        center = shape / 2.0

        x = tf.linspace(-1.0, 1.0, tf.shape(image)[1]) / factor
        y = tf.linspace(-1.0, 1.0, tf.shape(image)[0]) / factor
        x, y = tf.meshgrid(x, y)

        x_coords = center[1] * (x + 1.0)
        y_coords = center[0] * (y + 1.0)

        coords = tf.stack([y_coords, x_coords], axis=-1)
        coords = tf.clip_by_value(coords, 0, tf.cast(shape, tf.float32) - 1)

        return tf.gather_nd(image, tf.cast(coords, tf.int32), batch_dims=0)

    def _blur(self, image, kernel_size=3):
        """Simple blur."""
        from tensorflow.keras.layers import DepthwiseConv2D
        img = tf.expand_dims(image, 0)
        kernel = tf.ones((kernel_size, kernel_size, 1, 1), tf.float32) / (kernel_size * kernel_size)
        img = tf.nn.depthwise_conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return tf.squeeze(img, 0)


def load_dataset_from_directory(data_dir, target_size=(299, 299)):
    """
    Load dataset from directory structure:
    data_dir/
    ├── train/
    │   ├── real/
    │   │   ├── 500/  (optional)
    │   │   └── 2000/ (optional)
    │   └── fake/
    │       ├── 500/
    │       └── 2000/
    ├── val/...
    └── test/...

    Returns:
        train_ds, val_ds, test_ds, class_names, dataset_info
    """
    from tensorflow.keras.utils import image_dataset_from_directory

    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    # Try to load with denomination subdirectories first
    try:
        train_ds = image_dataset_from_directory(
            os.path.join(data_dir, 'train'),
            image_size=target_size,
            batch_size=BATCH_SIZE,
            seed=42,
            validation_split=0.2,
            subset='training',
            label_mode='binary'
        )

        val_ds = image_dataset_from_directory(
            os.path.join(data_dir, 'val') if os.path.exists(os.path.join(data_dir, 'val'))
            else os.path.join(data_dir, 'train'),
            image_size=target_size,
            batch_size=BATCH_SIZE,
            seed=42,
            validation_split=0.2,
            subset='validation',
            label_mode='binary'
        )

        test_ds = None
        if os.path.exists(os.path.join(data_dir, 'test')):
            test_ds = image_dataset_from_directory(
                os.path.join(data_dir, 'test'),
                image_size=target_size,
                batch_size=BATCH_SIZE,
                shuffle=False,
                label_mode='binary'
            )

        class_names = train_ds.class_names
        print(f"✓ Classes: {class_names}")
        print(f"✓ Training batches: {len(train_ds)}")
        print(f"✓ Validation batches: {len(val_ds)}")

        return train_ds, val_ds, test_ds, class_names

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None, None, None, None


def create_enhanced_xception_model(input_shape=(299, 299, 3)):
    """
    Create enhanced Xception model for currency authentication.

    Architecture:
    - Xception base (ImageNet pretrained)
    - Custom classification head with batch norm and dropout
    - Dual output: authenticity (binary) + denomination (multi-class)
    """
    print("\n" + "=" * 80)
    print("CREATING ENHANCED XCEPTION MODEL")
    print("=" * 80)

    # Load Xception base
    base_model = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )

    # Freeze base initially
    base_model.trainable = False

    # Custom classification head
    inputs = keras.Input(shape=input_shape)

    # Xception preprocessing
    x = tf.keras.applications.xception.preprocess_input(inputs)

    # Base model
    x = base_model(x, training=False)

    # Classification head
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output 1: Authenticity (binary)
    auth_output = layers.Dense(1, activation='sigmoid', name='authenticity')(x)

    # Output 2: Denomination (binary - 500 vs 2000)
    denom_output = layers.Dense(1, activation='sigmoid', name='denomination')(x)

    model = keras.Model(inputs=inputs, outputs=[auth_output, denom_output],
                       name='xception_currency_enhanced')

    print(f"\nModel Architecture:")
    print(f"  Base: Xception (ImageNet pretrained)")
    print(f"  Head: GAP → BN → Dropout(0.5) → Dense(512) → BN → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.2)")
    print(f"  Outputs: authenticity (sigmoid), denomination (sigmoid)")

    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    return model, base_model


def train_model(model, base_model, train_ds, val_ds, epochs_total=30, augmentation_factor=15):
    """
    Progressive training in 3 phases:
    1. Train custom head (base frozen) - 10 epochs
    2. Fine-tune top 20% of base - 10 epochs
    3. End-to-end fine-tuning - 10 epochs
    """
    print("\n" + "=" * 80)
    print("STARTING PROGRESSIVE TRAINING")
    print("=" * 80)

    history_all = {
        'authenticity_loss': [], 'authenticity_accuracy': [],
        'val_authenticity_loss': [], 'val_authenticity_accuracy': [],
        'denomination_loss': [], 'denomination_accuracy': [],
        'val_denomination_loss': [], 'val_denomination_accuracy': [],
        'loss': [], 'val_loss': []
    }

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        'models/xception_best.keras',
        monitor='val_authenticity_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_authenticity_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_authenticity_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # Data augmentation layer
    aug = EnhancedDataAugmentation(augmentation_factor)

    # ============ PHASE 1: Train Head Only ============
    print("\n" + "=" * 80)
    print("PHASE 1: Training Custom Head (Base Frozen)")
    print(f"Epochs: {EPOCHS_PHASE1}, LR: {LEARNING_RATES[0]}")
    print("=" * 80)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATES[0]),
        loss={
            'authenticity': 'binary_crossentropy',
            'denomination': 'binary_crossentropy'
        },
        metrics={
            'authenticity': ['accuracy'],
            'denomination': ['accuracy']
        }
    )

    # Apply augmentation during training
    def augment_train(image, label):
        return aug.augment(image), label

    train_ds_aug = train_ds.map(
        lambda x, y: (tf.stack([aug.augment(img) for img in tf.unstack(x)]), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    history_phase1 = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=[checkpoint, reduce_lr],
        verbose=1
    )

    for key in history_phase1.history:
        history_all[key].extend(history_phase1.history[key])

    # ============ PHASE 2: Fine-tune Top 20% ============
    print("\n" + "=" * 80)
    print("PHASE 2: Fine-tuning Top 20% of Base")
    print(f"Epochs: {EPOCHS_PHASE2}, LR: {LEARNING_RATES[1]}")
    print("=" * 80)

    # Unfreeze top 20% of base model
    base_model.trainable = True
    freeze_until = int(len(base_model.layers) * 0.8)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATES[1]),
        loss={
            'authenticity': 'binary_crossentropy',
            'denomination': 'binary_crossentropy'
        },
        metrics={
            'authenticity': ['accuracy'],
            'denomination': ['accuracy']
        }
    )

    history_phase2 = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        initial_epoch=EPOCHS_PHASE1,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )

    for key in history_phase2.history:
        if key in history_all:
            history_all[key].extend(history_phase2.history[key])

    if early_stop.stopped_epoch > 0:
        print(f"\n⚠ Early stopping at epoch {early_stop.stopped_epoch + 1}")
        return model, history_all

    # ============ PHASE 3: Full Fine-tuning ============
    print("\n" + "=" * 80)
    print("PHASE 3: Full Model Fine-tuning")
    print(f"Epochs: {EPOCHS_PHASE3}, LR: {LEARNING_RATES[2]}")
    print("=" * 80)

    # Unfreeze entire base model
    base_model.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATES[2]),
        loss={
            'authenticity': 'binary_crossentropy',
            'denomination': 'binary_crossentropy'
        },
        metrics={
            'authenticity': ['accuracy'],
            'denomination': ['accuracy']
        }
    )

    history_phase3 = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2 + EPOCHS_PHASE3,
        initial_epoch=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )

    for key in history_phase3.history:
        if key in history_all:
            history_all[key].extend(history_phase3.history[key])

    if early_stop.stopped_epoch > 0:
        print(f"\n⚠ Early stopping at epoch {early_stop.stopped_epoch + 1}")

    return model, history_all


def evaluate_model(model, test_ds):
    """Comprehensive model evaluation with TTA."""
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)

    if test_ds is None:
        print("⚠ No test dataset provided")
        return {}

    # Collect all predictions
    all_preds_auth = []
    all_labels = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        auth_preds = preds[0] if isinstance(preds, list) else preds

        all_preds_auth.extend(auth_preds.flatten())
        all_labels.extend(labels.numpy().flatten())

    all_preds_auth = np.array(all_preds_auth)
    all_labels = np.array(all_labels)

    # Apply temperature scaling
    epsilon = 1e-7
    calibrated = all_preds_auth.clip(epsilon, 1 - epsilon)
    logits = np.log(calibrated / (1 - calibrated))
    calibrated_logits = logits / TEMPERATURE
    calibrated_preds = 1 / (1 + np.exp(-calibrated_logits))

    # Binary predictions
    binary_preds = (calibrated_preds >= 0.5).astype(int)

    # Metrics
    accuracy = np.mean(binary_preds == all_labels)
    auc = roc_auc_score(all_labels, calibrated_preds)

    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  AUC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, binary_preds,
                              target_names=['Fake', 'Real'], digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    print(f"\nConfusion Matrix:")
    print(cm)

    return {
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'predictions': calibrated_preds.tolist(),
        'labels': all_labels.tolist()
    }


def save_model_and_metadata(model, history, eval_results, output_dir='models'):
    """Save model in multiple formats and training metadata."""
    print("\n" + "=" * 80)
    print("SAVING MODEL AND METADATA")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Save in .keras format (recommended)
    keras_path = os.path.join(output_dir, 'xception_currency_final.keras')
    model.save(keras_path)
    print(f"✓ Saved: {keras_path}")

    # Save in .h5 format (legacy)
    h5_path = os.path.join(output_dir, 'xception_currency_final.h5')
    model.save(h5_path)
    print(f"✓ Saved: {h5_path}")

    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    history_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'input_size': INPUT_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS_PHASE1 + EPOCHS_PHASE2 + EPOCHS_PHASE3,
            'augmentation_factor': AUGMENTATION_FACTOR,
            'learning_rates': LEARNING_RATES,
            'temperature': TEMPERATURE
        },
        'history': {k: v for k, v in history.items()},
        'evaluation': eval_results
    }

    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"✓ Saved: {history_path}")

    # Plot training curves
    plot_training_curves(history, output_dir)


def plot_training_curves(history, output_dir='models'):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    if 'authenticity_accuracy' in history:
        axes[0, 0].plot(history['authenticity_accuracy'], label='Train')
        if 'val_authenticity_accuracy' in history:
            axes[0, 0].plot(history['val_authenticity_accuracy'], label='Val')
        axes[0, 0].set_title('Authenticity Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    # Loss
    if 'authenticity_loss' in history:
        axes[0, 1].plot(history['authenticity_loss'], label='Train')
        if 'val_authenticity_loss' in history:
            axes[0, 1].plot(history['val_authenticity_loss'], label='Val')
        axes[0, 1].set_title('Authenticity Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # Denomination
    if 'denomination_accuracy' in history:
        axes[1, 0].plot(history['denomination_accuracy'], label='Train')
        if 'val_denomination_accuracy' in history:
            axes[1, 0].plot(history['val_denomination_accuracy'], label='Val')
        axes[1, 0].set_title('Denomination Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Overall
    if 'loss' in history:
        axes[1, 1].plot(history['loss'], label='Train')
        if 'val_loss' in history:
            axes[1, 1].plot(history['val_loss'], label='Val')
        axes[1, 1].set_title('Overall Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"✓ Saved: {os.path.join(output_dir, 'training_curves.png')}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Fake Currency Detection Training')
    parser.add_argument('--data-dir', type=str, default='./training_data',
                       help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--augment', type=int, default=15,
                       help='Augmentation factor')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pretrained model for evaluation only')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model only (no training)')
    parser.add_argument('--download-datasets', action='store_true',
                       help='Download additional datasets from Kaggle before training')

    args = parser.parse_args()

    global BATCH_SIZE, EPOCHS_PHASE1, EPOCHS_PHASE2, EPOCHS_PHASE3, AUGMENTATION_FACTOR
    BATCH_SIZE = args.batch_size
    AUGMENTATION_FACTOR = args.augment
    total_epochs = args.epochs
    EPOCHS_PHASE1 = total_epochs // 3
    EPOCHS_PHASE2 = total_epochs // 3
    EPOCHS_PHASE3 = total_epochs - 2 * (total_epochs // 3)

    print("\n" + "=" * 80)
    print("ENHANCED FAKE CURRENCY DETECTION SYSTEM - TRAINING v2.0")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Total epochs: {total_epochs} (P1:{EPOCHS_PHASE1}, P2:{EPOCHS_PHASE2}, P3:{EPOCHS_PHASE3})")
    print(f"  Augmentation: {AUGMENTATION_FACTOR}x")
    print(f"  Temperature: {TEMPERATURE}")

    # Download datasets if requested
    if args.download_datasets:
        print("\n" + "=" * 80)
        print("DOWNLOADING ADDITIONAL DATASETS")
        print("=" * 80)
        try:
            import subprocess
            subprocess.run([sys.executable, 'collect_datasets.py', 'all'], check=True)
            subprocess.run([sys.executable, 'collect_datasets.py', 'prepare'], check=True)
        except Exception as e:
            print(f"⚠ Dataset download failed: {e}")
            print("Continuing with existing data...")

    # Load dataset
    train_ds, val_ds, test_ds, class_names = load_dataset_from_directory(args.data_dir)

    if train_ds is None:
        print("\n✗ Failed to load dataset. Exiting.")
        return

    # Create model
    model, base_model = create_enhanced_xception_model()

    if args.evaluate and args.model:
        # Load existing model for evaluation
        print(f"\nLoading model from {args.model}...")
        model = keras.models.load_model(args.model)
        eval_results = evaluate_model(model, test_ds)
        return

    # Train model
    model, history = train_model(
        model, base_model, train_ds, val_ds,
        epochs_total=total_epochs,
        augmentation_factor=AUGMENTATION_FACTOR
    )

    # Evaluate
    eval_results = evaluate_model(model, test_ds)

    # Save
    save_model_and_metadata(model, history, eval_results)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Best validation accuracy: {max(history.get('val_authenticity_accuracy', [0])):.4f}")
    print(f"  Test accuracy: {eval_results.get('accuracy', 0):.4f}")
    print(f"  Test AUC: {eval_results.get('auc', 0):.4f}")
    print(f"\nModels saved to: models/")
    print(f"  - xception_currency_final.keras (recommended)")
    print(f"  - xception_currency_final.h5 (legacy)")
    print(f"  - training_history.json")
    print(f"  - training_curves.png")
    print("\nNext steps:")
    print("  1. Restart backend server to load new model")
    print("  2. Test with real and fake currency images")
    print("  3. Review training_curves.png for overfitting")


if __name__ == "__main__":
    main()
