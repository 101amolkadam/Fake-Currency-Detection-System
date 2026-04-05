"""
Advanced training script for Fake Currency Detection System.
Improves model accuracy with:
- Better data augmentation pipeline
- Progressive fine-tuning
- Class balancing
- Learning rate scheduling
- Early stopping with model checkpointing
- Comprehensive evaluation metrics
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class DataAugmentation:
    """Advanced data augmentation pipeline for currency images."""
    
    def __init__(self, augmentation_factor=20):
        self.augmentation_factor = augmentation_factor
        
    def augment_image(self, image, label):
        """Apply random augmentations to a single image."""
        aug_image = tf.identity(image)
        
        # Random rotation (-30 to +30 degrees)
        angle = np.random.uniform(-30, 30)
        aug_image = self._rotate_image(aug_image, angle)
        
        # Random zoom (0.8x to 1.3x)
        zoom_factor = np.random.uniform(0.8, 1.3)
        aug_image = self._zoom_image(aug_image, zoom_factor)
        
        # Random brightness adjustment
        aug_image = tf.image.random_brightness(aug_image, max_delta=0.3)
        
        # Random contrast adjustment
        aug_image = tf.image.random_contrast(aug_image, 0.7, 1.3)
        
        # Random saturation adjustment
        aug_image = tf.image.random_saturation(aug_image, 0.8, 1.2)
        
        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            aug_image = tf.image.flip_left_right(aug_image)
        
        # Random vertical flip (30% chance - currency can be upside down)
        if np.random.random() > 0.7:
            aug_image = tf.image.flip_up_down(aug_image)
        
        # Add Gaussian noise (30% chance)
        if np.random.random() > 0.7:
            noise = tf.random.normal(
                shape=tf.shape(aug_image), 
                mean=0.0, 
                stddev=np.random.uniform(0.01, 0.05)
            )
            aug_image = tf.clip_by_value(aug_image + noise, 0.0, 1.0)
        
        # Random blur (20% chance)
        if np.random.random() > 0.8:
            kernel_size = np.random.choice([3, 5, 7])
            aug_image = self._blur_image(aug_image, kernel_size)
        
        return aug_image
    
    def _rotate_image(self, image, angle):
        """Rotate image by given angle."""
        shape = tf.cast(tf.shape(image)[:2], tf.float32)
        center = shape / 2.0
        
        angle_rad = angle * np.pi / 180.0
        cos_val = tf.cos(angle_rad)
        sin_val = tf.sin(angle_rad)
        
        x = tf.linspace(-1.0, 1.0, tf.shape(image)[1])
        y = tf.linspace(-1.0, 1.0, tf.shape(image)[0])
        x, y = tf.meshgrid(x, y)
        
        x_new = x * cos_val - y * sin_val
        y_new = x * sin_val + y * cos_val
        
        # Scale to image coordinates
        x_coords = (x_new + 1.0) * shape[1] / 2.0
        y_coords = (y_new + 1.0) * shape[0] / 2.0
        
        coords = tf.stack([y_coords, x_coords], axis=-1)
        rotated = tf.gather_nd(image, tf.cast(coords, tf.int32), batch_dims=0)
        
        return rotated
    
    def _zoom_image(self, image, zoom_factor):
        """Zoom into image by given factor."""
        shape = tf.cast(tf.shape(image)[:2], tf.float32)
        center = shape / 2.0
        
        x = tf.linspace(-1.0, 1.0, tf.shape(image)[1])
        y = tf.linspace(-1.0, 1.0, tf.shape(image)[0])
        x, y = tf.meshgrid(x, y)
        
        x_scaled = x / zoom_factor
        y_scaled = y / zoom_factor
        
        x_coords = (x_scaled + 1.0) * shape[1] / 2.0
        y_coords = (y_scaled + 1.0) * shape[0] / 2.0
        
        x_coords = tf.clip_by_value(x_coords, 0, shape[1] - 1)
        y_coords = tf.clip_by_value(y_coords, 0, shape[0] - 1)
        
        coords = tf.stack([y_coords, x_coords], axis=-1)
        zoomed = tf.gather_nd(image, tf.cast(coords, tf.int32), batch_dims=0)
        
        return zoomed
    
    def _blur_image(self, image, kernel_size):
        """Apply Gaussian blur to image."""
        # Simple blur using average pooling
        image_exp = tf.expand_dims(image, axis=0)
        blurred = tf.nn.avg_pool2d(
            image_exp, 
            ksize=[1, kernel_size, kernel_size, 1],
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        return tf.squeeze(blurred, axis=0)


def load_dataset_from_directory(data_dir):
    """
    Load currency dataset from directory structure.
    Expected structure:
        data_dir/
            real/
                500_real_1.jpg
                2000_real_1.jpg
                ...
            fake/
                500_fake_1.jpg
                2000_fake_1.jpg
                ...
    """
    data_path = Path(data_dir)
    images = []
    labels = []
    filenames = []
    
    # Load real images
    real_dir = data_path / "real"
    if real_dir.exists():
        for img_path in real_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (299, 299))
                    images.append(img)
                    labels.append(1)  # 1 for REAL
                    filenames.append(img_path.name)
    
    # Load fake images
    fake_dir = data_path / "fake"
    if fake_dir.exists():
        for img_path in fake_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (299, 299))
                    images.append(img)
                    labels.append(0)  # 0 for FAKE
                    filenames.append(img_path.name)
    
    if len(images) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics:")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    print(f"Real images: {np.sum(labels == 1)}")
    print(f"Fake images: {np.sum(labels == 0)}")
    print(f"Class imbalance ratio: {np.sum(labels == 1) / max(np.sum(labels == 0), 1):.2f}:1")
    
    return images, labels, filenames


def create_model(num_classes=2, dropout_rate=0.5, trainable_base=False):
    """
    Create improved Xception model with custom head.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Base dropout rate
        trainable_base: Whether to make base layers trainable
    
    Returns:
        Compiled Keras model
    """
    # Load Xception base with ImageNet weights
    base_model = keras.applications.Xception(
        weights='imagenet',
        input_shape=(299, 299, 3),
        include_top=False,
        pooling='avg'
    )
    
    # Freeze/unfreeze base layers
    base_model.trainable = trainable_base
    print(f"\n{'='*60}")
    print(f"Base model: Xception (ImageNet pretrained)")
    print(f"Base model trainable: {trainable_base}")
    print(f"Total layers: {len(base_model.layers)}")
    
    # Build custom classification head
    inputs = keras.Input(shape=(299, 299, 3))
    
    # Preprocess input using Xception's preprocessing
    x = keras.applications.xception.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=trainable_base)
    
    # Custom head with progressive dropout
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.8)(x)
    
    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.6)(x)
    
    x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.4)(x)
    
    # Output layer
    if num_classes == 2:
        # Binary classification (REAL vs FAKE)
        outputs = layers.Dense(1, activation='sigmoid', name='authenticity')(x)
    else:
        # Multi-class classification
        outputs = layers.Dense(num_classes, activation='softmax', name='authenticity')(x)
    
    model = models.Model(inputs, outputs)
    
    # Print parameter summary
    total_params = model.count_params()
    trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"{'='*60}\n")
    
    return model


def compile_model(model, learning_rate=1e-3, phase=1):
    """Compile model with appropriate optimizer and metrics."""
    
    if phase == 1:
        # Phase 1: Training classification head only
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    else:
        # Phase 2: Fine-tuning with lower learning rate
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    print(f"\nModel compiled - Phase {phase}")
    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Loss: binary_crossentropy")
    
    return model


def train_phase1(X_train, y_train, X_val, y_val, epochs=50, batch_size=8):
    """
    Phase 1: Train classification head with frozen base.
    """
    print("\n" + "="*60)
    print("PHASE 1: Training Classification Head (Frozen Base)")
    print("="*60)
    
    # Create model
    model = create_model(num_classes=2, dropout_rate=0.5, trainable_base=False)
    model = compile_model(model, learning_rate=1e-3, phase=1)
    
    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        'backend/models/xception_best.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Data augmentation
    augmenter = DataAugmentation()
    
    def augment_train_data():
        """Generator that yields augmented images."""
        while True:
            indices = np.random.choice(len(X_train), batch_size)
            for idx in indices:
                img = X_train[idx] / 255.0
                label = y_train[idx]
                aug_img = augmenter.augment_image(img, label)
                # Apply Xception preprocessing
                aug_img = keras.applications.xception.preprocess_input(aug_img)
                yield aug_img, label
    
    # Validation data preprocessing
    X_val_processed = keras.applications.xception.preprocess_input(X_val.copy())
    
    # Calculate steps
    steps_per_epoch = len(X_train) * 15 // batch_size  # 15x augmentation
    
    print(f"\nTraining for up to {epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Batch size: {batch_size}")
    
    # Train
    history = model.fit(
        augment_train_data(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val_processed, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Save model
    model.save('backend/models/xception_currency_final.keras')
    print("\n✓ Phase 1 training complete!")
    print(f"Best model saved to: backend/models/xception_currency_final.keras")
    
    return model, history


def train_phase2(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=4):
    """
    Phase 2: Progressive fine-tuning of base model layers.
    """
    print("\n" + "="*60)
    print("PHASE 2: Progressive Fine-Tuning")
    print("="*60)
    
    # Unfreeze more layers progressively
    # Unfreeze last 30% of base model layers
    base_model = model.layers[2]  # Xception base is at index 2
    base_model.trainable = True
    
    # Keep BatchNormalization layers frozen
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    
    # Fine-tune from layer 100 onwards (approximately last 30%)
    fine_tune_at = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"Fine-tuning from layer {fine_tune_at} onwards")
    print(f"Trainable layers: {sum(1 for l in model.layers if l.trainable)}")
    
    # Recompile with lower learning rate
    model = compile_model(model, learning_rate=1e-5, phase=2)
    
    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        'backend/models/xception_finetuned.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    # Class weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Data augmentation
    augmenter = DataAugmentation()
    
    def augment_train_data():
        while True:
            indices = np.random.choice(len(X_train), batch_size)
            for idx in indices:
                img = X_train[idx] / 255.0
                label = y_train[idx]
                aug_img = augmenter.augment_image(img, label)
                aug_img = keras.applications.xception.preprocess_input(aug_img)
                yield aug_img, label
    
    X_val_processed = keras.applications.xception.preprocess_input(X_val.copy())
    steps_per_epoch = len(X_train) * 15 // batch_size
    
    print(f"\nFine-tuning for up to {epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Train
    history = model.fit(
        augment_train_data(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val_processed, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Save fine-tuned model
    model.save('backend/models/xception_currency_finetuned.keras')
    print("\n✓ Phase 2 training complete!")
    print(f"Fine-tuned model saved to: backend/models/xception_currency_finetuned.keras")
    
    return model, history


def evaluate_model(model, X_test, y_test, test_filenames=None):
    """Comprehensive model evaluation."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Preprocess test data
    X_test_processed = keras.applications.xception.preprocess_input(X_test.copy())
    
    # Predictions
    predictions = model.predict(X_test_processed, verbose=1)
    predictions = np.squeeze(predictions)
    
    # Convert to binary predictions
    y_pred = (predictions >= 0.5).astype(int)
    
    # Metrics
    accuracy = np.mean(y_pred == y_test)
    auc = roc_auc_score(y_test, predictions)
    
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"AUC: {auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
    
    # Per-class confidence
    for class_idx, class_name in enumerate(['FAKE', 'REAL']):
        mask = y_test == class_idx
        if np.sum(mask) > 0:
            class_preds = predictions[mask]
            print(f"\n{class_name} samples ({np.sum(mask)} total):")
            print(f"  Mean confidence: {np.mean(class_preds)*100:.2f}%")
            print(f"  Min confidence: {np.min(class_preds)*100:.2f}%")
            print(f"  Max confidence: {np.max(class_preds)*100:.2f}%")
            print(f"  Std deviation: {np.std(class_preds)*100:.2f}%")
    
    # Per-image analysis if filenames provided
    if test_filenames:
        print(f"\n{'='*60}")
        print("PER-IMAGE ANALYSIS:")
        print(f"{'='*60}")
        print(f"{'Filename':<30} {'Actual':<8} {'Predicted':<10} {'Confidence':<12} {'Correct'}")
        print("-"*80)
        
        for i, filename in enumerate(test_filenames):
            actual = "REAL" if y_test[i] == 1 else "FAKE"
            predicted = "REAL" if y_pred[i] == 1 else "FAKE"
            confidence = predictions[i] if y_pred[i] == 1 else 1 - predictions[i]
            correct = "✓" if y_pred[i] == y_test[i] else "✗"
            
            print(f"{filename:<30} {actual:<8} {predicted:<10} {confidence*100:.1f}%{'':<7} {correct}")
    
    return accuracy, auc, predictions


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUC
    axes[0, 1].plot(history.history['auc'], label='Training AUC')
    if 'val_auc' in history.history:
        axes[0, 1].plot(history.history['val_auc'], label='Validation AUC')
    axes[0, 1].set_title('Model AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss
    axes[1, 0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[1, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[1, 0].set_title('Model Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision & Recall
    if 'precision' in history.history:
        axes[1, 1].plot(history.history['precision'], label='Precision')
        axes[1, 1].plot(history.history['recall'], label='Recall')
        if 'val_precision' in history.history:
            axes[1, 1].plot(history.history['val_precision'], label='Val Precision')
            axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to: {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train Fake Currency Detection Model')
    parser.add_argument('--data-dir', type=str, default='backend/training_data',
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per phase')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--phase', type=int, choices=[1, 2, 'both'], default='both',
                        help='Which phase to run')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only evaluate existing model')
    parser.add_argument('--test-dir', type=str, default='test_images',
                        help='Path to test images directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FAKE CURRENCY DETECTION - ADVANCED TRAINING")
    print("="*60)
    
    # Load dataset
    try:
        print(f"\nLoading dataset from: {args.data_dir}")
        images, labels, filenames = load_dataset_from_directory(args.data_dir)
        
        # Split dataset
        X_train_val, X_test, y_train_val, y_test, filenames_train_val, filenames_test = \
            train_test_split(images, labels, filenames, test_size=0.15, random_state=42, stratify=labels)
        
        X_train, X_val, y_train, y_val, filenames_train, filenames_val = \
            train_test_split(X_train_val, y_train_val, filenames_train_val, 
                           test_size=0.15, random_state=42, stratify=y_train_val)
        
        print(f"\nDataset splits:")
        print(f"Training: {len(X_train)} images")
        print(f"Validation: {len(X_val)} images")
        print(f"Test: {len(X_test)} images")
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("Please ensure training data exists in the specified directory.")
        return
    
    if args.evaluate_only:
        # Load existing model and evaluate
        print("\nLoading existing model for evaluation...")
        try:
            model = keras.models.load_model('backend/models/xception_currency_final.keras')
            evaluate_model(model, X_test, y_test, filenames_test)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
        return
    
    # Training
    if args.phase in [1, 'both']:
        model, history_phase1 = train_phase1(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        plot_training_history(history_phase1, save_path='training_history_phase1.png')
    
    if args.phase in [2, 'both']:
        if args.phase == 2:
            # Load best model from phase 1
            model = keras.models.load_model('backend/models/xception_best.keras')
        
        model, history_phase2 = train_phase2(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=max(4, args.batch_size // 2)
        )
        plot_training_history(history_phase2, save_path='training_history_phase2.png')
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    try:
        best_model = keras.models.load_model('backend/models/xception_best.keras')
    except:
        best_model = keras.models.load_model('backend/models/xception_currency_final.keras')
    
    evaluate_model(best_model, X_test, y_test, filenames_test)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModel files saved to:")
    print("  - backend/models/xception_currency_final.keras (final model)")
    print("  - backend/models/xception_best.keras (best validation AUC)")
    print("  - backend/models/xception_finetuned.keras (fine-tuned, if phase 2)")


if __name__ == '__main__':
    main()
