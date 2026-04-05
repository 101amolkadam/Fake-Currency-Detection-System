"""
Quick Training Script for Fake Currency Detection
Simplified version for Windows compatibility
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

INPUT_SIZE = 299
BATCH_SIZE = 16
EPOCHS = 20

def load_data(data_dir):
    """Load dataset from directory structure."""
    print(f"\nLoading data from {data_dir}...")
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Load training data
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        seed=42,
        label_mode='binary'
    )
    
    # Load validation data
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir if os.path.exists(val_dir) else train_dir,
        image_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        seed=42,
        validation_split=0.2,
        subset='validation',
        label_mode='binary'
    )
    
    # Load test data if exists
    test_ds = None
    if os.path.exists(test_dir):
        test_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(INPUT_SIZE, INPUT_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=False,
            label_mode='binary'
        )
    
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")
    print(f"Training batches: {len(train_ds)}")
    print(f"Validation batches: {len(val_ds)}")
    
    return train_ds, val_ds, test_ds, class_names

def create_model():
    """Create Xception model for currency classification."""
    print("\nCreating Xception model...")
    
    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.2),
    ])
    
    # Load Xception base
    base_model = keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        pooling='avg'
    )
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = data_augmentation(inputs)
    x = keras.applications.xception.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    print(f"Model created: {model.count_params():,} parameters")
    return model, base_model

def train(model, base_model, train_ds, val_ds):
    """Train model in 2 phases."""
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        'models/xception_best.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Phase 1: Train head only
    print("\nPhase 1: Training classification head...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )
    
    # Phase 2: Fine-tune
    print("\nPhase 2: Fine-tuning base model...")
    base_model.trainable = True
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=10,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )
    
    # Merge histories
    for key in history2.history:
        history.history[key].extend(history2.history[key])
    
    return model, history

def evaluate(model, test_ds):
    """Evaluate model on test set."""
    if test_ds is None:
        print("\nNo test dataset available")
        return
    
    print("\n" + "="*80)
    print("EVALUATING MODEL")
    print("="*80)
    
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--data-dir', default='./training_data')
    args = parser.parse_args()
    
    global BATCH_SIZE, EPOCHS
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    print("\n" + "="*80)
    print("FAKE CURRENCY DETECTION - MODEL TRAINING")
    print("="*80)
    print(f"\nConfig:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    
    # Load data
    train_ds, val_ds, test_ds, class_names = load_data(args.data_dir)
    
    # Create model
    model, base_model = create_model()
    
    # Train
    model, history = train(model, base_model, train_ds, val_ds)
    
    # Evaluate
    evaluate(model, test_ds)
    
    # Save final model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    model.save('models/xception_currency_final.keras')
    model.save('models/xception_currency_final.h5')
    print("✓ Models saved to models/")
    
    # Save history
    import json
    history_data = {
        'epochs': len(history.history.get('accuracy', [])),
        'final_accuracy': history.history.get('accuracy', [0])[-1],
        'final_val_accuracy': history.history.get('val_accuracy', [0])[-1],
        'best_val_accuracy': max(history.history.get('val_accuracy', [0])),
    }
    
    with open('models/training_history.json', 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"  Final accuracy: {history_data['final_accuracy']*100:.2f}%")
    print(f"  Best val accuracy: {history_data['best_val_accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
