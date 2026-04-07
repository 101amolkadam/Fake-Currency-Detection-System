"""
PyTorch CNN Classifier for Fake Currency Detection
Uses Xception architecture with enhanced classification head
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path
import json
from datetime import datetime


class CurrencyDataset(Dataset):
    """Dataset for fake currency detection.
    
    Supports both flat structure (fake/*.jpg, real/*.jpg) and 
    nested by denomination (fake/500/*.jpg, real/500/*.jpg, etc.)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_map = {'fake': 0, 'real': 1}

        for class_name, class_idx in self.class_map.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            # Check if there are subdirectories (denomination folders)
            subdirs = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
            
            if subdirs:
                # Nested structure: fake/500/*.jpg, fake/2000/*.jpg, etc.
                for subdir in subdirs:
                    subdir_path = os.path.join(class_dir, subdir)
                    for img_name in os.listdir(subdir_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                            self.samples.append((
                                os.path.join(subdir_path, img_name),
                                class_idx
                            ))
            else:
                # Flat structure: fake/*.jpg, real/*.jpg
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            class_idx
                        ))

        print(f"✓ Loaded {len(self.samples)} images from {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


class EnhancedXception(nn.Module):
    """Enhanced MobileNetV3 model for fake currency detection.
    
    Uses MobileNetV3-Large (depthwise-separable convolutions) as backbone,
    which is lightweight and fits within GTX 1050 3GB VRAM.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained MobileNetV3-Large
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)

        # Remove original classifier, keep features from avgpool (960 dims)
        self.backbone.classifier = nn.Identity()
        num_features = 960  # MobileNetV3-Large avgpool output size

        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        # MobileNetV3 outputs (N, 960, 1, 1) after avgpool - need to flatten
        x = x.flatten(1)
        x = self.classifier(x)
        return x.squeeze(1)
    
    def unfreeze_backbone(self, unfreeze_percent=0.2):
        """Unfreeze top percentage of backbone layers."""
        total_layers = len(list(self.backbone.parameters()))
        unfreeze_count = int(total_layers * unfreeze_percent)
        
        for i, param in enumerate(self.backbone.parameters()):
            if i >= (total_layers - unfreeze_count):
                param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False


def get_transforms(augment=True):
    """Get data transforms."""
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class ModelTrainer:
    """Handles model training with class balancing."""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.best_val_acc = 0.0
    
    def create_model(self, pretrained=True):
        """Create the model."""
        self.model = EnhancedXception(pretrained=pretrained).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel created: {total_params:,} total params, {trainable_params:,} trainable")
        return self.model
    
    def train_from_datasets(self, train_dataset, val_dataset, epochs=15, batch_size=8, num_workers=0):
        """Train with pre-split datasets and progressive fine-tuning."""
        # Calculate class weights
        fake_count = sum(1 for _, l in train_dataset.base_samples if l == 0)
        real_count = sum(1 for _, l in train_dataset.base_samples if l == 1)
        total = fake_count + real_count

        weight_for_fake = total / (2 * fake_count)
        weight_for_real = total / (2 * real_count)

        print(f"\n{'='*80}")
        print(f"PYTORCH TRAINING - FAKE CURRENCY DETECTION")
        print(f"{'='*80}")
        print(f"\nDataset:")
        print(f"  Fake: {fake_count}, Real: {real_count}, Total: {total}")
        print(f"  Balance ratio: {real_count/max(fake_count,1):.2f}:1")
        print(f"  Class weights: Fake={weight_for_fake:.2f}x, Real={weight_for_real:.2f}x")
        print(f"\nDevice: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self._train_loop(train_loader, val_loader, class_weight_tensor=torch.tensor([weight_for_fake, weight_for_real], dtype=torch.float32).to(self.device), 
                        epochs=epochs, weight_for_fake=weight_for_fake, weight_for_real=weight_for_real)

    def train(self, train_dir, val_dir, epochs=15, batch_size=8):
        """Train with progressive fine-tuning using directory paths."""
        # Datasets
        train_dataset = CurrencyDataset(train_dir, transform=get_transforms(augment=True))
        val_dataset = CurrencyDataset(val_dir, transform=get_transforms(augment=False))

        # Calculate class weights
        fake_count = sum(1 for _, l in train_dataset.samples if l == 0)
        real_count = sum(1 for _, l in train_dataset.samples if l == 1)
        total = fake_count + real_count

        weight_for_fake = total / (2 * fake_count)
        weight_for_real = total / (2 * real_count)

        print(f"\n{'='*80}")
        print(f"PYTORCH TRAINING - FAKE CURRENCY DETECTION")
        print(f"{'='*80}")
        print(f"\nDataset:")
        print(f"  Fake: {fake_count}, Real: {real_count}, Total: {total}")
        print(f"  Balance ratio: {real_count/max(fake_count,1):.2f}:1")
        print(f"  Class weights: Fake={weight_for_fake:.2f}x, Real={weight_for_real:.2f}x")
        print(f"\nDevice: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        self._train_loop(train_loader, val_loader, class_weight_tensor=torch.tensor([weight_for_fake, weight_for_real], dtype=torch.float32).to(self.device),
                        epochs=epochs, weight_for_fake=weight_for_fake, weight_for_real=weight_for_real)
    
    def _train_loop(self, train_loader, val_loader, class_weight_tensor, epochs=15, weight_for_fake=1.0, weight_for_real=1.0):
        """Core training loop with progressive fine-tuning."""
        # Calculate phase split (roughly 53% head training, 47% fine-tuning)
        head_epochs = max(1, int(epochs * 0.53))
        finetune_epochs = epochs - head_epochs

        # Optimizer and scheduler
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)

        # Create class weight tensor
        class_weight_tensor = torch.tensor([weight_for_fake, weight_for_real], dtype=torch.float32).to(self.device)

        history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

        os.makedirs('models', exist_ok=True)

        # Phase 1: Train head only
        print(f"\n{'='*80}")
        print(f"PHASE 1: Training Classification Head (Epochs 1-{head_epochs})")
        print(f"{'='*80}")

        for epoch in range(head_epochs):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, class_weight_tensor)
            val_loss, val_acc = self._validate(val_loader, class_weight_tensor)
            scheduler.step(val_acc)

            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"\nEpoch [{epoch+1}/{head_epochs}]")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint('models/cnn_pytorch_best.pth')
                print(f"  ✓ Best model saved (Val Acc: {val_acc*100:.2f}%)")

        # Phase 2: Fine-tune backbone
        print(f"\n{'='*80}")
        print(f"PHASE 2: Fine-tuning Entire Model (Epochs {head_epochs+1}-{epochs})")
        print(f"{'='*80}")

        self.model.unfreeze_backbone(unfreeze_percent=0.3)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)

        for epoch in range(finetune_epochs):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, class_weight_tensor)
            val_loss, val_acc = self._validate(val_loader, class_weight_tensor)
            scheduler.step(val_acc)

            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"\nEpoch [{epoch+head_epochs+1}/{epochs}]")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint('models/cnn_pytorch_best.pth')
                print(f"  ✓ Best model saved (Val Acc: {val_acc*100:.2f}%)")
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'val_accuracy': self.best_val_acc,
            'class_weights': {'fake': weight_for_fake, 'real': weight_for_real},
            'architecture': 'xception_pytorch',
        }, 'models/cnn_pytorch_final.pth')
        
        # Save history
        with open('models/training_history_pytorch.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        print(f"Models saved to: models/")
        
        return self.model, history
    
    def _train_epoch(self, loader, optimizer, class_weights):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # Weighted loss
            weights = torch.where(labels == 0, class_weights[0], class_weights[1])
            loss = nn.BCELoss(reduction='none')(outputs, labels)
            loss = (loss * weights).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def _validate(self, loader, class_weights):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                weights = torch.where(labels == 0, class_weights[0], class_weights[1])
                loss = nn.BCELoss(reduction='none')(outputs, labels)
                loss = (loss * weights).mean()
                
                total_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def _save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'val_accuracy': self.best_val_acc,
        }, path)
    
    def load_model(self, path):
        """Load trained model."""
        if self.model is None:
            self.create_model(pretrained=False)
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Model loaded from {path}")
        return self.model
    
    def predict(self, image_tensor):
        """Make prediction on single image."""
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            confidence = output.item()
            result = "REAL" if confidence >= 0.5 else "FAKE"
            return result, confidence


if __name__ == "__main__":
    import argparse
    from torch.utils.data import random_split

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='dataset_downloads/preetrank',
                        help='Root directory containing fake/ and real/ folders')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Total epochs (8 head + 7 fine-tuning)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (reduce if OOM on GPU)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (0 for Windows)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"FAKE CURRENCY DETECTION - MODEL TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print(f"Data directory: {args.data_dir}")

    trainer = ModelTrainer(device=device)
    trainer.create_model()

    # Load full dataset, then split into train/val
    full_dataset = CurrencyDataset(args.data_dir, transform=get_transforms(augment=True))
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    print(f"\nDataset split:")
    print(f"  Total: {total_size}")
    print(f"  Train: {train_size}")
    print(f"  Val:   {val_size}")

    # Use same transform for val (no augmentation)
    val_transform = get_transforms(augment=False)
    
    # For proper split, we need separate datasets
    # Since transforms differ, we'll split indices and use Subset
    train_indices, val_indices = random_split(range(total_size), [train_size, val_size],
                                               generator=torch.Generator().manual_seed(42))
    
    # Create wrapper datasets with appropriate transforms
    class SplitDataset(Dataset):
        def __init__(self, base_dataset, indices, transform=None):
            self.base_samples = [base_dataset.samples[i] for i in indices]
            self.transform = transform
        
        def __len__(self):
            return len(self.base_samples)
        
        def __getitem__(self, idx):
            img_path, label = self.base_samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
    
    train_dataset = SplitDataset(full_dataset, train_indices, transform=get_transforms(augment=True))
    val_dataset = SplitDataset(full_dataset, val_indices, transform=get_transforms(augment=False))
    
    print(f"\n✓ Train dataset: {len(train_dataset)} images")
    print(f"✓ Val dataset:   {len(val_dataset)} images")

    trainer.train_from_datasets(train_dataset, val_dataset, 
                                 epochs=args.epochs, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
