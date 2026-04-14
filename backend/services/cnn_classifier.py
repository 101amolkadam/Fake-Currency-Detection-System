"""CNN classifier service - PyTorch MobileNetV3-Large model inference with TTA.

This module provides currency note classification (REAL/FAKE) using a
MobileNetV3-Large CNN trained on Indian currency dataset with:
- Test-Time Augmentation (TTA) for robust predictions
- Temperature scaling for calibrated confidence
- GPU acceleration (CUDA) when available
"""
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from config import settings

# Model state
_model: Optional[nn.Module] = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_INPUT_SIZE = 224

# Log device initialization
print(f"[INFO] PyTorch backend initialized on {_device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[INFO] GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")


class MobileNetV3Currency(nn.Module):
    """MobileNetV3-Large model for binary currency authenticity classification."""
    
    def __init__(self):
        super().__init__()
        # Load pretrained MobileNetV3-Large backbone
        self.backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
        self.backbone.classifier = nn.Identity()  # Remove default classifier

        # Custom classification head
        num_features = 960  # MobileNetV3-Large avgpool output size
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x.squeeze(1)


def load_model() -> bool:
    """Load trained PyTorch model at startup.
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global _model
    
    try:
        # Search for model checkpoint
        model_dir = Path(settings.MODEL_PATH).parent
        model_paths = [
            model_dir / 'cnn_pytorch_best.pth',
            model_dir / 'cnn_pytorch_final.pth',
        ]
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            print("[WARN] No PyTorch model checkpoint found. Running in fallback mode.")
            return False

        print(f"[INFO] Loading PyTorch model from {model_path}...")
        _model = MobileNetV3Currency()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=_device, weights_only=True)
        
        # Handle both full checkpoint dict and state_dict only
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            _model.load_state_dict(checkpoint['model_state_dict'])
        else:
            _model.load_state_dict(checkpoint)

        _model.to(_device)
        _model.eval()

        # Warm up model with dummy input
        dummy_input = torch.zeros(1, 3, _INPUT_SIZE, _INPUT_SIZE).to(_device)
        with torch.no_grad():
            _model(dummy_input)

        print(f"[INFO] PyTorch model loaded successfully on {_device}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load PyTorch model: {e}")
        import traceback
        traceback.print_exc()
        _model = None
        return False


def get_model() -> Optional[nn.Module]:
    """Get the loaded model instance."""
    return _model


def is_model_loaded() -> bool:
    """Check if model is loaded and ready."""
    return _model is not None


def _apply_tta_augmentations(image_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Generate augmented versions for Test-Time Augmentation (TTA).
    
    Creates 7 variations:
    1. Original
    2. Horizontal flip
    3. Rotation +10°
    4. Rotation -10°
    5. Brightness +10%
    6. Brightness -10%
    7. Slight zoom (1.1x)
    
    Args:
        image_tensor: Input image tensor (C, H, W)
        
    Returns:
        List of augmented image tensors
    """
    augmented = [image_tensor]  # Original

    # Horizontal flip
    augmented.append(torch.flip(image_tensor, [2]))

    # Convert to numpy for geometric transformations
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    rows, cols = img_np.shape[:2]

    # Rotation +10°
    M_plus = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1.0)
    rotated_plus = cv2.warpAffine(img_np, M_plus, (cols, rows))
    augmented.append(
        torch.from_numpy(rotated_plus.transpose(2, 0, 1)).float().to(_device)
    )

    # Rotation -10°
    M_minus = cv2.getRotationMatrix2D((cols/2, rows/2), -10, 1.0)
    rotated_minus = cv2.warpAffine(img_np, M_minus, (cols, rows))
    augmented.append(
        torch.from_numpy(rotated_minus.transpose(2, 0, 1)).float().to(_device)
    )

    # Brightness +10%
    augmented.append(torch.clamp(image_tensor * 1.1, -1.0, 1.0))

    # Brightness -10%
    augmented.append(torch.clamp(image_tensor * 0.9, -1.0, 1.0))

    # Zoom 1.1x
    zoomed_np = cv2.resize(img_np, (int(cols * 1.1), int(rows * 1.1)))
    crop_y = int((rows * 0.1) // 2)
    crop_x = int((cols * 0.1) // 2)
    zoomed_np = zoomed_np[crop_y:crop_y + rows, crop_x:crop_x + cols]
    augmented.append(
        torch.from_numpy(zoomed_np.transpose(2, 0, 1)).float().to(_device)
    )

    return augmented


def _calibrate_confidence(raw_confidence: float, temperature: float = 1.5) -> float:
    """Apply temperature scaling to calibrate model confidence.
    
    Args:
        raw_confidence: Raw model output (0-1)
        temperature: Temperature scaling parameter (>1 softens confidence)
        
    Returns:
        Calibrated confidence score
    """
    epsilon = 1e-7
    raw_confidence = np.clip(raw_confidence, epsilon, 1 - epsilon)
    
    # Convert to logit space
    logit = np.log(raw_confidence / (1 - raw_confidence))
    
    # Apply temperature scaling
    calibrated_logit = logit / temperature
    
    # Convert back to probability
    calibrated = 1 / (1 + np.exp(-calibrated_logit))

    return float(calibrated)


def classify_currency(
    preprocessed_image: np.ndarray,
    use_tta: bool = True
) -> tuple[str, str, float, float]:
    """Run CNN inference for authenticity classification.
    
    Uses Test-Time Augmentation (TTA) for robust predictions.
    
    Args:
        preprocessed_image: numpy array, shape (224, 224, 3) or (3, 224, 224)
                           ImageNet normalized
        use_tta: Whether to use test-time augmentation (default: True)
        
    Returns:
        Tuple of (authenticity_result, denomination_result, denom_confidence, auth_confidence)
        - authenticity_result: "REAL" or "FAKE"
        - denomination_result: Detected denomination (currently defaults to ₹500)
        - denom_confidence: Confidence in denomination (currently 0.5)
        - auth_confidence: Confidence in authenticity (0-1)
    """
    global _model

    if _model is None:
        # Fallback when model not loaded
        return "REAL", "₹500", 0.5, 0.5

    try:
        # Convert to tensor if needed
        if preprocessed_image.shape[0] == 3:  # Already (C, H, W)
            image_tensor = torch.from_numpy(preprocessed_image).float().to(_device)
        else:  # (H, W, C)
            image_tensor = torch.from_numpy(
                preprocessed_image.transpose(2, 0, 1)
            ).float().to(_device)

        _model.eval()

        if use_tta:
            augmented_images = _apply_tta_augmentations(image_tensor)
            authenticity_scores = []

            for aug_img in augmented_images:
                with torch.no_grad():
                    output = _model(aug_img.unsqueeze(0))
                    authenticity_scores.append(output.item())

            # Average predictions and calculate variance
            auth_score = float(np.mean(authenticity_scores))
            auth_std = float(np.std(authenticity_scores))
        else:
            # Single prediction
            with torch.no_grad():
                output = _model(image_tensor.unsqueeze(0))
                auth_score = output.item()
                auth_std = 0.0

        # Apply temperature scaling calibration
        auth_score = _calibrate_confidence(auth_score)

        # Determine authenticity result
        authenticity_result = "REAL" if auth_score >= 0.5 else "FAKE"
        auth_confidence = auth_score if auth_score >= 0.5 else 1.0 - auth_score

        # Reduce confidence if TTA has high variance (uncertain prediction)
        if use_tta and auth_std > 0.1:
            penalty = min(0.2, auth_std * 0.5)
            auth_confidence = max(0.5, auth_confidence - penalty)

        # Denomination (single output model - defaults to ₹500)
        # TODO: Implement multi-output model for denomination classification
        denomination_result = "₹500"
        denom_confidence = 0.5

        return (
            authenticity_result,
            denomination_result,
            round(denom_confidence, 4),
            round(auth_confidence, 4)
        )
        
    except Exception as e:
        print(f"[ERROR] PyTorch CNN classification failed: {e}")
        import traceback
        traceback.print_exc()
        return "REAL", "₹500", 0.5, 0.5
