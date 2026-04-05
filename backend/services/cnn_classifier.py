"""CNN classifier service - PyTorch Xception model inference with test-time augmentation."""
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
from config import settings
from PIL import Image
import torchvision.transforms as transforms

_model = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_input_size = 299

# Print device info on import
print(f"[INFO] PyTorch backend initialized on {_device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB)")


class XceptionCurrency(nn.Module):
    """PyTorch Xception model for currency classification."""
    def __init__(self):
        super().__init__()
        # Load pretrained Xception
        self.backbone = models.xception(weights=models.Xception_Weights.IMAGENET1K_V1)
        num_features = self.backbone.last_linear.in_features
        self.backbone.last_linear = nn.Identity()
        
        # Custom classification head (same architecture as training)
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
        x = self.classifier(x)
        return x.squeeze(1)


def load_model():
    """Load trained PyTorch Xception model at startup."""
    global _model
    try:
        # Try PyTorch .pth format first, then fallback to .keras/.h5
        pth_path = os.path.join(os.path.dirname(settings.MODEL_PATH), 'cnn_pytorch_best.pth')
        pth_final = os.path.join(os.path.dirname(settings.MODEL_PATH), 'cnn_pytorch_final.pth')
        
        model_path = None
        if os.path.exists(pth_path):
            model_path = pth_path
        elif os.path.exists(pth_final):
            model_path = pth_final
        else:
            print(f"[WARN] No PyTorch model found. Trying TensorFlow model...")
            return False

        print(f"[INFO] Loading PyTorch model from {model_path}...")
        _model = XceptionCurrency()
        checkpoint = torch.load(model_path, map_location=_device, weights_only=True)
        
        # Handle both full checkpoint and state_dict only
        if 'model_state_dict' in checkpoint:
            _model.load_state_dict(checkpoint['model_state_dict'])
        else:
            _model.load_state_dict(checkpoint)
        
        _model.to(_device)
        _model.eval()
        
        # Warm up
        dummy = torch.zeros(1, 3, _input_size, _input_size).to(_device)
        with torch.no_grad():
            _model(dummy)
        
        print(f"[INFO] PyTorch model loaded successfully on {_device}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load PyTorch model: {e}")
        import traceback
        traceback.print_exc()
        _model = None
        return False


def get_model():
    return _model


def is_model_loaded():
    return _model is not None


def _apply_tta_augmentations(image_tensor: torch.Tensor) -> list:
    """
    Generate augmented versions for Test-Time Augmentation (TTA).
    Creates 7 variations:
    1. Original
    2. Horizontal flip
    3. Rotation +10°
    4. Rotation -10°
    5. Brightness +10%
    6. Brightness -10%
    7. Slight zoom (1.1x)
    """
    augmented = [image_tensor]  # Original
    
    # Horizontal flip
    augmented.append(torch.flip(image_tensor, [2]))
    
    # For rotations and other augmentations, convert to numpy and back
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Rotation +10°
    rows, cols = img_np.shape[:2]
    M_plus = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1.0)
    rotated_plus = cv2.warpAffine(img_np, M_plus, (cols, rows))
    augmented.append(torch.from_numpy(rotated_plus.transpose(2, 0, 1)).float().to(_device))
    
    # Rotation -10°
    M_minus = cv2.getRotationMatrix2D((cols/2, rows/2), -10, 1.0)
    rotated_minus = cv2.warpAffine(img_np, M_minus, (cols, rows))
    augmented.append(torch.from_numpy(rotated_minus.transpose(2, 0, 1)).float().to(_device))
    
    # Brightness +10%
    brighter = torch.clamp(image_tensor * 1.1, -1.0, 1.0)
    augmented.append(brighter)
    
    # Brightness -10%
    darker = torch.clamp(image_tensor * 0.9, -1.0, 1.0)
    augmented.append(darker)
    
    # Zoom 1.1x
    zoomed_np = cv2.resize(img_np, (int(cols * 1.1), int(rows * 1.1)))
    zoomed_np = zoomed_np[(rows*0.1)//2:(rows*0.1)//2 + rows, (cols*0.1)//2:(cols*0.1)//2 + cols]
    augmented.append(torch.from_numpy(zoomed_np.transpose(2, 0, 1)).float().to(_device))
    
    return augmented


def _calibrate_confidence(raw_confidence: float) -> float:
    """Apply temperature scaling to calibrate model confidence."""
    temperature = 1.5
    
    epsilon = 1e-7
    raw_confidence = np.clip(raw_confidence, epsilon, 1 - epsilon)
    logit = np.log(raw_confidence / (1 - raw_confidence))
    calibrated_logit = logit / temperature
    calibrated = 1 / (1 + np.exp(-calibrated_logit))
    
    return calibrated


def classify_currency(preprocessed_image: np.ndarray, use_tta: bool = True) -> tuple:
    """
    Run CNN inference for authenticity classification.
    Uses Test-Time Augmentation (TTA) for robust predictions.
    
    Args:
        preprocessed_image: numpy array shape (299, 299, 3), values in [-1, 1]
        use_tta: Whether to use test-time augmentation
    
    Returns:
        (authenticity_result, denomination_result, denom_confidence, auth_confidence)
    """
    global _model
    
    if _model is None:
        return "REAL", "₹500", 0.5, 0.5
    
    try:
        # Convert numpy to tensor: (H, W, C) -> (C, H, W) -> add batch dim
        if preprocessed_image.shape[0] == 3:  # Already (C, H, W)
            image_tensor = torch.from_numpy(preprocessed_image).float().to(_device)
        else:  # (H, W, C)
            image_tensor = torch.from_numpy(preprocessed_image.transpose(2, 0, 1)).float().to(_device)
        
        _model.eval()
        
        if use_tta:
            augmented_images = _apply_tta_augmentations(image_tensor)
            
            authenticity_scores = []
            
            for aug_img in augmented_images:
                with torch.no_grad():
                    output = _model(aug_img.unsqueeze(0))
                    authenticity_scores.append(output.item())
            
            # Average predictions
            auth_score = np.mean(authenticity_scores)
            auth_std = np.std(authenticity_scores)
        else:
            # Single prediction
            with torch.no_grad():
                output = _model(image_tensor.unsqueeze(0))
                auth_score = output.item()
                auth_std = 0.0
        
        # Apply calibration
        auth_score = _calibrate_confidence(auth_score)
        
        # Authenticity: sigmoid output (0=FAKE, 1=REAL)
        authenticity_result = "REAL" if auth_score >= 0.5 else "FAKE"
        auth_confidence = auth_score if auth_score >= 0.5 else 1.0 - auth_score
        
        # Reduce confidence if TTA has high variance
        if use_tta and auth_std > 0.1:
            penalty = min(0.2, auth_std * 0.5)
            auth_confidence = max(0.5, auth_confidence - penalty)
        
        # Denomination (single output model - default to ₹500)
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
