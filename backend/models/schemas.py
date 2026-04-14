"""Pydantic schemas for request / response models."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import base64
import re

from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for the currency-analysis endpoint.

    The ``image`` field must be a base64-encoded data URI in the format
    ``data:image/<type>;base64,<data>`` where ``<type>`` is one of
    ``jpeg``, ``png``, or ``webp``.  The decoded payload must not exceed 10 MB.
    """

    image: str  # Base64 data URI: "data:image/jpeg;base64,..."
    source: str = "upload"  # "upload" or "camera"

    @field_validator("image")
    @classmethod
    def validate_base64_image(cls, v: str) -> str:
        pattern = r"^data:image/(jpeg|png|webp);base64,[A-Za-z0-9+/]+=*$"
        if not re.match(pattern, v):
            raise ValueError("Invalid base64 image format. Expected: data:image/<type>;base64,<data>")

        mime_match = re.match(r"data:image/(\w+);base64,", v)
        if not mime_match:
            raise ValueError("Could not extract MIME type from data URI")

        mime_type = mime_match.group(1)
        if mime_type not in ("jpeg", "png", "webp"):
            raise ValueError(f"Unsupported image type: {mime_type}. Supported: jpeg, png, webp")

        header, encoded_data = v.split(",", 1)
        try:
            decoded_bytes = base64.b64decode(encoded_data)
        except Exception as exc:
            raise ValueError("Invalid base64 encoding") from exc

        max_size = 10 * 1024 * 1024  # 10 MB
        if len(decoded_bytes) > max_size:
            raise ValueError(
                f"Image size ({len(decoded_bytes) / 1024 / 1024:.1f}MB) exceeds 10MB limit"
            )

        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in ("upload", "camera"):
            raise ValueError("Source must be 'upload' or 'camera'")
        return v


# ---------------------------------------------------------------------------
# Nested response schemas
# ---------------------------------------------------------------------------

class WatermarkAnalysis(BaseModel):
    status: str
    confidence: float
    location: Optional[Dict[str, Any]] = None
    ssim_score: Optional[float] = None


class SecurityThreadAnalysis(BaseModel):
    status: str
    confidence: float
    position: Optional[str] = None
    coordinates: Optional[Dict[str, Any]] = None


class ColorAnalysis(BaseModel):
    status: str
    confidence: float
    bhattacharyya_distance: Optional[float] = None
    dominant_colors: Optional[List[str]] = None


class TextureAnalysis(BaseModel):
    status: str
    confidence: float
    glcm_contrast: Optional[float] = None
    glcm_energy: Optional[float] = None
    sharpness_score: Optional[float] = None


class SerialNumberAnalysis(BaseModel):
    status: str
    confidence: float
    extracted_text: Optional[str] = None
    format_valid: Optional[bool] = None


class DimensionsAnalysis(BaseModel):
    status: str
    confidence: float
    aspect_ratio: Optional[float] = None
    expected_aspect_ratio: Optional[float] = None
    deviation_percent: Optional[float] = None


class CNNClassification(BaseModel):
    result: str
    confidence: float
    model: str = "Xception"
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Top-level response schemas
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """Complete analysis result returned by the analyse endpoint."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [{"annotated_image": "data:image/jpeg;base64,..."}],
        },
    )

    id: int
    result: str
    confidence: float
    currency_denomination: Optional[str] = None
    denomination_confidence: Optional[float] = None
    analysis: Dict[str, Any]
    ensemble_score: float
    annotated_image: str
    processing_time_ms: int
    timestamp: datetime


class HistoryItem(BaseModel):
    """Compact representation for history listing."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    result: str
    confidence: float
    denomination: Optional[str] = None
    thumbnail: str
    analyzed_at: datetime


class PaginationInfo(BaseModel):
    page: int
    limit: int
    total: int
    total_pages: int


class HistoryResponse(BaseModel):
    data: List[HistoryItem]
    pagination: PaginationInfo


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    uptime_seconds: int
    version: str = "1.0.0"
