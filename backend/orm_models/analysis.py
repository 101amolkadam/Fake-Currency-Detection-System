"""Currency analysis ORM model for MySQL database."""
from sqlalchemy import (
    Column, BigInteger, String, Enum, DECIMAL, JSON, Boolean,
    Integer, TIMESTAMP, Index, Text
)
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMTEXT
from sqlalchemy.sql import func
from database import Base
import enum


class ResultType(str, enum.Enum):
    """Analysis result enumeration."""
    REAL = "REAL"
    FAKE = "FAKE"


class ImageSource(str, enum.Enum):
    """Image source enumeration."""
    UPLOAD = "upload"
    CAMERA = "camera"


class CurrencyAnalysis(Base):
    """ORM model for storing currency analysis results."""
    __tablename__ = "currency_analyses"

    # Primary key
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Base64 encoded images (large text fields)
    original_image_base64 = Column(LONGTEXT, nullable=False, comment="Original uploaded image")
    annotated_image_base64 = Column(LONGTEXT, nullable=False, comment="Annotated result image")
    thumbnail_base64 = Column(MEDIUMTEXT, nullable=False, comment="Small thumbnail image")
    image_mime_type = Column(String(20), nullable=False, default="image/jpeg")

    # Overall analysis result
    result = Column(Enum(ResultType), nullable=False, index=True)
    confidence = Column(DECIMAL(5, 4), nullable=False, comment="Overall confidence (0-1)")
    ensemble_score = Column(DECIMAL(5, 4), nullable=False, comment="Ensemble scoring result")

    # Currency denomination detection
    currency_denomination = Column(String(50), nullable=True, index=True)
    denomination_confidence = Column(DECIMAL(5, 4), nullable=True)

    # CNN Classification results
    cnn_result = Column(Enum(ResultType), nullable=True)
    cnn_confidence = Column(DECIMAL(5, 4), nullable=True)
    cnn_model = Column(String(100), default="MobileNetV3-Large")
    cnn_processing_time_ms = Column(Integer, nullable=True)

    # Watermark Analysis
    watermark_status = Column(String(50), nullable=True)
    watermark_confidence = Column(DECIMAL(5, 4), nullable=True)
    watermark_ssim_score = Column(DECIMAL(5, 4), nullable=True)
    watermark_location = Column(JSON, nullable=True, comment="Bounding box coordinates")

    # Security Thread Analysis
    security_thread_status = Column(String(50), nullable=True)
    security_thread_confidence = Column(DECIMAL(5, 4), nullable=True)
    security_thread_position = Column(String(50), nullable=True)

    # Color Analysis
    color_status = Column(String(50), nullable=True)
    color_confidence = Column(DECIMAL(5, 4), nullable=True)
    color_bhattacharyya_distance = Column(DECIMAL(6, 4), nullable=True)

    # Texture Analysis
    texture_status = Column(String(50), nullable=True)
    texture_confidence = Column(DECIMAL(5, 4), nullable=True)
    texture_glcm_contrast = Column(DECIMAL(6, 4), nullable=True)
    texture_glcm_energy = Column(DECIMAL(6, 4), nullable=True)
    texture_sharpness = Column(DECIMAL(5, 4), nullable=True)

    # Serial Number Detection
    serial_number_status = Column(String(50), nullable=True)
    serial_number_confidence = Column(DECIMAL(5, 4), nullable=True)
    serial_number_extracted = Column(String(100), nullable=True)
    serial_number_format_valid = Column(Boolean, nullable=True)

    # Dimensions Verification
    dimensions_status = Column(String(50), nullable=True)
    dimensions_confidence = Column(DECIMAL(5, 4), nullable=True)
    dimensions_aspect_ratio = Column(DECIMAL(6, 4), nullable=True)
    dimensions_deviation_percent = Column(DECIMAL(5, 2), nullable=True)

    # Metadata
    image_source = Column(Enum(ImageSource), default=ImageSource.UPLOAD)
    total_processing_time_ms = Column(Integer, nullable=True)
    analyzed_at = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        index=True,
        comment="Analysis timestamp"
    )
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )

    # Database indexes for query optimization
    __table_args__ = (
        Index("idx_result", "result"),
        Index("idx_denomination", "currency_denomination"),
        Index("idx_analyzed_at", "analyzed_at"),
        Index("idx_confidence", "confidence"),
        Index("idx_result_analyzed", "result", "analyzed_at"),
    )

    def __repr__(self):
        return f"<CurrencyAnalysis(id={self.id}, result={self.result}, confidence={self.confidence})>"
