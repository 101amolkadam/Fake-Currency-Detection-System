"""History endpoints -- list, get, and delete past analyses."""

import logging
import math
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from models.schemas import AnalysisResult, HistoryItem, HistoryResponse, PaginationInfo
from orm_models.analysis import CurrencyAnalysis, ResultType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analyze", tags=["history"])


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    filter: str = Query("all", pattern="^(all|real|fake)$"),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """Return a paginated list of past currency analyses.

    Args:
        page: 1-based page number.
        limit: Number of items per page (max 100).
        filter: One of ``"all"``, ``"real"``, or ``"fake"``.
        db: Active database session.
    """
    query = db.query(CurrencyAnalysis)
    count_query = db.query(func.count(CurrencyAnalysis.id))

    if filter != "all":
        result_val = ResultType.REAL if filter == "real" else ResultType.FAKE
        query = query.filter(CurrencyAnalysis.result == result_val)
        count_query = count_query.filter(CurrencyAnalysis.result == result_val)

    total: int = count_query.scalar() or 0
    total_pages: int = math.ceil(total / limit) if total > 0 else 1

    analyses: List[CurrencyAnalysis] = (
        query.order_by(CurrencyAnalysis.analyzed_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    items: List[HistoryItem] = [
        HistoryItem(
            id=a.id,
            result=a.result.value,
            confidence=float(a.confidence) if a.confidence else 0.0,
            denomination=a.currency_denomination,
            thumbnail=a.thumbnail_base64,
            analyzed_at=a.analyzed_at,
        )
        for a in analyses
    ]

    return HistoryResponse(
        data=items,
        pagination=PaginationInfo(
            page=page,
            limit=limit,
            total=total,
            total_pages=total_pages,
        ),
    )


@router.get("/history/{analysis_id}")
async def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
) -> AnalysisResult:
    """Return the full analysis result for a single record."""
    analysis: CurrencyAnalysis | None = (
        db.query(CurrencyAnalysis)
        .filter(CurrencyAnalysis.id == analysis_id)
        .first()
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisResult(
        id=analysis.id,
        result=analysis.result.value,
        confidence=float(analysis.confidence) if analysis.confidence else 0.0,
        currency_denomination=analysis.currency_denomination,
        denomination_confidence=(
            float(analysis.denomination_confidence)
            if analysis.denomination_confidence
            else None
        ),
        analysis={
            "cnn_classification": {
                "result": analysis.cnn_result.value if analysis.cnn_result else "UNKNOWN",
                "confidence": float(analysis.cnn_confidence) if analysis.cnn_confidence else 0.0,
                "model": analysis.cnn_model or "Xception",
                "processing_time_ms": analysis.cnn_processing_time_ms or 0,
            },
            "watermark": {
                "status": analysis.watermark_status or "unknown",
                "confidence": (
                    float(analysis.watermark_confidence)
                    if analysis.watermark_confidence
                    else 0.0
                ),
                "location": analysis.watermark_location,
                "ssim_score": (
                    float(analysis.watermark_ssim_score)
                    if analysis.watermark_ssim_score
                    else None
                ),
            },
            "security_thread": {
                "status": analysis.security_thread_status or "unknown",
                "confidence": (
                    float(analysis.security_thread_confidence)
                    if analysis.security_thread_confidence
                    else 0.0
                ),
                "position": analysis.security_thread_position,
                "coordinates": None,
            },
            "color_analysis": {
                "status": analysis.color_status or "unknown",
                "confidence": (
                    float(analysis.color_confidence)
                    if analysis.color_confidence
                    else 0.0
                ),
                "bhattacharyya_distance": (
                    float(analysis.color_bhattacharyya_distance)
                    if analysis.color_bhattacharyya_distance
                    else None
                ),
                "dominant_colors": None,
            },
            "texture_analysis": {
                "status": analysis.texture_status or "unknown",
                "confidence": (
                    float(analysis.texture_confidence)
                    if analysis.texture_confidence
                    else 0.0
                ),
                "glcm_contrast": (
                    float(analysis.texture_glcm_contrast)
                    if analysis.texture_glcm_contrast
                    else None
                ),
                "glcm_energy": (
                    float(analysis.texture_glcm_energy)
                    if analysis.texture_glcm_energy
                    else None
                ),
                "sharpness_score": (
                    float(analysis.texture_sharpness)
                    if analysis.texture_sharpness
                    else None
                ),
            },
            "serial_number": {
                "status": analysis.serial_number_status or "unknown",
                "confidence": (
                    float(analysis.serial_number_confidence)
                    if analysis.serial_number_confidence
                    else 0.0
                ),
                "extracted_text": analysis.serial_number_extracted,
                "format_valid": analysis.serial_number_format_valid,
            },
            "dimensions": {
                "status": analysis.dimensions_status or "unknown",
                "confidence": (
                    float(analysis.dimensions_confidence)
                    if analysis.dimensions_confidence
                    else 0.0
                ),
                "aspect_ratio": (
                    float(analysis.dimensions_aspect_ratio)
                    if analysis.dimensions_aspect_ratio
                    else None
                ),
                "expected_aspect_ratio": 1.69,
                "deviation_percent": (
                    float(analysis.dimensions_deviation_percent)
                    if analysis.dimensions_deviation_percent
                    else None
                ),
            },
        },
        ensemble_score=(
            float(analysis.ensemble_score) if analysis.ensemble_score else 0.0
        ),
        annotated_image=analysis.annotated_image_base64,
        processing_time_ms=analysis.total_processing_time_ms or 0,
        timestamp=analysis.analyzed_at,
    )


@router.delete("/history/{analysis_id}")
async def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Delete a past analysis record."""
    analysis: CurrencyAnalysis | None = (
        db.query(CurrencyAnalysis)
        .filter(CurrencyAnalysis.id == analysis_id)
        .first()
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    db.delete(analysis)
    db.commit()
    return {"message": "Analysis deleted successfully"}
