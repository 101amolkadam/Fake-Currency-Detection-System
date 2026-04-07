"""Currency analysis endpoint."""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import time
from datetime import datetime, timezone

from models.schemas import AnalyzeRequest, AnalysisResult
from services.cnn_classifier import classify_currency, is_model_loaded
from services.opencv_analyzer import analyze_security_features
from services.ensemble_engine import compute_ensemble_score
from services.image_annotator import generate_annotated_image, generate_thumbnail
from services.image_preprocessor import decode_base64_image, preprocess_image
from orm_models.analysis import CurrencyAnalysis, ResultType, ImageSource
from database import get_db

router = APIRouter(prefix="/api/v1/analyze", tags=["analyze"])


@router.post("", response_model=AnalysisResult)
async def analyze_currency(request: AnalyzeRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    
    try:
        # Step 1: Decode base64 image
        image, mime_type = decode_base64_image(request.image)
        
        # Step 2: Preprocess
        cnn_input, denoised, enhanced = preprocess_image(image)
        
        # Step 3: CNN Classification
        cnn_start = time.time()
        cnn_result, denom_result, denom_confidence, cnn_confidence = classify_currency(cnn_input)
        cnn_time_ms = int((time.time() - cnn_start) * 1000)
        
        # Step 4: OpenCV Security Feature Analysis
        features = analyze_security_features(image, denoised, enhanced, denom_result)

        # Step 5: Ensemble Decision
        ensemble_score, final_result, overall_confidence, feature_agreement, critical_failures = compute_ensemble_score(
            cnn_result, cnn_confidence, features
        )
        
        # Step 6: Generate Annotated Image & Thumbnail
        annotated_base64 = generate_annotated_image(image, {
            "overall_result": final_result,
            "ensemble_score": ensemble_score,
            **features
        })
        thumbnail_base64 = generate_thumbnail(image, max_size=200)
        
        # Step 7: Store in Database
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        analysis = CurrencyAnalysis(
            original_image_base64=request.image,
            annotated_image_base64=annotated_base64,
            thumbnail_base64=thumbnail_base64,
            image_mime_type=mime_type,
            result=ResultType(final_result),
            confidence=overall_confidence,
            ensemble_score=ensemble_score,
            currency_denomination=denom_result,
            denomination_confidence=denom_confidence,
            cnn_result=ResultType(cnn_result),
            cnn_confidence=cnn_confidence,
            cnn_processing_time_ms=cnn_time_ms,
            watermark_status=features["watermark"]["status"],
            watermark_confidence=features["watermark"]["confidence"],
            watermark_ssim_score=features["watermark"].get("ssim_score"),
            watermark_location=features["watermark"].get("location"),
            security_thread_status=features["security_thread"]["status"],
            security_thread_confidence=features["security_thread"]["confidence"],
            security_thread_position=features["security_thread"].get("position"),
            color_status=features["color_analysis"]["status"],
            color_confidence=features["color_analysis"]["confidence"],
            color_bhattacharyya_distance=features["color_analysis"].get("bhattacharyya_distance"),
            texture_status=features["texture_analysis"]["status"],
            texture_confidence=features["texture_analysis"]["confidence"],
            texture_glcm_contrast=features["texture_analysis"].get("glcm_contrast"),
            texture_glcm_energy=features["texture_analysis"].get("glcm_energy"),
            texture_sharpness=features["texture_analysis"].get("sharpness_score"),
            serial_number_status=features["serial_number"]["status"],
            serial_number_confidence=features["serial_number"]["confidence"],
            serial_number_extracted=features["serial_number"].get("extracted_text"),
            serial_number_format_valid=features["serial_number"].get("format_valid"),
            dimensions_status=features["dimensions"]["status"],
            dimensions_confidence=features["dimensions"]["confidence"],
            dimensions_aspect_ratio=features["dimensions"].get("aspect_ratio"),
            dimensions_deviation_percent=features["dimensions"].get("deviation_percent"),
            image_source=ImageSource(request.source),
            total_processing_time_ms=processing_time_ms,
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        analysis_id = analysis.id
        
        # Step 8: Build Response
        return AnalysisResult(
            id=analysis_id,
            result=final_result,
            confidence=overall_confidence,
            currency_denomination=denom_result,
            denomination_confidence=denom_confidence,
            analysis={
                "cnn_classification": {
                    "result": cnn_result,
                    "confidence": cnn_confidence,
                    "model": "MobileNetV3-Large" if is_model_loaded() else "OpenCV-only",
                    "processing_time_ms": cnn_time_ms,
                },
                "watermark": features["watermark"],
                "security_thread": features["security_thread"],
                "color_analysis": features["color_analysis"],
                "texture_analysis": features["texture_analysis"],
                "serial_number": features["serial_number"],
                "dimensions": features["dimensions"],
                "intaglio_printing": features.get("intaglio_printing", {}),
                "latent_image": features.get("latent_image", {}),
                "optically_variable_ink": features.get("optically_variable_ink", {}),
                "microlettering": features.get("microlettering", {}),
                "identification_mark": features.get("identification_mark", {}),
                "angular_lines": features.get("angular_lines", {}),
                "fluorescence": features.get("fluorescence", {}),
                "see_through_registration": features.get("see_through_registration", {}),
                "critical_failures": critical_failures,
                "feature_agreement": feature_agreement,
            },
            ensemble_score=ensemble_score,
            annotated_image=annotated_base64,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(timezone.utc),
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
