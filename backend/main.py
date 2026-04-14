"""FastAPI main application with modern lifespan events."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

from config import settings
from database import init_db, engine
from routers import analyze, history

# Import model loader to trigger startup loading
import services.model_loader  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    print("[INFO] Initializing Fake Currency Detection System...")
    
    # Initialize database
    print("[INFO] Initializing database...")
    init_db()
    print("[INFO] Database initialized successfully.")
    
    # Record startup time
    app.state.start_time = time.time()
    
    print("[INFO] System ready. Model loaded: {}".format(
        services.model_loader.is_model_loaded()
    ))
    
    yield
    
    # Shutdown (if needed)
    print("[INFO] Shutting down Fake Currency Detection System...")


# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)

# FastAPI application
app = FastAPI(
    title="Fake Currency Detection API",
    description="Hybrid CNN + OpenCV ensemble currency authentication system for Indian currency",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS middleware configuration
origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up rate limiter
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "RATE_LIMITED",
                "message": "Too many requests. Please try again later.",
                "retry_after_seconds": 60
            }
        }
    )


@app.get("/api/v1/health", tags=["health"])
async def health_check():
    """Health check endpoint with system status."""
    db_ok = False
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        pass

    uptime = int(time.time() - getattr(app.state, "start_time", time.time()))

    return {
        "status": "healthy" if (db_ok and services.model_loader.is_model_loaded()) else "degraded",
        "model_loaded": services.model_loader.is_model_loaded(),
        "database_connected": db_ok,
        "uptime_seconds": uptime,
        "version": "2.0.0"
    }


@app.get("/api/v1/model/info", tags=["model"])
async def model_info():
    """Model metadata endpoint."""
    return {
        "architecture": "MobileNetV3-Large (ImageNet pretrained + fine-tuned)",
        "parameters": "3.25M",
        "input_size": "224x224",
        "status": "loaded" if services.model_loader.is_model_loaded() else "not_loaded",
        "supported_denominations": [
            "₹10", "₹20", "₹50", "₹100", "₹200", "₹500", "₹2000"
        ],
        "fallback_mode": not services.model_loader.is_model_loaded(),
        "features": [
            "Test-Time Augmentation (TTA)",
            "Temperature Scaling Calibration",
            "15 OpenCV Security Features",
            "Critical Feature Override"
        ]
    }


# Register API routers
app.include_router(analyze.router)
app.include_router(history.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
