"""FastAPI main application."""
import time
import pymysql
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

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Fake Currency Detection API",
    description="Hybrid CNN + OpenCV ensemble currency authentication",
    version="1.0.0",
)

# CORS
origins = settings.ALLOWED_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": {"code": "RATE_LIMITED", "message": "Too many requests. Try again later."}}
    )


@app.on_event("startup")
def startup():
    """Initialize database and load models on startup."""
    print("[INFO] Initializing database...")
    init_db()
    print("[INFO] Database initialized.")
    start_time = time.time()
    app.state.start_time = start_time


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    db_ok = False
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        pass
    
    import services.model_loader
    uptime = int(time.time() - getattr(app.state, "start_time", time.time()))
    
    return {
        "status": "healthy",
        "model_loaded": services.model_loader.is_model_loaded(),
        "database_connected": db_ok,
        "uptime_seconds": uptime,
        "version": "1.0.0"
    }


@app.get("/api/v1/model/info")
async def model_info():
    """Model metadata endpoint."""
    return {
        "architecture": "Xception (ImageNet pretrained + fine-tuned)",
        "status": "loaded" if services.model_loader.is_model_loaded() else "not_loaded",
        "supported_denominations": ["₹10", "₹20", "₹50", "₹100", "₹200", "₹500", "₹2000"],
        "fallback_mode": not services.model_loader.is_model_loaded(),
    }


# Register routers
app.include_router(analyze.router)
app.include_router(history.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
