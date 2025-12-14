"""
FastAPI Main Application
NASA Bearing Anomaly Detection API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

from api.routers import health, anomaly
from api.dependencies import get_model_manager
from api.models.schemas import ErrorResponse

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="NASA Bearing Anomaly Detection API",
    description="REST API for bearing anomaly detection using LSTM Autoencoder",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Set max file upload size to 3GB (for full bearing data files)
app.state.max_file_size = 3 * 1024 * 1024 * 1024  # 3GB in bytes

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(anomaly.router)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="ValidationError",
            detail=str(exc)
        ).model_dump()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            detail=str(exc.detail)
        ).model_dump()
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("="*60)
    logger.info("Starting NASA Bearing Anomaly Detection API")
    logger.info("="*60)

    manager = get_model_manager()

    try:
        logger.info("Loading models...")
        manager.load_model()
        manager.load_scaler()
        manager.load_preprocessor()
        logger.info("Models loaded successfully")
        logger.info(f"  - LSTM Autoencoder: OK")
        logger.info(f"  - StandardScaler: OK")
        logger.info(f"  - Preprocessor: OK")
        logger.info(f"  - Threshold: {manager.model.threshold:.6f}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error("API will start but may not function properly")

    logger.info("="*60)
    logger.info("API is ready to accept requests")
    logger.info("  - Swagger UI: http://localhost:8000/docs")
    logger.info("  - ReDoc: http://localhost:8000/redoc")
    logger.info("="*60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint - API information

    Returns:
        dict: API information
    """
    return {
        "message": "NASA Bearing Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "anomaly_detection": "/api/anomaly/detect-file",
            "model_info": "/api/anomaly/models/info"
        }
    }


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
