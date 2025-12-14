"""
Health Check Endpoint
"""
from fastapi import APIRouter, Depends
from api.models.schemas import HealthResponse
from api.dependencies import get_model_manager, ModelManager

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """
    Health check endpoint

    Returns:
        HealthResponse: API health status and model loading status
    """
    models_loaded = manager.is_loaded()

    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded
    )
