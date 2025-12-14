"""
Request/Response Models (Pydantic Schemas)
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class AnomalyDetectRequest(BaseModel):
    """Single sequence anomaly detection request"""
    data: List[float] = Field(..., min_length=2048, max_length=2048)
    threshold: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "data": [0.1] * 2048,
                "threshold": 6.995087
            }
        }


class AnomalyDetectResponse(BaseModel):
    """Single sequence anomaly detection response"""
    is_anomaly: bool
    reconstruction_error: float
    threshold: float
    confidence: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "is_anomaly": True,
                "reconstruction_error": 8.5,
                "threshold": 6.995087,
                "confidence": 0.215,
                "timestamp": "2024-12-04T23:00:00"
            }
        }


class FileUploadResponse(BaseModel):
    """File upload batch anomaly detection response"""
    total_sequences: int
    anomalies_detected: int
    anomaly_rate: float
    anomaly_indices: List[int]
    reconstruction_errors: List[float]
    threshold: float
    processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "total_sequences": 100,
                "anomalies_detected": 5,
                "anomaly_rate": 0.05,
                "anomaly_indices": [23, 45, 67, 89, 91],
                "reconstruction_errors": [0.5, 1.2],
                "threshold": 6.995087,
                "processing_time_ms": 234.5
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str = "LSTM Autoencoder"
    loaded: bool
    threshold: float
    input_shape: tuple
    metrics: dict

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "LSTM Autoencoder",
                "loaded": True,
                "threshold": 6.995087,
                "input_shape": [2048, 1],
                "metrics": {
                    "precision": 0.93,
                    "recall": 0.92,
                    "f1_score": 0.925
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    models_loaded: bool

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-12-04T23:00:00",
                "version": "1.0.0",
                "models_loaded": True
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ProcessingError",
                "detail": "Failed to process CSV file",
                "timestamp": "2024-12-04T23:00:00"
            }
        }
