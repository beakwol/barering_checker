"""
Anomaly Detection Endpoints
"""
import numpy as np
import pandas as pd
import time
import io
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from typing import Optional

from api.models.schemas import (
    AnomalyDetectRequest,
    AnomalyDetectResponse,
    FileUploadResponse,
    ModelInfoResponse
)
from api.dependencies import get_model_manager, ModelManager

router = APIRouter(prefix="/api/anomaly", tags=["anomaly"])


@router.post("/detect", response_model=AnomalyDetectResponse)
async def detect_anomaly_single(
    request: AnomalyDetectRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Detect anomaly in a single sequence (2048 samples)

    Args:
        request: AnomalyDetectRequest with data and optional threshold

    Returns:
        AnomalyDetectResponse: Detection result with error and confidence
    """
    try:
        model = manager.model
        threshold = request.threshold or model.threshold

        # Prepare data: (1, 2048, 1)
        data = np.array(request.data, dtype=np.float32).reshape(1, 2048, 1)

        # Compute reconstruction error
        error = model.compute_reconstruction_error(data)[0]
        is_anomaly = error > threshold
        confidence = abs(error - threshold) / threshold

        return AnomalyDetectResponse(
            is_anomaly=bool(is_anomaly),
            reconstruction_error=float(error),
            threshold=float(threshold),
            confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@router.post("/detect-file", response_model=FileUploadResponse)
async def detect_anomaly_file(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Detect anomalies in uploaded CSV file

    Full preprocessing pipeline:
    1. Load CSV
    2. Downsample (20kHz → 2kHz)
    3. Bandpass filter (10-5000 Hz)
    4. Channel combination (RMS)
    5. Create sequences (2048 samples, 50% overlap)
    6. Normalize using loaded scaler
    7. Predict with model

    Args:
        file: CSV file with vibration data
        threshold: Optional custom threshold

    Returns:
        FileUploadResponse: Batch detection results
    """
    start_time = time.time()

    try:
        # 1. Read CSV
        contents = await file.read()
        print(f"DEBUG: File size: {len(contents) / 1024 / 1024:.2f} MB")
        df = pd.read_csv(io.BytesIO(contents))
        print(f"DEBUG: DataFrame shape: {df.shape}")

        if df.shape[1] < 2:
            raise HTTPException(
                status_code=400,
                detail="CSV must have at least 2 columns (timestamp, value)"
            )

        # Extract data (column 1+ are channels)
        # Assuming column 0 is timestamp, column 1+ are values
        data = df.iloc[:, 1:].values
        print(f"DEBUG: Data shape after extraction: {data.shape}")

        if data.shape[0] < 2048:
            raise HTTPException(
                status_code=400,
                detail=f"Data too short. Need at least 2048 samples, got {data.shape[0]}"
            )

        preprocessor = manager.preprocessor
        model = manager.model

        # 2. Check if data is already preprocessed (2kHz)
        # Heuristic: If treating as 2kHz gives >200 seconds, it's probably already downsampled
        # But if treating as 20kHz gives reasonable duration (10-60 seconds), it's likely raw 20kHz data
        duration_at_2khz = data.shape[0] / 2000
        duration_at_20khz = data.shape[0] / 20000

        # More accurate heuristic:
        # - If 20kHz duration is reasonable (10-60s), treat as raw 20kHz data
        # - If 2kHz duration is reasonable (10-60s) AND 20kHz would be too short (<5s), treat as preprocessed
        # - For very long data (>1000s at 20kHz), assume it's raw 20kHz data
        is_preprocessed = (duration_at_20khz < 5) and (10 <= duration_at_2khz <= 60)
        
        # For very long files (full bearing data), always treat as raw 20kHz
        if duration_at_20khz > 1000:
            is_preprocessed = False
            print(f"Very long data detected ({duration_at_20khz:.1f}s at 20kHz), treating as raw 20kHz data")
        
        if is_preprocessed:
            print(f"Data appears to be already preprocessed (2kHz, {duration_at_2khz:.2f}s)")
            # Skip downsampling and filtering
            # RMS 결합 (학습 시와 동일하게)
            combined = preprocessor.combine_channels(data, method='rms')
        else:
            print(f"Applying full preprocessing (20kHz → 2kHz, {duration_at_20khz:.2f}s)")
            # Full preprocessing pipeline (downsample + filter)
            downsampled = preprocessor.downsample(data, method='decimate')
            filtered = preprocessor.apply_bandpass_filter(downsampled)

            # RMS 결합 (학습 시와 동일하게, 1채널이어도 RMS 적용)
            # 학습 시 TEST1은 2채널 RMS, TEST2는 1채널이지만 RMS 결합을 통해 양수로 변환
            combined = preprocessor.combine_channels(filtered, method='rms')

        # Normalize BEFORE creating sequences (same as training pipeline)
        # 학습 시와 동일: 시계열을 먼저 normalize한 후 sequences 생성
        print(f"DEBUG: Combined data shape before normalization: {combined.shape}")
        
        # Normalize the time series (not sequences)
        # combined shape: (n_samples, 1) -> normalize -> (n_samples, 1)
        normalized_timeseries = preprocessor.normalize(combined, fit=False)
        print(f"DEBUG: Normalized timeseries shape: {normalized_timeseries.shape}")
        
        # Create sequences AFTER normalization (same as training)
        sequences = preprocessor.create_sequences(normalized_timeseries)
        print(f"DEBUG: Created {len(sequences)} sequences")

        if len(sequences) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot create sequences. Data length: {len(normalized_timeseries)}, need at least {preprocessor.sequence_length} samples"
            )

        # 3. Detect anomalies
        threshold_val = threshold or model.threshold

        # Update model threshold temporarily if custom threshold provided
        if threshold:
            original_threshold = model.threshold
            model.threshold = threshold

        predictions, errors = model.detect_anomalies(sequences)

        # Restore original threshold
        if threshold:
            model.threshold = original_threshold

        anomaly_indices = np.where(predictions == 1)[0].tolist()
        processing_time = (time.time() - start_time) * 1000

        return FileUploadResponse(
            total_sequences=int(len(sequences)),
            anomalies_detected=int(np.sum(predictions)),
            anomaly_rate=float(np.mean(predictions)),
            anomaly_indices=anomaly_indices,
            reconstruction_errors=errors.tolist(),
            threshold=float(threshold_val),
            processing_time_ms=float(processing_time)
        )

    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/models/info", response_model=ModelInfoResponse)
async def get_model_info(manager: ModelManager = Depends(get_model_manager)):
    """
    Get loaded model information

    Returns:
        ModelInfoResponse: Model type, threshold, and performance metrics
    """
    model = manager.model

    return ModelInfoResponse(
        model_type="LSTM Autoencoder",
        loaded=model is not None,
        threshold=float(model.threshold),
        input_shape=(2048, 1),
        metrics={
            "precision": 0.93,
            "recall": 0.92,
            "f1_score": 0.925,
            "auc_roc": 0.99
        }
    )
