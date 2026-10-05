"""Router cho API dự đoán rung tâm nhĩ (AFib)."""

import io

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.health_data import (
    HealthCheckResponse,
    HRVFeatures,
    ModelListResponse,
    PredictionResponse,
    SelectModelRequest,
    SensorDataRequest,
)
from app.services.feature_engineering import (
    PoorSignalQualityError,
    extract_features_from_csv_data,
    extract_hrv_features,
)
from app.services.prediction import prediction_service

router = APIRouter(prefix="/api", tags=["Prediction"])

# Router riêng cho quản lý model.
#
# FastAPI CỘNG DỒN tag của route vào tag của router chứ không ghi đè. Trước đây
# hai route /models và /models/active nằm chung `router` (tag "Prediction") rồi
# tự khai thêm tags=["Model Management"], nên chúng mang CẢ HAI tag và Swagger
# vẽ chúng ở cả hai mục — nhìn như service có endpoint trùng lặp. Tách thành
# router riêng là cách duy nhất cho mỗi route đúng một tag.
models_router = APIRouter(prefix="/api", tags=["Model Management"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Kiểm tra trạng thái hoạt động của AI Service và model hiện tại."""
    if not prediction_service.is_model_loaded:
        raise HTTPException(status_code=503, detail="Configured AI model is not loaded.")
    return HealthCheckResponse(
        status="ok",
        model_loaded=prediction_service.is_model_loaded,
        model_version=prediction_service.model_version,
        active_model_file=prediction_service.active_model_file,
        feature_count=len(prediction_service.expected_features),
    )


@models_router.get("/models", response_model=ModelListResponse)
async def list_models():
    """Liệt kê danh sách tất cả các mô hình có sẵn trong thư mục app/models."""
    models = prediction_service.list_available_models()
    return ModelListResponse(
        active_model=prediction_service.active_model_file,
        available_models=models,
    )


@models_router.post("/models/active", response_model=HealthCheckResponse)
async def switch_model(request: SelectModelRequest):
    """Chuyển đổi nóng mô hình AI đang chạy mà không cần restart server."""
    success = prediction_service.load_model(request.model_file)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Không thể nạp mô hình '{request.model_file}'. Vui lòng kiểm tra file tồn tại trong thư mục app/models.",
        )
    return await health_check()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: SensorDataRequest):
    if request.is_workout_active:
        return PredictionResponse(
            prediction="MUTED_DURING_WORKOUT",
            confidence=0.0,
            features=HRVFeatures(),
            model_version="muted",
        )

    if len(request.rr_intervals) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Dữ liệu quá ngắn ({len(request.rr_intervals)} nhịp). Cần tối thiểu 10 nhịp tim để phân tích.",
        )

    try:
        # Lọc sinh lý + SQI nằm trong extract_hrv_features (pipeline v4)
        features = extract_hrv_features(request.rr_intervals)
        if request.ppg_ac is not None:
            features["PPG_AC"] = request.ppg_ac
        prediction_label, confidence = prediction_service.predict(features)

        return PredictionResponse(
            prediction=prediction_label,
            confidence=round(confidence, 4),
            features=HRVFeatures(**features),
            model_version=prediction_service.model_version,
        )

    except HTTPException:
        raise
    except PoorSignalQualityError as pe:
        raise HTTPException(status_code=422, detail=str(pe)) from pe
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý dữ liệu: {str(e)}",
        ) from e


@router.post("/predict-csv", response_model=PredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Vui lòng tải lên file định dạng CSV.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Hỗ trợ cả Time(ms) và time, IR hoặc ppg
        time_col = "Time(ms)" if "Time(ms)" in df.columns else "time"
        ppg_col = "IR" if "IR" in df.columns else "ppg"

        if time_col not in df.columns or ppg_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"File CSV phải có cột thời gian và tín hiệu (VD: '{time_col}' và '{ppg_col}')",
            )

        time_ms_array = np.asarray(df[time_col].values, dtype=float)
        ppg_array = np.asarray(df[ppg_col].values, dtype=float)

        # Nếu cột time nhỏ (ví dụ tính bằng giây), chuyển sang ms
        if np.max(time_ms_array) < 100000 and np.mean(np.diff(time_ms_array)) < 1.0:
            time_ms_array = time_ms_array * 1000.0

        # Tính tần số lấy mẫu (fs) tự động
        fs = 125.0  # Default cho MIMIC
        if len(time_ms_array) >= 2:
            dt = np.mean(np.diff(time_ms_array[: min(100, len(time_ms_array))])) / 1000.0
            if dt > 0:
                fs = 1.0 / dt

        # Trích xuất đặc trưng
        features = extract_features_from_csv_data(time_ms_array, ppg_array, fs=fs)

        # Gọi model
        prediction_label, confidence = prediction_service.predict(features)

        return PredictionResponse(
            prediction=prediction_label,
            confidence=round(confidence, 4),
            features=HRVFeatures(**features),
            model_version=prediction_service.model_version,
        )
    except PoorSignalQualityError as pe:
        raise HTTPException(status_code=422, detail=str(pe)) from pe
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý dữ liệu: {str(e)}") from e
