"""HealthSense AI AFib Prediction Service - FastAPI Application.

Microservice chuyên phục vụ ML inference cho dự án HealthSense.
Nhận dữ liệu cảm biến PPG, trích xuất đặc trưng HRV, và trả về dự đoán trạng thái sức khỏe.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import prediction
from app.services.consumer import consumer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Khởi động RabbitMQ consumer
    await consumer.start()
    yield
    # Shutdown: Đóng kết nối RabbitMQ
    await consumer.stop()


app = FastAPI(
    title="HealthSense AI AFib Prediction Service",
    description=(
        "PPG-based suspected-AF screening using V6C stacking "
        "(ExtraTrees, RandomForest, XGBoost and a scaled logistic meta model). "
        "The model uses 14 features including PPG autocorrelation."
    ),
    version="0.4.0-v6-model",
    lifespan=lifespan,
)

# CORS - cho phép Spring Boot Backend và ESP32 gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký Router (tương tự @ComponentScan trong Spring Boot)
app.include_router(prediction.router)
app.include_router(prediction.models_router)


@app.get("/", tags=["Root"])
async def root():
    """Trang chủ API."""
    return {
        "service": "HealthSense AI AFib Prediction Service",
        "version": "0.4.0-v6-model",
        "docs": "/docs",
    }
