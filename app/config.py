"""Cấu hình ứng dụng HealthSense AI AFib Prediction Service."""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Config
    APP_NAME: str = os.getenv("APP_NAME", "HealthSense AI AFib Prediction Service")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    # RabbitMQ Config
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "")
    RABBITMQ_QUEUE: str = os.getenv("RABBITMQ_QUEUE", "health.record.processing.queue")
    AI_MAX_WORKERS: int = int(os.getenv("AI_MAX_WORKERS", "2"))

    # AWS S3 / MinIO Config (Không hardcode credentials trên mã nguồn để đảm bảo bảo mật tuyệt đối)
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "")
    AWS_S3_REGION: str = os.getenv("AWS_S3_REGION", "us-east-1")
    AWS_S3_ENDPOINT_URL: str = os.getenv("AWS_S3_ENDPOINT_URL", "")

    # V6C stacking bundle and matching JSON card, downloaded before Docker build.
    MODEL_FILE: str = os.getenv("MODEL_FILE", "v6_stacking.pkl")

    # Core Service Webhook URL & Secret
    CORE_CALLBACK_URL: str = os.getenv("CORE_CALLBACK_URL", "")
    AI_CALLBACK_SECRET: str = os.getenv("AI_CALLBACK_SECRET", "")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
