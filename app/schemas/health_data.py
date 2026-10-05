from pydantic import BaseModel, Field


class SensorDataRequest(BaseModel):
    """Dữ liệu cảm biến gửi từ Web React hoặc ESP32.

    Thay vì gửi tín hiệu quang học khổng lồ, Client tự phát hiện đỉnh nhịp tim
    và chỉ gửi mảng khoảng cách thời gian giữa các nhịp (tính bằng mili-giây).
    """

    rr_intervals: list[float] = Field(..., description="Mảng R-R intervals (tính bằng mili-giây)")
    ppg_ac: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="V6 PPG autocorrelation, computed from the same raw PPG window. Required for RR-only inference with V6.",
    )
    is_workout_active: bool = Field(
        default=False,
        description="Cờ trạng thái phiên tập luyện. Nếu True, bỏ qua dự đoán AFib do nhiễu vận động.",
    )


class HRVFeatures(BaseModel):
    """13 đặc trưng được trích xuất bằng toán học (Khớp 100% với mô hình Random Forest)."""

    PPG_AC: float | None = Field(None, description="Mean absolute normalized PPG autocorrelation")
    HR_mean: float | None = Field(None, description="Nhịp tim trung bình (BPM)")
    Mean_NN: float | None = Field(None, description="Khoảng R-R trung bình (ms)")
    SDNN: float | None = Field(None, description="Độ lệch chuẩn R-R (ms)")
    RMSSD: float | None = Field(None, description="RMSSD (ms)")
    pNN50: float | None = Field(None, description="Tỷ lệ cặp R-R chênh > 50ms (%)")
    NN50: float | None = Field(None, description="Số lượng cặp R-R chênh > 50ms")
    CV: float | None = Field(None, description="Hệ số biến thiên (%)")
    LF: float | None = Field(None, description="Năng lượng phổ tần số thấp")
    HF: float | None = Field(None, description="Năng lượng phổ tần số cao")
    LF_HF_Ratio: float | None = Field(None, description="Tỷ lệ LF/HF")
    LF_norm: float | None = Field(None, description="Năng lượng phổ LF chuẩn hóa (%)")
    HF_norm: float | None = Field(None, description="Năng lượng phổ HF chuẩn hóa (%)")
    Total_Power: float | None = Field(None, description="Tổng năng lượng phổ")
    SD1: float | None = Field(None, description="Poincaré SD1 (ms)")
    SD2: float | None = Field(None, description="Poincaré SD2 (ms)")
    SampEn: float | None = Field(None, description="Sample Entropy")


class PredictionResponse(BaseModel):
    """Kết quả dự đoán trạng thái sức khỏe (Rung tâm nhĩ)."""

    prediction: str = Field(..., description="Trạng thái dự đoán (AFib hoặc Normal)")
    confidence: float = Field(..., description="Độ tin cậy (0.0 - 1.0)")
    features: HRVFeatures = Field(..., description="Các đặc trưng HRV đã tính")
    model_version: str = Field(
        default="dummy",
        description=(
            "Phiên bản model đang sử dụng, đọc từ thẻ model đi kèm (VD: 'v4.1.0', "
            "lấy từ app/models/v4.json). Nếu model không có thẻ thì chỉ hiện tên file."
        ),
        examples=["V6_RESEARCH_FREEZE_2026_09"],
    )


class HealthCheckResponse(BaseModel):
    """Trạng thái hoạt động của server."""

    status: str = Field(default="ok")
    model_loaded: bool = Field(default=False, description="Model đã được load chưa")
    model_version: str = Field(default="none")
    active_model_file: str | None = Field(default=None, description="Tên file model đang kích hoạt")
    feature_count: int | None = Field(default=None, description="Số lượng đặc trưng model yêu cầu")


class ModelInfo(BaseModel):
    """Thông tin chi tiết về một file model."""

    filename: str = Field(..., description="Tên file model")
    is_active: bool = Field(..., description="Model đang được sử dụng hay không")
    size_kb: float = Field(..., description="Kích thước file (KB)")
    feature_count: int | None = Field(None, description="Số lượng đặc trưng yêu cầu")
    expected_features: list[str] | None = Field(None, description="Danh sách tên các đặc trưng")
    model_type: str | None = Field(
        None, description="Kiểu class của model (Pipeline, MLPClassifier,...)"
    )


class ModelListResponse(BaseModel):
    """Danh sách các model có sẵn trong thư mục app/models."""

    active_model: str | None = Field(None, description="Tên file model đang kích hoạt")
    available_models: list[ModelInfo] = Field(default_factory=list)


class SelectModelRequest(BaseModel):
    """Select a validated V6 stacking bundle or legacy preprocessing pipeline."""

    model_file: str = Field(
        ...,
        description="File name inside app/models, e.g. v6_stacking.pkl (default) or v4.pkl.",
        examples=["v6_stacking.pkl", "v4.pkl"],
    )
