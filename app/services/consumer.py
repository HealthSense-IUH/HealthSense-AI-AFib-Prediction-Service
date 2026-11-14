import asyncio
import hashlib
import hmac
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import aio_pika
import httpx
import numpy as np
import pandas as pd

from app.config import settings
from app.services.feature_engineering import extract_features_from_csv_data
from app.services.prediction import prediction_service
from app.services.recording import prepare_recording
from app.services.s3_client import s3_client
from app.services.windowing import analyse_recording

logger = logging.getLogger("uvicorn.error")

CALLBACK_RETRY_DELAYS = (1.0, 3.0, 7.0)


def _redact_broker_url(url: str) -> str:
    """Che thông tin xác thực trong RABBITMQ_URL trước khi ghi log."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or "unknown"
        port = f":{parsed.port}" if parsed.port else ""
        path = parsed.path or ""
        return f"{parsed.scheme}://{host}{port}{path}"
    except Exception:
        return "<redacted>"


def _build_callback_headers(record_id: int, status_or_label: str) -> dict[str, str]:
    """Tạo chữ ký HMAC-SHA256 cho callback gửi về Core Service (nếu cấu hình secret)."""
    timestamp = str(int(time.time()))
    headers: dict[str, str] = {"X-AI-Timestamp": timestamp}
    secret = (settings.AI_CALLBACK_SECRET or "").strip()
    if secret:
        canonical = f"{timestamp}.{record_id}.{status_or_label}"
        signature = hmac.new(
            secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        headers["X-AI-Signature"] = signature
    return headers


def _send_callback_with_retry(
    url: str, payload: dict, record_id: int, status_or_label: str
) -> bool:
    """Gửi PATCH callback về Core Service kèm retry backoff khi lỗi mạng hoặc HTTP 5xx."""
    if not url or not url.strip():
        logger.warning(f"[Record {record_id}] CORE_CALLBACK_URL not configured, skipping callback.")
        return False
    max_attempts = len(CALLBACK_RETRY_DELAYS) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            headers = _build_callback_headers(record_id, status_or_label)
            with httpx.Client(timeout=10.0) as client:
                response = client.patch(url, json=payload, headers=headers)
            status_code = getattr(response, "status_code", 200)
            if not isinstance(status_code, int) or status_code in (200, 204):
                return True
            if status_code < 500 and status_code != 429:
                logger.error(
                    f"[Record {record_id}] Callback rejected with client status "
                    f"{status_code}: {getattr(response, 'text', '')}"
                )
                return False
            logger.warning(
                f"[Record {record_id}] Callback attempt {attempt}/{max_attempts} failed "
                f"with status {status_code}"
            )
        except Exception as err:
            logger.warning(
                f"[Record {record_id}] Callback attempt {attempt}/{max_attempts} error: {err}"
            )
        if attempt < max_attempts:
            time.sleep(CALLBACK_RETRY_DELAYS[attempt - 1])
    return False


class RabbitMQConsumer:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.queue = None
        self._is_running = False
        max_workers = max(1, int(getattr(settings, "AI_MAX_WORKERS", 2) or 2))
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None

    async def start(self):
        """Kết nối RabbitMQ và bắt đầu lắng nghe hàng đợi."""
        if not settings.RABBITMQ_URL:
            logger.info("RabbitMQ URL not configured. Consumer will not start.")
            return

        try:
            safe_url = _redact_broker_url(settings.RABBITMQ_URL)
            logger.info(
                f"Connecting to RabbitMQ at {safe_url} (prefetch={self._max_workers})..."
            )
            self.connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=self._max_workers)

            # Khai báo queue (durable=True để khớp với cấu hình Queue(..., true) bên Spring Boot)
            self.queue = await self.channel.declare_queue(settings.RABBITMQ_QUEUE, durable=True)

            self._is_running = True
            logger.info(f"Connected to RabbitMQ! Listening on queue '{settings.RABBITMQ_QUEUE}'...")
            await self.queue.consume(self.on_message, no_ack=False)
        except Exception as e:
            logger.error(f"Failed to start RabbitMQ consumer: {str(e)}")
            # Không raise để ứng dụng vẫn khởi động được khi local chưa mở RabbitMQ

    async def stop(self):
        """Đóng kết nối RabbitMQ khi ứng dụng tắt."""
        self._is_running = False
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("RabbitMQ connection closed.")
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="ai-infer"
            )
        return self._executor

    async def on_message(self, message: aio_pika.abc.AbstractIncomingMessage):
        """Xử lý mỗi khi có tin nhắn mới từ hàng đợi."""
        async with message.process():
            try:
                payload_str = message.body.decode("utf-8")
                data = json.loads(payload_str)

                record_id = data.get("recordId")
                s3_key = data.get("s3Key")
                user_id = data.get("userId")
                logger.info(f"Received RabbitMQ message for recordId={record_id}, userId={user_id}")

                if not record_id or not s3_key:
                    logger.error(
                        "Message missing required fields 'recordId' or 's3Key'. Discarding."
                    )
                    return

                # Chạy tải S3 và tính toán trong executor giới hạn luồng để ổn định RAM/CPU
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._get_executor(), self._process_record_sync, record_id, s3_key, user_id
                )

            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)

    def _process_record_sync(self, record_id: int, s3_key: str, user_id: int):
        """Hàm đồng bộ chạy trong thread pool: tải S3, dự đoán AFib, gửi callback."""
        try:
            # 1. Tải file CSV từ S3
            logger.info(f"[Record {record_id}] Downloading CSV from S3 key: {s3_key}")
            content = s3_client.download_file_as_bytes(s3_key)

            # 2. Phân tích CSV
            df = pd.read_csv(io.BytesIO(content))
            time_col = "Time(ms)" if "Time(ms)" in df.columns else "time"
            ppg_col = "IR" if "IR" in df.columns else "ppg"

            if time_col not in df.columns or ppg_col not in df.columns:
                raise ValueError(f"File CSV thiếu cột '{time_col}' hoặc '{ppg_col}'")

            # Đổi đơn vị thời gian, ước lượng fs và bỏ 2 giây đầu (dùng chung với endpoint HTTP)
            time_ms_array, ppg_array, fs = prepare_recording(
                df[time_col].to_numpy(dtype=float), df[ppg_col].to_numpy(dtype=float)
            )

            logger.info(
                f"[Record {record_id}] Extracting HRV features (fs={fs:.2f}Hz, samples after 2s trim={len(ppg_array)})..."
            )
            features = extract_features_from_csv_data(time_ms_array, ppg_array, fs=fs)

            # SpO2 / BPM do firmware tính sẵn (cột trong CSV thiết bị).
            # Vài giây đầu firmware chưa kịp tính nên ghi 0 -> chỉ lấy giá trị > 0.
            # Lưu ý: SpO2 từ firmware cũ chưa hiệu chuẩn — hiển thị mức "tham khảo".
            for col, key in [("SpO2", "deviceSpO2"), ("Bpm", "deviceBpm")]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals) & (vals > 0)]
                    if len(vals) >= 5:
                        features[key] = round(float(np.median(vals)), 1)

            # --- TẠO DỮ LIỆU ĐỒ THỊ BẰNG PYTHON ---
            try:
                # pyrefly: ignore[untyped-import]
                import scipy.signal

                from app.services.feature_engineering import butter_bandpass_filter

                # 1. Nghịch đảo (Inversion)
                y_inverted = -1.0 * ppg_array

                # 2. Lọc nhiễu (Butterworth Bandpass Filter)
                # Dùng dải 0.5Hz đến 5.0Hz để làm phẳng dốc hô hấp và loại bỏ nhiễu điện
                y_clean = butter_bandpass_filter(
                    y_inverted, lowcut=0.5, highcut=5.0, fs=fs, order=3
                )

                # 3. Giảm mẫu nội suy (Downsampling bằng Fourier) xuống 300 điểm
                target_points = 300
                if len(y_clean) > target_points:
                    y_resampled = scipy.signal.resample(y_clean, target_points)
                else:
                    y_resampled = y_clean

                # 4. Chuẩn hóa biên độ (Min-Max Scaling về khung [0, 100])
                y_min = np.min(y_resampled)
                y_max = np.max(y_resampled)
                if y_max > y_min:
                    y_norm = 100.0 * (y_resampled - y_min) / (y_max - y_min)
                else:
                    y_norm = np.zeros_like(y_resampled)

                # Làm tròn 2 chữ số thập phân cho nhẹ JSON (300 điểm ~ 3KB)
                features["chartData"] = np.round(y_norm, 2).tolist()
                logger.info(
                    f"[Record {record_id}] Generated chartData ({len(features['chartData'])} points)"
                )
            except Exception as chart_err:
                logger.error(f"[Record {record_id}] Failed to generate chartData: {str(chart_err)}")
                features["chartData"] = []
            # --------------------------------------

            # 3. Dự đoán theo cửa sổ 30 s / bước 10 s và quy tắc temporal persistence.
            # confidence = trung bình P_meta của các cửa sổ qua SQI; chi tiết từng cửa sổ và cờ
            # cảnh báo đi kèm trong hrvFeaturesJson nên hợp đồng callback với Core giữ nguyên.
            logger.info(f"[Record {record_id}] Running window-level AFib prediction...")
            analysis = analyse_recording(time_ms_array, ppg_array, fs, prediction_service)
            afib_probability = round(analysis["meanProbability"], 4)
            features["windowAnalysis"] = analysis
            features["afAlert"] = analysis["alert"]
            logger.info(
                f"[Record {record_id}] Windows accepted={analysis['acceptedWindows']} "
                f"rejected={analysis['rejectedWindows']} alert={analysis['alert']} "
                f"branch={analysis['alertBranch']}"
            )

            if afib_probability < 0.30:
                callback_label = "NORMAL"
            elif afib_probability < 0.50:
                callback_label = "UNCERTAIN"
            elif afib_probability < 0.70:
                callback_label = "AFIB_SUSPECTED"
            else:
                callback_label = "AFIB"

            logger.info(
                f"[Record {record_id}] Prediction done: prob={afib_probability:.4f}, label={callback_label}"
            )

            # 4. Gửi HTTP PATCH callback về Core Service (kèm retry & HMAC signature)
            callback_payload = {
                "recordId": record_id,
                "predictionLabel": callback_label,
                "confidence": round(afib_probability, 4),
                "hrvFeaturesJson": json.dumps(features),
                "afAlert": bool(analysis["alert"]),
                "modelVersion": settings.MODEL_FILE,
            }

            logger.info(
                f"[Record {record_id}] Sending callback to Core Service: {settings.CORE_CALLBACK_URL}"
            )
            if _send_callback_with_retry(
                settings.CORE_CALLBACK_URL, callback_payload, record_id, callback_label
            ):
                logger.info(f"[Record {record_id}] Callback successful! Core Service updated.")
            else:
                logger.error(f"[Record {record_id}] Callback exhausted retries.")

        except Exception as e:
            logger.error(
                f"[Record {record_id}] Failed during processing or callback: {str(e)}",
                exc_info=True,
            )
            fail_url = f"{settings.CORE_CALLBACK_URL}/fail"
            fail_payload = {"recordId": record_id, "errorReason": str(e)}
            logger.info(f"[Record {record_id}] Sending fail callback to {fail_url}")
            if _send_callback_with_retry(fail_url, fail_payload, record_id, "FAILED"):
                logger.info(f"[Record {record_id}] FAIL callback successful.")
            else:
                logger.error(f"[Record {record_id}] FAIL callback exhausted retries.")


consumer = RabbitMQConsumer()
