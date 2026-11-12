"""Bỏ 2 giây đầu và ước lượng fs (app/services/recording.py), dùng chung cho consumer và endpoint."""

import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from app.services.hrv_v4 import PoorSignalQualityError
from app.services.recording import WARMUP_MS, prepare_recording
from app.services.windowing import window_slices

FS = 100.0
SECONDS = 72.0  # thời lượng đo của ứng dụng di động


def device_csv(seconds: float = SECONDS, fs: float = FS) -> bytes:
    t_ms = np.arange(int(seconds * fs)) * 1000.0 / fs
    ppg = np.sin(2 * np.pi * 1.2 * t_ms / 1000.0)
    frame = pd.DataFrame({"Time(ms)": t_ms, "IR": ppg})
    return frame.to_csv(index=False).encode()


class PrepareRecordingTest(unittest.TestCase):
    def test_72_s_recording_gives_70_s_and_five_windows(self):
        t_ms = np.arange(int(SECONDS * FS)) * 1000.0 / FS
        time_ms, ppg, fs = prepare_recording(t_ms, np.zeros_like(t_ms))
        self.assertAlmostEqual(fs, FS)
        self.assertEqual(time_ms[0], WARMUP_MS)
        self.assertEqual(len(ppg), int((SECONDS - 2) * FS))
        self.assertEqual(len(window_slices(len(ppg), fs)), 5)

    def test_time_in_seconds_is_converted_to_ms(self):
        t_s = np.arange(int(10 * FS)) / FS
        time_ms, _, fs = prepare_recording(t_s, np.zeros_like(t_s))
        self.assertAlmostEqual(fs, FS)
        self.assertEqual(time_ms[0], WARMUP_MS)

    def test_broken_time_column_falls_back_to_sample_count(self):
        n = 1000
        time_ms, ppg, fs = prepare_recording(np.zeros(n), np.arange(n, dtype=float))
        self.assertEqual(fs, 125.0)  # không ước lượng được fs
        self.assertEqual(len(ppg), n - 250)  # bỏ 2 s x 125 Hz mẫu
        self.assertEqual(ppg[0], 250.0)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_recording(np.arange(10.0), np.arange(9.0))


class SameInputForBothEntryPointsTest(unittest.TestCase):
    """Endpoint /api/predict-csv và consumer RabbitMQ phải đưa cùng một tín hiệu vào phân tích."""

    def capture(self):
        captured = {}

        def fake_extract(time_ms, ppg, fs):
            captured.update(time_ms=np.asarray(time_ms), ppg=np.asarray(ppg), fs=fs)
            raise PoorSignalQualityError("stop after capture")

        return captured, fake_extract

    def test_endpoint_and_consumer_trim_identically(self):
        from fastapi.testclient import TestClient

        from app.main import app
        from app.services.consumer import consumer

        content = device_csv()

        endpoint, fake = self.capture()
        with patch("app.routers.prediction.extract_features_from_csv_data", side_effect=fake):
            response = TestClient(app).post(
                "/api/predict-csv", files={"file": ("rec.csv", io.BytesIO(content), "text/csv")}
            )
        self.assertEqual(response.status_code, 422)

        queue, fake = self.capture()
        with (
            patch("app.services.consumer.s3_client.download_file_as_bytes", return_value=content),
            patch("app.services.consumer.extract_features_from_csv_data", side_effect=fake),
            patch("app.services.consumer.httpx.Client", MagicMock()),
        ):
            consumer._process_record_sync(1, "records/rec.csv", 1)

        for captured in (endpoint, queue):
            self.assertEqual(captured["time_ms"][0], WARMUP_MS)
            self.assertEqual(len(captured["ppg"]), int((SECONDS - 2) * FS))
        np.testing.assert_array_equal(endpoint["ppg"], queue["ppg"])
        self.assertAlmostEqual(endpoint["fs"], queue["fs"])


if __name__ == "__main__":
    unittest.main()
