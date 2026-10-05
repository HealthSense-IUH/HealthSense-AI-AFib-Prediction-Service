# ========================
# Stage 1: Dependencies
# ========================
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ========================
# Stage 2: Production
# ========================
FROM python:3.12-slim

# Tạo user non-root (bảo mật: không chạy container bằng root)
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy dependencies từ builder stage
COPY --from=builder /install /usr/local

# Copy source code
COPY ./app ./app

ENV MODEL_FILE=v6_stacking.pkl

# Refuse to publish an image if the configured model cannot load or predict.
RUN python -m app.services.verify_model

# Tạo thư mục models và gán quyền cho appuser
RUN mkdir -p ./app/models && chown -R appuser:appuser ./app

# Chuyển sang user non-root
USER appuser

# Port mặc định
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Chạy server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
