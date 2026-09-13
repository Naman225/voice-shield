# ── Voice Shield 🛡️ Production Dockerfile (Memory-Optimized for Cloud Free Tiers) ──
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_TRIM_THRESHOLD_=100000

# Install system audio libraries required by soundfile & scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install lightweight CPU-optimized PyTorch (~120MB RAM vs 500MB+ for CUDA)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining application requirements (without pulling heavy CUDA torch)
COPY requirements.txt .
RUN grep -v -E "^(torch|torchaudio)" requirements.txt > req_light.txt && \
    pip install --no-cache-dir -r req_light.txt

# Step 3: Copy application source, models, checkpoints, and static UI
COPY . .

# Expose default port
EXPOSE 8000

# Run FastAPI server (Render sets $PORT dynamically, single worker for minimal memory)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
