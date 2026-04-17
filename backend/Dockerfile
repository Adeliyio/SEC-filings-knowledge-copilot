FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install CPU-only PyTorch first (avoids downloading ~2.5GB CUDA version)
# Increase timeout and retries for large wheel download (~190MB)
RUN pip install --no-cache-dir --timeout 300 --retries 3 torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
