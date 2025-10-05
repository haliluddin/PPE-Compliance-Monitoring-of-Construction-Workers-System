FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
      python3-dev \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      ffmpeg \
      ca-certificates \
      wget && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    apt-get remove -y --purge build-essential gcc g++ python3-dev wget && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY /app /app

CMD ["uvicorn", "app_triton_http:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]