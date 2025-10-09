FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install apt packages (including build-essential for compiled wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    ca-certificates \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Upgrade pip and install requirements
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY . /app

EXPOSE 9000

CMD ["uvicorn", "app_triton_http:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
