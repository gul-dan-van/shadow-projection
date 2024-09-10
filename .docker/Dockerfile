# Build stage
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_VERSION=3.10 \
    PYTHONPATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages \
    PATH=/usr/bin:/usr/local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    && sed -i 's|/health|/cocreation/health|g' /usr/local/lib/python3.10/dist-packages/litserve/server.py

# Final stage
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY . .

ENTRYPOINT ["python", "app.py"]