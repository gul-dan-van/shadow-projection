# Use Nvidia CUDA base image optimized for size and version
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --target=/app/requirements -r requirements.txt

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY --from=builder /app/requirements /opt/conda/lib/python3.10/site-packages/
RUN sed -i 's|/health|/cocreation/health|g'/opt/conda/lib/python3.10/site-packages/litserve/server.py
COPY . /app
EXPOSE 8000
RUN useradd -m server
USER server
ENTRYPOINT ["python", "app.py"]
