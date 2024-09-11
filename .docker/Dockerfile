# Use Nvidia CUDA base image optimized for size and version
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel as builder
WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --target=/app/requirements -r requirements.txt

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/requirements /opt/conda/lib/python3.10/site-packages/
RUN sed -i 's|/health|/cocreation/health|g' /opt/conda/lib/python3.10/site-packages/litserve/server.py
COPY . /app
RUN adduser -m server
USER server
RUN chmod -R 755 /app
EXPOSE 8000
ENTRYPOINT ["python", "app.py"]
