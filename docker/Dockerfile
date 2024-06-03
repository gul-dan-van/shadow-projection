# Use Nvidia CUDA base image optimized for size and version
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

# Configure the time zone and install system dependencies in a single layer to reduce image size
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_VERSION=3.10 \
    PYTHONPATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages \
    PATH=/usr/bin:/usr/local/bin:$PATH

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3.10 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python dependencies in a single layer, ensure no cache is stored to minimize image size
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir -r /app/requirements.txt
# Copy the rest of the application code to the working directory
COPY . .
# Set the default command to run when the container starts
ENTRYPOINT ["python", "app.py"]