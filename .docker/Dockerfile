# Use Nvidia CUDA base image optimized for size and version

# Builder stage
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel as builder
WORKDIR /app

# Install necessary dependencies from requirements.txt
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --target=/app/requirements -r requirements.txt

# Install necessary system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Runtime stage
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
WORKDIR /app

# Install necessary system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from the builder stage
COPY --from=builder /app/requirements /opt/conda/lib/python3.10/site-packages/

# Uninstall psutil if necessary
RUN pip3 uninstall psutil -y

# Modify litserve health check endpoint
RUN sed -i 's|/health|/cocreation/health|g' /opt/conda/lib/python3.10/site-packages/litserve/server.py

# Copy application code
COPY . /app


# Clone InstanceShadowDetection repository and install pysobatools
RUN git clone https://github.com/stevewongv/InstanceShadowDetection.git /app/InstanceShadowDetection
WORKDIR /app/InstanceShadowDetection
RUN python setup.py install

# Install PythonAPI
WORKDIR /app/InstanceShadowDetection/PythonAPI
RUN python setup.py install
WORKDIR /app

# Expose the application port
EXPOSE 8000

# Start the application
ENTRYPOINT ["python", "app.py"]
