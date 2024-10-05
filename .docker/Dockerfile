# Use Nvidia CUDA base image optimized for size and version

# Builder stage
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel AS builder
WORKDIR /app

# Install necessary Python dependencies without caching
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --target=/app/requirements -r requirements.txt

# Runtime stage
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app

# Install only the necessary system dependencies for runtime
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

# Copy installed Python dependencies from the builder stage
COPY --from=builder /app/requirements /opt/conda/lib/python3.10/site-packages/

# Modify litserve health check endpoint
RUN sed -i 's|/health|/cocreation/health|g' /opt/conda/lib/python3.10/site-packages/litserve/server.py

# Copy the application code
COPY . /app

# Clone InstanceShadowDetection repository and install dependencies
RUN git clone https://github.com/stevewongv/InstanceShadowDetection.git /app/InstanceShadowDetection
WORKDIR /app/InstanceShadowDetection
RUN python setup.py install

# Install PythonAPI within the project
WORKDIR /app/InstanceShadowDetection/PythonAPI
RUN python setup.py install
WORKDIR /app

# Expose the application port
EXPOSE 8000

# Start the application
ENTRYPOINT ["python", "app.py"]
