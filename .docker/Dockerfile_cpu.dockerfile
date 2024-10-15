# Use a base image without GPU support
FROM python:3.10-slim

# Configure the time zone and install system dependencies in a single layer to reduce image size
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/usr/local/lib/python3.10/site-packages \
    PATH=/usr/bin:/usr/local/bin:$PATH

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    wget\
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# Set the working directory
WORKDIR /app

# Install Python dependencies in a single layer, ensure no cache is stored to minimize image size
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code to the working directory
COPY . .
# Set the default command to run when the container starts
ENTRYPOINT ["python", "app.py"]
