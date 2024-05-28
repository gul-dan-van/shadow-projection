# Use Nvidia CUDA base image with latest version
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set default Python version to Python 3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set environment variable to prevent Python from writing bytecode (.pyc files)
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install albumentations einops kornia opencv-python pytorch_msssim torch==1.12.0+cu116 torchvision==0.13.0+cu116

# Copy the rest of the application code to the working directory
COPY . .

# Set the default command to run when the container starts
CMD ["python", "app.py"]
