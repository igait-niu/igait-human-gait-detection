FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set the working directory
WORKDIR /app

# Update package lists and install required dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-venv \
    python3-pip \
    python3-dev \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Ensure latest pip + setuptools
RUN pip install --upgrade pip setuptools wheel

# Clone your repo
RUN git clone https://github.com/wufan-tb/yolo_slowfast.git . 
COPY yolo_slowfast.py /app/yolo_slowfast.py

# Install PyTorch (CUDA 11.3 build)
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install required Python packages
RUN pip install \
        seaborn \
        pytorchvideo \
        scipy \
        pandas \
        numpy \
        requests \
        ultralytics \
        gitpython>=3.1.30 \
        setuptools>=70.0.0 \
        matplotlib>=3.3 \
        opencv-python>=4.1.1 \
        pillow>=10.3.0 \
        psutil \
        PyYAML>=5.3.1 \
        thop>=0.1.1 \
        tqdm>=4.66.3

# Pre-download YOLOv5 weights (l6 model)
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -O /root/.cache/torch/hub/checkpoints/yolov5l6.pt \
    https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt
RUN cp /root/.cache/torch/hub/checkpoints/yolov5l6.pt /app/yolov5l6.pt

# Pre-download SlowFast model checkpoint
RUN wget -O /root/.cache/torch/hub/checkpoints/SLOWFAST_8x8_R50_DETECTION.pyth \
    https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ava/SLOWFAST_8x8_R50_DETECTION.pyth

# Create directory for DeepSort checkpoint
RUN mkdir -p deep_sort/deep_sort/deep/checkpoint/
RUN mkdir -p /files

# Copy deep sort parameters
COPY ./deepsort_parameters deep_sort/deep_sort/deep/checkpoint/
COPY ./data /files

# Set environment variables for CUDA + Ultralytics
ENV YOLOV5_CACHE=/root/.cache/torch/hub/checkpoints
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV YOLO_CONFIG_DIR=/root/.config/Ultralytics
ENV TORCH_HOME=/root/.cache/torch

# Default entrypoint
CMD ["python3", "yolo_slowfast.py"]