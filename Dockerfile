FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /app/models && \
    wget -q -O /app/models/pose_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

COPY gait_detect.py /app/gait_detect.py

RUN mkdir -p /files /output

ENV POSE_MODEL_PATH=/app/models/pose_landmarker.task

ENTRYPOINT ["python3", "gait_detect.py"]
