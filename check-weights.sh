#!/bin/bash

YOLO_WEIGHT="/root/.cache/torch/hub/checkpoints/yolov5l6.pt"
SLOWFAST_WEIGHT="/root/.cache/torch/hub/checkpoints/SLOWFAST_8x8_R50_DETECTION.pyth"
DEEPSORT_DIR="/app/deep_sort/deep_sort/deep/checkpoint"

# Check if required model weights exist
missing_files=0

echo "Checking for required model files..."

if [ ! -f "$YOLO_WEIGHT" ]; then
  echo "ERROR: YOLOv5 weights not found at $YOLO_WEIGHT"
  echo "Please mount the YOLOv5 weights file into the container."
  missing_files=1
fi

if [ ! -f "$SLOWFAST_WEIGHT" ]; then
  echo "ERROR: SlowFast weights not found at $SLOWFAST_WEIGHT"
  echo "Please mount the SlowFast weights file into the container."
  missing_files=1
fi

if [ ! "$(ls -A $DEEPSORT_DIR)" ]; then
  echo "ERROR: DeepSort checkpoint files not found in $DEEPSORT_DIR"
  echo "Please mount the DeepSort checkpoint files into the container."
  missing_files=1
fi

if [ $missing_files -ne 0 ]; then
  echo ""
  echo "Missing required model files. Please run with proper volume mounts:"
  echo "docker run -it --gpus all \\"
  echo "  -v /path/to/yolov5l6.pt:/root/.cache/torch/hub/checkpoints/yolov5l6.pt \\"
  echo "  -v /path/to/SLOWFAST_8x8_R50_DETECTION.pyth:/root/.cache/torch/hub/checkpoints/SLOWFAST_8x8_R50_DETECTION.pyth \\"
  echo "  -v /path/to/deepsort_checkpoints/:/app/deep_sort/deep_sort/deep/checkpoint/ \\"
  echo "  -v /path/to/data:/files \\"
  echo "  your_image_name"
  exit 1
fi

echo "All required model files found. Starting application..."
exec "$@"