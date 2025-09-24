#!/bin/bash

# Script to download all necessary model weights for igait-human-gait-detection
# This script will download the weights to the host machine for mounting into Docker

# Create base directories
echo "Creating directories for model weights..."
mkdir -p model_weights/torch/hub/checkpoints
mkdir -p model_weights/deep_sort/deep_sort/deep/checkpoint
mkdir -p data

echo "Downloading YOLOv5 weights..."
wget -O model_weights/torch/hub/checkpoints/yolov5l6.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt

echo "Downloading SlowFast model checkpoint..."
wget -O model_weights/torch/hub/checkpoints/SLOWFAST_8x8_R50_DETECTION.pyth https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ava/SLOWFAST_8x8_R50_DETECTION.pyth

# If DeepSort parameters exist locally, copy them to the model_weights directory
if [ -d "deepsort_parameters" ]; then
    echo "Copying DeepSort parameters..."
    cp -r deepsort_parameters/* model_weights/deep_sort/deep_sort/deep/checkpoint/
else
    echo "Warning: deepsort_parameters directory not found. You'll need to manually copy these files."
    echo "The DeepSort parameters should be placed in: model_weights/deep_sort/deep_sort/deep/checkpoint/"
    # Create a placeholder file so the directory exists
    touch model_weights/deep_sort/deep_sort/deep/checkpoint/.placeholder
fi

# Check downloaded files
echo -e "\nVerifying downloads:"

if [ -f "model_weights/torch/hub/checkpoints/yolov5l6.pt" ]; then
    size=$(ls -lh model_weights/torch/hub/checkpoints/yolov5l6.pt | awk '{print $5}')
    echo "✓ YOLOv5 weights downloaded: ${size}"
else
    echo "✗ YOLOv5 weights download failed"
fi

if [ -f "model_weights/torch/hub/checkpoints/SLOWFAST_8x8_R50_DETECTION.pyth" ]; then
    size=$(ls -lh model_weights/torch/hub/checkpoints/SLOWFAST_8x8_R50_DETECTION.pyth | awk '{print $5}')
    echo "✓ SlowFast weights downloaded: ${size}"
else
    echo "✗ SlowFast weights download failed"
fi

echo -e "\nSetup complete! You can now build and run the Docker container with:"
echo "  docker build -t igait-human-gait ."
echo "  docker run -it --gpus all \\"
echo "    -v $(pwd)/model_weights/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \\"
echo "    -v $(pwd)/model_weights/deep_sort/deep_sort/deep/checkpoint:/app/deep_sort/deep_sort/deep/checkpoint \\"
echo "    -v $(pwd)/data:/files \\"
echo "    igait-human-gait"