# igait-human-gait-detection

<img src="./person_walking.gif"/>

Detection service for determining whether a video contains **exactly one person walking**. This is a YOLO + SlowFast + DeepSORT pipeline that detects people in a video, tracks them, recognizes their actions, and outputs a processed video while also checking if exactly one person is walking.

**Return values:**

* `0` → Video contains **one person walking**
* `1` → Video does **not** contain exactly one person

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/michaelslice/igait-human-gait-detection.git
cd igait-human-gait-detection
```

### 2. Download Model Weights

Use the provided script to automatically download all required model weights:

```bash
# Make script executable if needed
chmod +x download_weights.sh

# Run the download script
./download_weights.sh
```

This script will:
- Create the necessary directory structure
- Download YOLOv5 and SlowFast model weights
- Copy DeepSort parameters if available
- Verify successful downloads

### 3. Prepare Input Folder

Copy your input video files to the data folder that was created by the download script:

```bash
cp <PATH_TO_YOUR_VIDEOS>/* data/
```

### 4. Build Docker Image

```bash
docker build -t igait-human-gait -f Dockerfile .
```

---

## Running YOLO + SlowFast Pipeline

```bash
docker run -it --gpus all \
  -v $(pwd)/model_weights/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \
  -v $(pwd)/model_weights/deep_sort/deep_sort/deep/checkpoint:/app/deep_sort/deep_sort/deep/checkpoint \
  -v $(pwd)/data:/files \
  -v $(pwd)/output:/output \
  igait-human-gait \
  python3 yolo_slowfast.py [OPTIONS]
```

### Arguments

| Argument        | Type        | Default                            | Description                                                   |
| --------------- | ----------- | ---------------------------------- | ------------------------------------------------------------- |
| `--input`       | `str`       | `/home/wufan/images/video/vad.mp4` | Input video or folder                                         |
| `--output`      | `str`       | `/output/output.mp4`               | Path for processed output                                     |
| `--imsize`      | `int`       | `640`                              | Inference size (pixels) for YOLO                              |
| `--conf`        | `float`     | `0.4`                              | Object detection confidence threshold                         |
| `--iou`         | `float`     | `0.4`                              | IOU threshold for NMS                                         |
| `--device`      | `str`       | `cuda`                             | Device to run on: `cpu`, `cuda`, or specific GPU(s)           |
| `--classes`     | `list[int]` | `None`                             | Filter by YOLO class IDs (see below)                          |
| `--show`        | `flag`      | `False`                            | Display video frames during processing                        |
| `--max-seconds` | `int`       | `None`                             | Process only first N seconds of the video                     |
| `--mode`        | `str`       | `full`                             | `"full"` = full YOLO+SlowFast, `"walk"` = detect walking only |

---

## Example Usage

**Full pipeline on a video:**

```bash
docker run -it --gpus all \
  -v $(pwd)/model_weights/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \
  -v $(pwd)/model_weights/deep_sort/deep_sort/deep/checkpoint:/app/deep_sort/deep_sort/deep/checkpoint \
  -v $(pwd)/data:/files \
  -v $(pwd)/output:/output \
  igait-human-gait \
  python3 yolo_slowfast.py \
    --input /files/person_walking.mp4 \
    --output /output/processed_person_walking.mp4
```

**Walking detection only (first 60 seconds):**

```bash
docker run -it --gpus all \
  -v $(pwd)/model_weights/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \
  -v $(pwd)/model_weights/deep_sort/deep_sort/deep/checkpoint:/app/deep_sort/deep_sort/deep/checkpoint \
  -v $(pwd)/data:/files \
  -v $(pwd)/output:/output \
  igait-human-gait \
  python3 yolo_slowfast.py \
    --input /files/person_walking.mp4 \
    --output /output/walking_only.mp4 \
    --mode walk \
    --max-seconds 60
```

**Filter by classes (person + car + truck):**

```bash
--classes 0 2 7
```

**Check exit code:**

```bash
(docker run -it --gpus all \
  -v $(pwd)/model_weights/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \
  -v $(pwd)/model_weights/deep_sort/deep_sort/deep/checkpoint:/app/deep_sort/deep_sort/deep/checkpoint \
  -v $(pwd)/data:/files \
  -v $(pwd)/output:/output \
  igait-human-gait \
  python3 yolo_slowfast.py \
    --input /files/person_walking.mp4 \
    --output /output/out.mp4 \
    --max-seconds 60); echo $?
```

* `0` → Video contains exactly one person walking
* `1` → Video does **not** contain exactly one person walking

---

## Human Gait Detection Usage

```bash
docker run --gpus all \
  -v $(pwd)/model_weights/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints \
  -v $(pwd)/model_weights/deep_sort/deep_sort/deep/checkpoint:/app/deep_sort/deep_sort/deep/checkpoint \
  -v $(pwd)/data:/files \
  igait-human-gait \
  python3 validate_user.py /files/<VIDEO_FILE> --model KP --type video
```

* **Output:**

  * `0` → Video contains **exactly one person**
  * `1` → Video does **not** contain exactly one person

---

## Resources

* [Detectron2](https://github.com/facebookresearch/detectron2) – Framework powering the models
* [YOLOv5](https://github.com/ultralytics/yolov5) – Object detection
* [PyTorchVideo SlowFast](https://pytorchvideo.org/) – Action recognition
