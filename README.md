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

### 2. Prepare Input Folder

Create a `data/` folder and copy your input files:

```bash
mkdir data
cp <PATH_TO_DATA_FOLDER>/* data/
```

### 3. Setup Model Weights

Download the model weights from [Google Drive](https://drive.google.com/drive/folders/18I0CykECZOG6NUEF-JubJBTHXOKJz3qM?usp=drive_link) and place them in `weights/`:

```bash
mkdir weights
cp <PATH_TO_WEIGHTS_FOLDER>/* weights/
```

### 4. Build Docker Image

```bash
docker build -t igait-human-gait-detection .
```

---

## Running YOLO + SlowFast Pipeline

```bash
docker run -it --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  igait-human-gait-detection \
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
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  igait-human-gait-detection \
  python3 yolo_slowfast.py \
    --input /data/person_walking.mp4 \
    --output /output/processed_person_walking.mp4
```

**Walking detection only (first 60 seconds):**

```bash
docker run -it --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  igait-human-gait-detection \
  python3 yolo_slowfast.py \
    --input /data/person_walking.mp4 \
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
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  igait-human-gait-detection \
  python3 yolo_slowfast.py \
    --input /data/person_walking.mp4 \
    --output /output/out.mp4 \
    --max-seconds 60); echo $?
```

* `0` → Video contains exactly one person walking
* `1` → Video does **not** contain exactly one person walking

---

## Human Gait Detection Usage

```bash
docker run --gpus all \
  -v $(pwd)/data:/data \
  igait-video-precheck \
  python3 validate_user.py /data/<VIDEO_FILE> --model KP --type video
```

* **Output:**

  * `0` → Video contains **exactly one person**
  * `1` → Video does **not** contain exactly one person

---

## Resources

* [Detectron2](https://github.com/facebookresearch/detectron2) – Framework powering the models
* [YOLOv5](https://github.com/ultralytics/yolov5) – Object detection
* [PyTorchVideo SlowFast](https://pytorchvideo.org/) – Action recognition
