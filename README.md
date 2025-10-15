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
git clone https://github.com/igait-niu/igait-human-gait-detection.git
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

# Converting and Running on Metis

1. Pull the Docker Image
```sh
docker pull ghcr.io/igait-niu/igait-human-gait-detection:latest
```
2. Save the Docker Image as a Tarball
```sh
docker save -o igait-human-gait-detection.tar ghcr.io/igait-niu/igait-human-gait-detection:latest
```
3. Convert Docker Tarball to Apptainer Image
```sh
apptainer build igait-human-gait-detection.sif docker-archive://igait-human-gait-detection.tar
```
4. Transfer the Image to Metis
```sh
# If you already built the .sif
scp igait-human-gait-detection.sif your_zid@metis.niu.edu:/path/to/project/

# OR transfer the .tar if you’ll build on Metis
scp igait-human-gait-detection.tar your_zid@metis.niu.edu:/path/to/project/
```
Then, on Metis, if needed, build the .sif:
```sh
apptainer build igait-human-gait-detection.sif docker-archive://igait-human-gait-detection.tar
```
5. Run the Apptainer Image with Model Weights Binded
Below is an example command for running the igait-human-gait.sif Apptainer image on the Metis cluster, using GPU acceleration and bound data directories.
```sh
apptainer run --nv \
  --bind /etc/ssl/certs:/etc/ssl/certs \
  --bind /etc/pki:/etc/pki \
  --bind $(pwd)/model_weights/torch/hub:/root/.cache/torch/hub \
  --bind $(pwd)/model_weights/deep_sort/deep_sort/deep/checkpoint:/app/deep_sort/deep_sort/deep/checkpoint \
  --bind $(pwd)/data:/files \
  --bind $(pwd)/output:/output \
  igait-human-gait.sif \
  python3 yolo_slowfast.py \
    --input /files/person_walking.mp4 \
    --output /output/walking_only.mp4 \
    --mode walk \
    --max-seconds 60; echo $?
```

---

## Command Breakdown

### `apptainer run --nv`

* **`apptainer run`** executes the container’s default runtime environment.
* **`--nv`** enables **NVIDIA GPU support**, automatically binding CUDA libraries from the host system so the container can use the GPU.

  * Required for PyTorch, CUDA, or TensorRT workloads.

---

### `--bind <host_path>:<container_path>`

Each `--bind` flag mounts a directory or file from the host system into the container.
This allows your container to access external data, model weights, certificates, or output directories.

| Bind Mount                                                                                          | Purpose                                                                                                          |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `/etc/ssl/certs:/etc/ssl/certs`                                                                     | Gives the container access to host SSL certificates — needed for HTTPS requests (e.g., PyTorch model downloads). |
| `/etc/pki:/etc/pki`                                                                                 | Additional SSL certificates (CentOS/RHEL systems).                                                               |
| `$(pwd)/model_weights/torch/hub:/root/.cache/torch/hub`                                             | Mounts pretrained model weights for YOLO or SlowFast.                                                            |
| `$(pwd)/model_weights/deep_sort/deep_sort/deep/checkpoint:/app/deep_sort/deep_sort/deep/checkpoint` | Mounts the DeepSORT checkpoint directory used for object tracking.                                               |
| `$(pwd)/data:/files`                                                                                | Mounts local data directory containing input videos.                                                             |
| `$(pwd)/output:/output`                                                                             | Mounts a local output directory for saving processed results.                                                    |

> `$(pwd)` dynamically inserts your **current working directory**, making the command portable without hardcoded paths.

---

### `igait-human-gait.sif`

This is your **Apptainer image file** — a self-contained environment that includes:

* CUDA & cuDNN runtime
* PyTorch
* YOLO + SlowFast + DeepSORT code and dependencies

It behaves like a lightweight virtual machine dedicated to your application.

---

### `python3 yolo_slowfast.py`

Specifies the **entry command** to run inside the container.
You can replace this with any other Python script or command (e.g., `python3 demo.py` or `bash`).

---

### Script Arguments

| Argument                            | Description                                                                            |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| `--input /files/person_walking.mp4` | Input video path inside the container (mapped from your local `data/` directory).      |
| `--output /output/walking_only.mp4` | Output file path inside the container (will appear in your local `output/` directory). |
| `--mode walk`                       | Custom script parameter (sets mode to walking detection).                              |
| `--max-seconds 60`                  | Limits video processing to the first 60 seconds.                                       |

---

### `; echo $?`

This prints the **exit status code** of the last command:

* `0` = success
* Nonzero = error occurred

Useful for batch scripts or job monitoring on HPC systems.

---

## General Template

You can adapt this pattern for any container and dataset:

```bash
apptainer run --nv \
  --bind /path/to/weights:/app/weights \
  --bind /path/to/data:/app/data \
  --bind /path/to/output:/app/output \
  your-container.sif \
  python3 your_script.py \
    --input /app/data/input.mp4 \
    --output /app/output/result.mp4 \
    [additional args]
```

---

## Resources

* [Detectron2](https://github.com/facebookresearch/detectron2) – Framework powering the models
* [YOLOv5](https://github.com/ultralytics/yolov5) – Object detection
* [PyTorchVideo SlowFast](https://pytorchvideo.org/) – Action recognition
