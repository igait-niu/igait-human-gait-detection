# igait-human-gait-detection

<img src="./person_walking.gif"/>

MediaPipe-based detection service that inspects a video and determines whether:

1. At least one human is present.
2. More than one human appears at any point (returned as a flag).
3. At least one human is walking.

The pipeline uses the [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
task model for multi-person pose estimation on CPU, then derives walking motion
from ankle oscillation and hip translation in normalised body coordinates.

**Exit codes:**

* `0` — at least one human detected **and** at least one human walking.
* `1` — the video is missing a human or is missing walking motion.
* `2` — the input video could not be opened.

The `multiple_humans_detected` field in the JSON summary is informational;
it does **not** invalidate the result on its own.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/igait-niu/igait-human-gait-detection.git
cd igait-human-gait-detection
```

### 2. Build the Docker image

The pose landmarker model (`pose_landmarker_full.task`) is fetched inside the
image, so no separate weight-download step is required.

```bash
docker build -t igait-human-gait .
```

### 3. Prepare an input folder

Drop the video clips you want to inspect into `data/`:

```bash
cp <PATH_TO_YOUR_VIDEOS>/* data/
```

---

## Running the detector

```bash
docker run --rm \
  -v $(pwd)/data:/files \
  -v $(pwd)/output:/output \
  igait-human-gait \
    --input /files/person_walking.mp4 \
    --output /output/person_walking_annotated.mp4 \
    --output-json /output/person_walking.json
```

### Arguments

| Argument           | Type    | Default                         | Description                                                                                 |
| ------------------ | ------- | ------------------------------- | ------------------------------------------------------------------------------------------- |
| `--input`          | `str`   | *(required)*                    | Path to the input video.                                                                    |
| `--output`         | `str`   | `None`                          | Optional path for an annotated output video (skeleton overlay + status banner).             |
| `--output-json`    | `str`   | `None`                          | Optional path for a JSON summary of the run.                                                |
| `--model`          | `str`   | `models/pose_landmarker.task`   | Override the pose landmarker model path. Auto-downloaded to `models/` on first run locally. |
| `--max-poses`      | `int`   | `5`                             | Maximum number of poses detected per frame (controls the multi-person flag sensitivity).    |
| `--conf`           | `float` | `0.5`                           | Minimum pose detection and presence confidence.                                             |
| `--min-visibility` | `float` | `0.5`                           | Minimum landmark visibility before a frame contributes to walking analysis.                 |
| `--max-seconds`    | `float` | `None`                          | Process only the first *N* seconds of the video.                                            |

### Example: first 60 seconds only

```bash
docker run --rm \
  -v $(pwd)/data:/files \
  -v $(pwd)/output:/output \
  igait-human-gait \
    --input /files/person_walking.mp4 \
    --output-json /output/person_walking.json \
    --max-seconds 60
```

### Example: check the exit code

```bash
(docker run --rm \
  -v $(pwd)/data:/files \
  -v $(pwd)/output:/output \
  igait-human-gait \
    --input /files/person_walking.mp4); echo $?
```

* `0` → video contains a human and walking motion.
* `1` → human or walking missing.

---

## JSON output schema

When `--output-json` is supplied (or the summary is captured from stdout) the
detector emits the following structure:

```json
{
  "valid": true,
  "human_detected": true,
  "multiple_humans_detected": false,
  "walking_detected": true,
  "max_humans_in_frame": 1,
  "total_frames": 425,
  "frames_with_person": 418,
  "frames_with_multiple_persons": 0,
  "processing_time_seconds": 7.812
}
```

* `valid` — `true` when `human_detected` **and** `walking_detected`.
* `multiple_humans_detected` — informational flag; `true` if more than one pose
  was detected in any single frame.

---

## Local (non-Docker) usage

```bash
python3 -m pip install -r requirements.txt
python3 gait_detect.py \
  --input data/person_walking.mp4 \
  --output output/person_walking_annotated.mp4 \
  --output-json output/person_walking.json
```

On first run the script downloads `pose_landmarker_full.task` into
`./models/pose_landmarker.task`. Set `POSE_MODEL_PATH` or pass `--model` to
point at a cached copy.

---

## Running a batch over `data/`

```bash
python3 test.py
```

`test.py` iterates every file in `data/`, calls `gait_detect.py`, and drops an
annotated video plus a JSON summary into `output/` per input.

---

## Running on Metis (Apptainer)

1. Pull and export the image:

   ```bash
   docker pull ghcr.io/igait-niu/igait-human-gait-detection:latest
   docker save -o igait-human-gait-detection.tar ghcr.io/igait-niu/igait-human-gait-detection:latest
   apptainer build igait-human-gait-detection.sif docker-archive://igait-human-gait-detection.tar
   ```

2. Transfer the `.sif` (or the `.tar`) to Metis and run:

   ```bash
   apptainer run \
     --bind $(pwd)/data:/files \
     --bind $(pwd)/output:/output \
     igait-human-gait-detection.sif \
       --input /files/person_walking.mp4 \
       --output /output/person_walking_annotated.mp4 \
       --output-json /output/person_walking.json; echo $?
   ```

No `--nv` flag is required — the MediaPipe pose landmarker runs on CPU.

---

## Resources

* [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
* [BlazePose landmark reference](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model)
