"""
MediaPipe-based human gait detection.

Scans an input video and reports:
  - Whether at least one human is present.
  - Whether multiple humans were ever visible in the same frame (flag).
  - Whether at least one human was walking during the clip.

Exit codes:
  0 -> video contains at least one human and at least one of them is walking
  1 -> video is missing a human, missing walking motion, or otherwise invalid
  2 -> the input video could not be opened
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_MODEL_PATH = os.environ.get(
    "POSE_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker.task"),
)

# BlazePose landmark indices used by the walking heuristic.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
REQUIRED_LANDMARKS = (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_ANKLE, RIGHT_ANKLE,
)


def ensure_model(path: str) -> str:
    """Download the MediaPipe pose landmarker model if it is not already cached."""
    if os.path.exists(path):
        return path
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    print(f"Downloading MediaPipe pose model -> {path}")
    urllib.request.urlretrieve(POSE_MODEL_URL, path)
    return path


def _body_scale(landmarks) -> float:
    shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    ankle_y = (landmarks[LEFT_ANKLE].y + landmarks[RIGHT_ANKLE].y) / 2
    return max(abs(ankle_y - shoulder_y), 1e-3)


def _bbox_area(landmarks) -> float:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _landmarks_visible(landmarks, min_visibility: float) -> bool:
    return all(landmarks[i].visibility >= min_visibility for i in REQUIRED_LANDMARKS)


@dataclass
class WalkingDetector:
    """Flags walking by watching ankle oscillation and hip horizontal travel
    across a sliding window of frames. Positions are normalised against the
    subject's current body scale so depth changes do not bias the signal."""

    window: int
    ankle_osc_threshold: float = 0.04
    hip_motion_threshold: float = 0.05
    left_ankle_rel_y: List[float] = field(default_factory=list)
    right_ankle_rel_y: List[float] = field(default_factory=list)
    hip_x_norm: List[float] = field(default_factory=list)

    def push(self, landmarks) -> None:
        scale = _body_scale(landmarks)
        hip_mid_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2
        hip_mid_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2

        self.left_ankle_rel_y.append((landmarks[LEFT_ANKLE].y - hip_mid_y) / scale)
        self.right_ankle_rel_y.append((landmarks[RIGHT_ANKLE].y - hip_mid_y) / scale)
        self.hip_x_norm.append(hip_mid_x / scale)

        if len(self.left_ankle_rel_y) > self.window:
            self.left_ankle_rel_y.pop(0)
            self.right_ankle_rel_y.pop(0)
            self.hip_x_norm.pop(0)

    def is_walking(self) -> bool:
        if len(self.left_ankle_rel_y) < self.window:
            return False
        left_std = float(np.std(self.left_ankle_rel_y))
        right_std = float(np.std(self.right_ankle_rel_y))
        hip_range = max(self.hip_x_norm) - min(self.hip_x_norm)

        ankles_oscillating = (
            left_std > self.ankle_osc_threshold
            and right_std > self.ankle_osc_threshold
        )
        hip_translating = hip_range > self.hip_motion_threshold
        return ankles_oscillating or hip_translating

    def reset(self) -> None:
        self.left_ankle_rel_y.clear()
        self.right_ankle_rel_y.clear()
        self.hip_x_norm.clear()


def _draw_overlay(frame, poses, status_text: str):
    h, w = frame.shape[:2]
    for idx, landmarks in enumerate(poses):
        color = (0, 255, 0) if idx == 0 else (0, 128, 255)
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, color, -1)

    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def _open_writer(output_path: Optional[str], fps: float, width: int, height: int):
    if not output_path:
        return None
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def run(config) -> None:
    model_path = ensure_model(config.model or DEFAULT_MODEL_PATH)

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=config.max_poses,
        min_pose_detection_confidence=config.conf,
        min_pose_presence_confidence=config.conf,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(config.input)
    if not cap.isOpened():
        print(f"ERROR: unable to open video: {config.input}", file=sys.stderr)
        sys.exit(2)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(config.output, fps, width, height)

    walking = WalkingDetector(window=max(int(round(fps)), 10))

    frame_idx = 0
    frames_with_person = 0
    frames_with_multiple = 0
    max_humans = 0
    walking_detected = False
    multi_person_flag = False
    last_timestamp_ms = -1
    start = time.time()

    with mp_vision.PoseLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if config.max_seconds is not None and frame_idx / fps >= config.max_seconds:
                print(f"Reached max processing time of {config.max_seconds}s")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(round(frame_idx * 1000 / fps))
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(mp_image, timestamp_ms)
            poses = result.pose_landmarks
            n_people = len(poses)
            max_humans = max(max_humans, n_people)

            if n_people >= 1:
                frames_with_person += 1
                primary = max(poses, key=_bbox_area)
                if _landmarks_visible(primary, config.min_visibility):
                    walking.push(primary)
                    if walking.is_walking():
                        walking_detected = True
                else:
                    walking.reset()
            else:
                walking.reset()

            if n_people > 1:
                frames_with_multiple += 1
                multi_person_flag = True

            if writer is not None:
                status = (
                    f"people={n_people} walking={'yes' if walking_detected else 'no'} "
                    f"multi={'yes' if multi_person_flag else 'no'}"
                )
                writer.write(_draw_overlay(frame.copy(), poses, status))

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - start
    human_detected = frames_with_person > 0
    valid = human_detected and walking_detected

    summary = {
        "valid": valid,
        "human_detected": human_detected,
        "multiple_humans_detected": multi_person_flag,
        "walking_detected": walking_detected,
        "max_humans_in_frame": max_humans,
        "total_frames": frame_idx,
        "frames_with_person": frames_with_person,
        "frames_with_multiple_persons": frames_with_multiple,
        "processing_time_seconds": round(elapsed, 3),
    }

    print(json.dumps(summary, indent=2))

    if config.output_json:
        parent = os.path.dirname(os.path.abspath(config.output_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(config.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Validity results written to: {config.output_json}")

    sys.exit(0 if valid else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MediaPipe-based human gait detection.",
    )
    parser.add_argument("--input", required=True,
                        help="Path to input video file")
    parser.add_argument("--output", default=None,
                        help="Optional path for an annotated output video")
    parser.add_argument("--output-json", dest="output_json", default=None,
                        help="Optional path to write the JSON summary")
    parser.add_argument("--model", default=None,
                        help="Path to pose_landmarker.task (defaults to "
                             "models/pose_landmarker.task; auto-downloaded)")
    parser.add_argument("--max-poses", dest="max_poses", type=int, default=5,
                        help="Maximum number of poses detected per frame")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Minimum pose detection and presence confidence")
    parser.add_argument("--min-visibility", dest="min_visibility",
                        type=float, default=0.5,
                        help="Minimum landmark visibility before a frame is "
                             "fed to the walking detector")
    parser.add_argument("--max-seconds", dest="max_seconds", type=float,
                        default=None,
                        help="Process only the first N seconds of the video")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    if isinstance(config.input, str) and config.input.isdigit():
        config.input = int(config.input)
    run(config)
