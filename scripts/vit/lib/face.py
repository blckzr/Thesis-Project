# pyright: reportExplicitAny=false, reportMissingTypeStubs=false, reportAny=false

from __future__ import annotations

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from typing import Any


LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405,
                 321, 375, 291, 308, 402, 317, 14, 87,
                 178, 88, 95, 185, 40, 39, 37, 0,
                 267, 269, 270, 409]

def build_face_landmarker(model_path: str = "face_landmarker.task") -> Any: # returns mp_vision.FaceLandmarker
    """
    Downloads and builds the FaceLandmarker task.
    Call once at startup, pass the instance into extract_lip_roi().
    """
    # Download model asset if not present
    if not os.path.exists(model_path):
        import urllib.request
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        print(f"Downloading FaceLandmarker model to {model_path}...")
        _ = urllib.request.urlretrieve(url, model_path)

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,   # single-image, synchronous
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)

def extract_lip_roi(
    frame: np.ndarray,
    landmarker: Any, # is mp_vision.FaceLandmarker
    size: tuple[int, int] = (224, 224),
    padding: int = 10,
) -> np.ndarray | None:
    """
    Extracts and resizes the lip ROI from a BGR frame.

    Args:
        frame:       BGR numpy array (H, W, 3)
        landmarker:  reusable FaceLandmarker instance (Tasks API)
        size:        output crop size
        padding:     pixels to pad around the bounding box

    Returns:
        Resized lip crop as uint8 numpy array, or None if no face found.
    """
    h, w = frame.shape[:2]

    # tasks API expects RGB MediaPipe Image
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks: list[NormalizedLandmark] = result.face_landmarks[0]

    xs = [landmarks[i].x * w for i in LIP_LANDMARKS]
    ys = [landmarks[i].y * h for i in LIP_LANDMARKS]

    x1 = max(0, int(min(xs)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y1 = max(0, int(min(ys)) - padding)
    y2 = min(h, int(max(ys)) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return cv2.resize(frame[y1:y2, x1:x2], size)

def load_lip_frames(
    video_path: str,
    landmarker: Any, # is mp_vision.FaceLandmarker
) -> list[np.ndarray]:
    """
    Extracts lip ROI frames from a video file.

    Args:
        video_path:  path to .mp4 file
        landmarker:  reusable FaceLandmarker instance
    """
    cap    = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lip = extract_lip_roi(frame, landmarker)
        if lip is not None:
            frames.append(lip)

    cap.release()
    return frames
