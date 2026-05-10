# pyright: reportExplicitAny=false, reportUnknownArgumentType=false, reportAny=false, reportMissingTypeStubs=false

import os
import torch

from typing import Any, cast
from collections.abc import Iterator
from json_stream.base import TransientStreamingJSONObject
import kagglehub # type: ignore
import json_stream # type: ignore

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from scripts.vit.lib.dataset import LAVDFDataset
from scripts.vit.lib.parser import stream_lavdf, get_clip_label



dataset_root = kagglehub.dataset_download('elin75/localized-audio-visual-deepfake-dataset-lav-df') + "/LAV-DF"
metadata_path = os.path.join(dataset_root, "metadata.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset = LAVDFDataset(metadata_path, dataset_root, limit=50)
