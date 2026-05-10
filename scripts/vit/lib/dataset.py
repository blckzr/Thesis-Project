# pyright: reportExplicitAny=false, reportMissingTypeStubs=false, reportAny=false

import os
import librosa
import torch
import numpy as np

from torch import Tensor
from typing import Any, cast, TypeAlias, override
from collections.abc import Iterator
from json_stream.base import TransientStreamingJSONObject

from torch.utils.data import Dataset
from torchvision import transforms

from scripts.vit.lib.audio import extract_mfcc
from scripts.vit.lib.face import build_face_landmarker, load_lip_frames
from scripts.vit.lib.parser import get_clip_label, stream_lavdf

Sample: TypeAlias = tuple[Tensor, Tensor, Tensor]

class LAVDFDataset(Dataset[Sample]):
    samples: list[Sample]
    clip_len: int
    pos_weight: float
    transform: transforms.Compose
    def __init__(self, metadata_path: str, dataset_root: str, split: str = "train", clip_len: int = 16, limit: int | None = 50, n_mfcc: int = 128):
        self.samples = []
        self.clip_len = clip_len
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        landmarker = build_face_landmarker()

        print(f"\nBuilding {split} dataset...")
        for i, entry in enumerate(stream_lavdf(metadata_path, split)):
            if not entry.modify_video: # skipping unmodified videos.
                continue

            video_path = os.path.join(dataset_root, entry.file)
            if not os.path.exists(video_path):
                continue

            print(f"File:        {entry.file}")
            print(f"Duration:    {entry.duration:.2f}s")
            print(f"Fake periods:{entry.fake_periods}")
            print(f"Transcript:  {entry.transcript[:80]}...")
            print(f"Label (0–2s):{get_clip_label(entry, 0.0, 2.0)}")

            # ====== VIDEO =======
            print("Loading lip frames...")
            frames = load_lip_frames(video_path, landmarker)
            if len(frames) < clip_len:
                continue

            # ====== AUDIO =======
            print("Extracing MFCC speech features...")
            mfcc = extract_mfcc(video_path, n_mfcc=n_mfcc)
            fps = len(frames) / entry.duration
            print("Resampling MFCC speech features...")
            mfcc_resampled = librosa.util.fix_length(mfcc, size=len(frames), axis=1)

            # ====== VIDEO AND AUDIO AUGMENTATION ======= 
            print("Augmenting audio and lip video together + sliding window.")
            for start in range(0, len(frames) - clip_len, clip_len // 2):
                end         = start + clip_len
                clip_start  = start / fps # frame to seconds
                clip_end    = end   / fps # frame to seconds
                mid         = start + clip_len // 2

                # combines the multiple frames (clip_len = 16) into a single "average" frame as a representative.
                frame_t = cast(Tensor, self.transform(frames[mid]))
                mfcc_t = torch.tensor(mfcc_resampled[:, start:end].mean(axis=1), # torch.Tensor (128,) float32
                       dtype=torch.float32)
                # labelling if this extracted clip is a deepfake.
                label_t = torch.tensor(get_clip_label(entry, clip_start, clip_end), # torch.Tensor scalar float32
                                       dtype=torch.float32)
 
                self.samples.append((frame_t, mfcc_t, label_t))

            print("---")

            # TEMPORARY!!
            if limit is not None and i >= limit: 
                print(f"WARN!!!! The dataset stopped at {limit} videos for testing! Remove this if we're absolutely sure everything is okay!")
                break

        landmarker.close()

        n_fake = float(sum(l for _, _, l in self.samples))
        n_real = float(len(self.samples)) - n_fake
        print(f"  >> {len(self.samples)} clips | Real: {n_real} | Fake: {n_fake}")
        self.pos_weight = n_real / max(n_fake, 1)

    def __len__(self): 
        return len(self.samples)

    @override
    def __getitem__(self, idx: int) -> Sample:
        f, m, l = self.samples[idx]
        return f, m, l

