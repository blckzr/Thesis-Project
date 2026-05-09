
from dataclasses import dataclass

@dataclass
class LAVDFEntry:
    file: str
    n_fakes: int
    fake_periods: list[tuple[float, float]]
    timestamps: list[tuple[str, float, float]]
    duration: float
    transcript: str
    original: str | None
    modify_video: bool
    modify_audio: bool
    split: str          # "train" | "test"
    video_frames: int
    audio_channels: int
    audio_frames: int
