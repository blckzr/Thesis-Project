# pyright: reportExplicitAny=false, reportUnknownArgumentType=false, reportAny=false, reportMissingTypeStubs=false

from typing import Any, cast
from collections.abc import Iterator
from dataclasses import dataclass
from json_stream.base import TransientStreamingJSONObject
import kagglehub # type: ignore
import json_stream # type: ignore

dataset_root = kagglehub.dataset_download('elin75/localized-audio-visual-deepfake-dataset-lav-df')

# ── Entry type hint ───────────────────────────────────────────────
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

# ====== Streaming parser =========
def stream_lavdf(metadata_path: str, split: str | None = None) -> Iterator[LAVDFEntry]: #
    """
    Streams LAV-DF metadata entries lazily.
    
    Args:
        metadata_path: path to metadata.json
        split: if given, only yield entries from "train" or "test"
    
    Yields:
        dict with all fields fully materialized
    """
    with open(metadata_path, 'r') as f:
        data = cast(TransientStreamingJSONObject | None, json_stream.load(f))

        if isinstance(data, (int, float, str, bool)) or data is None:
            raise ValueError("Expected a JSON array or object, but got a primitive.")

        for streaming_raw in data.persistent(): # pyright: ignore[reportUnknownVariableType]
            # json_stream gives you a streaming dict, .collect() or manual read
            # probably need to materialize all of this so we can store it lol.
            raw = cast(dict[str, Any], json_stream.to_standard_types(streaming_raw)) # pyright: ignore[reportUnknownMemberType]

            materialized: LAVDFEntry = LAVDFEntry(
                file=str(raw["file"]),
                n_fakes=int(raw["n_fakes"]),
                duration=float(raw["duration"]),
                transcript=str(raw["transcript"]),
                original=raw["original"],
                modify_video=bool(raw["modify_video"]),
                modify_audio=bool(raw["modify_audio"]),
                split=str(raw["split"]),
                video_frames=int(raw["video_frames"]),
                audio_channels=int(raw["audio_channels"]),
                audio_frames=int(raw["audio_frames"]),
                fake_periods=[
                    (float(p[0]), float(p[1]))
                    for p in cast(list[tuple[float, float]], raw["fake_periods"])
                ],
                timestamps=[
                    (str(t[0]), float(t[1]), float(t[2]))
                    for t in cast(list[tuple[str, float, float]], raw["timestamps"])
                ]
            )


            # Filter by split early to avoid loading unneeded entries
            if split is not None and materialized.split != split:
                continue

            yield materialized

def get_clip_label(entry: LAVDFEntry, clip_start: float, clip_end: float) -> int:
    """
    Returns 1 (fake) if the clip overlaps any fake_period, else 0 (real).
    Uses localized labels.
    """
    for period_start, period_end in entry.fake_periods:
        overlap = min(clip_end, period_end) - max(clip_start, period_start)
        if overlap > 0:
            return 1
    return 0

if __name__ == "__main__":
    for i, entry in enumerate(stream_lavdf(dataset_root + "/LAV-DF/metadata.json", split="train")):
        print(f"File:        {entry.file}")
        print(f"Duration:    {entry.duration:.2f}s")
        print(f"Fake periods:{entry.fake_periods}")
        print(f"Transcript:  {entry.transcript[:80]}...")
        print(f"Label (0–2s):{get_clip_label(entry, 0.0, 2.0)}")
        print("---")
        if i >= 4:
            break
