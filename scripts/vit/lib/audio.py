
import os
import librosa
import subprocess, tempfile

def extract_mfcc(video_path: str, n_mfcc=128, sr=16000):
    _, tmp = tempfile.mkstemp(suffix=".wav")
    _ = subprocess.run(
        ["ffmpeg", "-i", video_path, "-ar", str(sr), "-ac", "1", tmp, "-y", "-loglevel", "quiet"]
    )
    audio, _ = librosa.load(tmp, sr=sr)
    os.remove(tmp)
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=512)
