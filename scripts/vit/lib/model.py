# pyright: reportExplicitAny=false, reportMissingTypeStubs=false, reportAny=false

from typing import override
from torch import Tensor, nn
import timm


class LipSyncVITDeepfakeDetector(nn.Module):
    vit: nn.Module
    audio_encoder: nn.Sequential
    fusion: nn.MultiheadAttention
    classifier: nn.Sequential
    def __init__(self, n_mfcc: int = 128):
        super().__init__()
        # small vit model from torch themselves for image recognition.
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)

        # Freeze all except last 2 transformer blocks
        for name, param in self.vit.named_parameters():
            param.requires_grad = any(f"blocks.{i}" in name for i in [10, 11])

        # multi-layer perceptron to expand the tensor into 384 dims, and discards any unimportant features.
        self.audio_encoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 384)
        )
        # used to fuse the two tensors that come out of the two neural networks
        self.fusion = nn.MultiheadAttention(embed_dim=384, num_heads=6, batch_first=True)
        # used to gradually compress the fused tensors into one value (+, -) representing two classifications
        self.classifier = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    @override
    def forward(self, frames: Tensor, mfcc: Tensor) -> Tensor:
        # tensor to one-dim tensor
        video_embedding: Tensor = self.vit(frames).unsqueeze(1) # (B, 1, 384)
        audio_embedding = self.audio_encoder(mfcc).unsqueeze(1) # (B, 1, 384)

        # cross attention, fusion(q, k, v), query, key, value respectively.
        #   this boils down to: 
        #   "this is the lip movement im looking at (query), does the model think it's related to the audio (key)?"
        #   "if so, get the audio tensor (value)."
        fused, _ = self.fusion(video_embedding, audio_embedding, audio_embedding) # (B, 1, 384)

        # all results into a single value.
        return self.classifier(fused.squeeze(1)) # (B, 1)

