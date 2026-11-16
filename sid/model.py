"""Hybrid CNN-GRU speaker encoder (paper Section III-B, Fig. 1)."""
import torch
import torch.nn as nn

from . import config as C
from .features import LogMel


class CNNGRU(nn.Module):
    def __init__(self, n_speakers=C.N_TRAIN_SPEAKERS, embedding_dim=C.EMBEDDING_DIM):
        super().__init__()
        # Layer 1 - low-level features: 32 filters 3x3, BN, ReLU, 2x2 max-pool
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Layer 2 - mid-level patterns: 64 filters 3x3, BN, ReLU, dropout 0.3, 2x2 max-pool
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.MaxPool2d(2),
        )
        # Layer 3 - high-level speaker features: 128 filters 3x3, BN, ReLU, dropout 0.4,
        # then global average pooling over the frequency axis (time is kept for the GRU)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        # Bidirectional GRU, 256 hidden units
        self.gru = nn.GRU(128, C.GRU_HIDDEN, batch_first=True, bidirectional=True)
        # Embedding projection: e = tanh(W h + b), 768-d
        self.proj = nn.Linear(2 * C.GRU_HIDDEN, embedding_dim)
        # Classification head - used only for training, discarded at inference
        self.classifier = nn.Linear(embedding_dim, n_speakers)

    def embed(self, mel):
        """mel: [B, 80, T] in 0..1 -> [B, 768] speaker embedding."""
        x = mel.unsqueeze(1)
        x = self.conv3(self.conv2(self.conv1(x)))      # [B, 128, 20, T/4]
        x = x.mean(dim=2).transpose(1, 2)              # [B, T/4, 128]
        out, _ = self.gru(x)                           # [B, T/4, 512]
        h = out.mean(dim=1)
        return torch.tanh(self.proj(h))

    def forward(self, mel):
        return self.classifier(self.embed(mel))


def load_encoder(path=C.BEST_MODEL, device="cpu"):
    """Load the trained model plus its feature front-end, ready for embedding extraction."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = CNNGRU(n_speakers=ckpt["n_speakers"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, LogMel().to(device)
