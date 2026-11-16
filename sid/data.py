"""LibriSpeech loading: utterance listing, 80/10/10 split, random 3 s crops."""
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from . import config as C


def list_utterances(root):
    """-> (items [(path, speaker_index)], speaker_ids) for a LibriSpeech subset folder."""
    speakers = sorted((d.name for d in root.iterdir() if d.is_dir()), key=int)
    index = {s: i for i, s in enumerate(speakers)}
    items = [(str(p), index[p.parts[-3]]) for p in sorted(root.glob("*/*/*.flac"))]
    return items, speakers


def split_utterances(items, seed=C.SEED):
    """Per-speaker 80/10/10 split so every speaker appears in train, val and test."""
    by_spk = {}
    for it in items:
        by_spk.setdefault(it[1], []).append(it)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for spk in sorted(by_spk):
        utts = by_spk[spk]
        rng.shuffle(utts)
        n = len(utts)
        n_val = max(1, round(0.1 * n))
        n_test = max(1, round(0.1 * n))
        val += utts[:n_val]
        test += utts[n_val:n_val + n_test]
        train += utts[n_val + n_test:]
    return train, val, test


class CropDataset(Dataset):
    """Each item: `crops` fixed-length crops of one utterance + its speaker label."""

    def __init__(self, items, crop_samples, crops, random_crops):
        self.items, self.crop_samples, self.crops, self.random_crops = items, crop_samples, crops, random_crops

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        wav, sr = sf.read(path, dtype="float32")
        assert sr == C.SAMPLE_RATE
        n = self.crop_samples
        if len(wav) < n + 1:
            wav = np.tile(wav, int(np.ceil((n + 1) / len(wav))))
        last = len(wav) - n
        if self.random_crops:
            starts = np.random.randint(0, last + 1, size=self.crops)
        else:
            starts = np.linspace(0, last, self.crops + 2)[1:-1].astype(int)
        x = np.stack([wav[s:s + n] for s in starts])
        return torch.from_numpy(x), torch.full((self.crops,), label, dtype=torch.long)


def collate(batch):
    return torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])
