"""Decode any audio/video file to 16 kHz mono float32 (uses PyAV, no ffmpeg install needed)."""
import av
import numpy as np

from . import config as C


def load_audio(path, sample_rate=C.SAMPLE_RATE):
    chunks = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("No audio track found in this file.")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("The audio track is empty.")
    return np.concatenate(chunks).astype(np.float32) / 32768.0
