"""Waveform -> normalised log-mel spectrogram (80 mel bands, 512-pt STFT, 50% hop, Hann)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def _hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_filterbank(n_freqs, n_mels, sample_rate, f_min=0.0, f_max=None):
    """Triangular mel filterbank, shape [n_freqs, n_mels]."""
    f_max = f_max or sample_rate / 2
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    m_pts = torch.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - freqs.unsqueeze(1)          # [n_freqs, n_mels+2]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return torch.clamp(torch.min(down, up), min=0.0)


def phase_vocoder(spec, rate, hop_length):
    """Time-stretch a complex STFT [B, F, T] by `rate` (>1 = faster) without changing pitch."""
    if abs(rate - 1.0) < 1e-4:
        return spec
    n_freq = spec.shape[1]
    phase_advance = torch.linspace(0, math.pi * hop_length, n_freq, device=spec.device)[None, :, None]
    time_steps = torch.arange(0, spec.shape[-1], rate, device=spec.device, dtype=torch.float32)
    alphas = (time_steps % 1.0)[None, None, :]
    phase_0 = spec[..., :1].angle()
    spec = F.pad(spec, [0, 2])
    idx = time_steps.long()
    s0 = spec.index_select(-1, idx)
    s1 = spec.index_select(-1, idx + 1)
    a0, a1 = s0.angle(), s1.angle()
    n0, n1 = s0.abs(), s1.abs()
    phase = a1 - a0 - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, dim=-1)
    mag = alphas * n1 + (1 - alphas) * n0
    return torch.polar(mag, phase_acc)


class LogMel(nn.Module):
    """Batch of waveforms [B, T] -> [B, 80, N_FRAMES], each spectrogram scaled to 0..1."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(C.N_FFT), persistent=False)
        self.register_buffer(
            "fbank",
            mel_filterbank(C.N_FFT // 2 + 1, C.N_MELS, C.SAMPLE_RATE, 0.0, 8000.0),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, wav, pitch_semitones=0.0, stretch=1.0, n_frames=C.N_FRAMES):
        # Stage 1: zero mean / unit variance per segment
        wav = wav - wav.mean(dim=-1, keepdim=True)
        wav = wav / (wav.std(dim=-1, keepdim=True) + 1e-5)

        # Pitch shift = resample by p (pitch and tempo both x p) then undo the tempo change
        # with the phase vocoder. Time stretch is folded into the same vocoder pass.
        p = 2.0 ** (pitch_semitones / 12.0)
        if abs(p - 1.0) > 1e-4:
            new_len = int(round(wav.shape[-1] / p))
            wav = F.interpolate(wav.unsqueeze(1), size=new_len, mode="linear", align_corners=False).squeeze(1)

        spec = torch.stft(wav, C.N_FFT, C.HOP_LENGTH, window=self.window, return_complex=True)
        spec = phase_vocoder(spec, stretch / p, C.HOP_LENGTH)

        power = spec.abs().pow(2)                                   # [B, 257, T]
        mel = torch.matmul(power.transpose(1, 2), self.fbank).transpose(1, 2)
        logmel = torch.log(mel + 1e-6)

        if n_frames is not None:
            if logmel.shape[-1] < n_frames:
                logmel = F.pad(logmel, [0, n_frames - logmel.shape[-1]], mode="replicate")
            logmel = logmel[..., :n_frames]

        # Stage 2: scale each spectrogram to 0..1
        lo = logmel.amin(dim=(1, 2), keepdim=True)
        hi = logmel.amax(dim=(1, 2), keepdim=True)
        return (logmel - lo) / (hi - lo + 1e-6)
