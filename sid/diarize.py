"""Speaker segmentation pipeline (paper III-D): RMS voice activity detection -> 3 s sliding
windows -> mel spectrograms -> CNN-GRU embeddings -> spectral clustering -> majority-vote
timeline -> post-processing."""
import numpy as np
import torch

from . import config as C
from .clustering import centroids, normalize, spectral_cluster

SR = C.SAMPLE_RATE
FRAME_HOP = 0.010          # VAD frame hop (s)


def _runs(x):
    """Yield (start, end, value) for each run of equal values in a 1-D array."""
    change = np.flatnonzero(np.diff(x.astype(int))) + 1
    bounds = np.concatenate([[0], change, [len(x)]])
    for s, e in zip(bounds[:-1], bounds[1:]):
        yield int(s), int(e), x[s]


def _fill_short_runs(mask, value, min_len):
    """Flip runs of `value` shorter than `min_len` (interior runs only)."""
    out = mask.copy()
    for s, e, v in _runs(mask):
        if v == value and e - s < min_len and s > 0 and e < len(mask):
            out[s:e] = not value
    return out


def rms_vad(wav):
    """RMS-energy voice activity detection (Stage 4). Returns a boolean mask per 10 ms frame.

    RMS = sqrt(mean(x^2)) per 25 ms frame; the recording is loudness-normalised first so the
    paper's fixed threshold (0.02) works at any recording level."""
    frame, hop = int(0.025 * SR), int(FRAME_HOP * SR)
    if len(wav) < frame:
        wav = np.pad(wav, (0, frame - len(wav)))
    n = 1 + (len(wav) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(wav[idx] ** 2, axis=1))
    ref = np.percentile(rms, 95)
    if ref < 1e-5:
        return np.zeros(n, bool)
    speech = (rms * (C.VAD_REFERENCE_LEVEL / ref)) > C.VAD_RMS_THRESHOLD
    speech = _fill_short_runs(speech, False, int(C.VAD_MIN_PAUSE_SEC / FRAME_HOP))   # bridge tiny gaps
    return _fill_short_runs(speech, True, int(0.15 / FRAME_HOP))                     # drop blips


def _windows(speech):
    """3 s sliding windows placed inside each continuous stretch of speech, so that a window
    rarely straddles a pause (where most speaker changes happen).
    -> list of (start_sample, end_sample); stretches shorter than 3 s give one short window."""
    w, hop = C.WINDOW_SAMPLES, int(C.INFER_HOP_SEC * SR)
    per = int(FRAME_HOP * SR)
    out = []
    for s, e, v in _runs(speech):
        if not v:
            continue
        a, b = s * per, e * per
        if b - a <= w:
            out.append((a, b))
        else:
            starts = list(range(a, b - w, hop)) + [b - w]
            out += [(st, st + w) for st in starts]
    return out


@torch.no_grad()
def extract_embeddings(wav, windows, model, logmel, device, batch=256):
    """Embedding for each (start, end) window. Short windows are repeated to fill 3 s."""
    w = C.WINDOW_SAMPLES
    segs = np.zeros((len(windows), w), dtype=np.float32)
    for i, (a, b) in enumerate(windows):
        seg = wav[a:b]
        segs[i] = seg if len(seg) == w else np.resize(seg, w)
    out = []
    for i in range(0, len(segs), batch):
        x = torch.from_numpy(segs[i:i + batch]).to(device)
        out.append(model.embed(logmel(x)).float().cpu())
    return torch.cat(out).numpy()


def _stable(x, windows):
    """Windows whose embedding agrees with its time-neighbours (i.e. not straddling a change)."""
    hop = int(C.INFER_HOP_SEC * SR)
    stab = np.ones(len(x))
    for i in range(len(x)):
        sims = [x[i] @ x[j] for j in (i - 2, i + 2)
                if 0 <= j < len(x) and abs(windows[j][0] - windows[i][0]) <= 3 * hop]
        if sims:
            stab[i] = min(sims)
    return stab


def diarize(wav, model, logmel, device, num_speakers=None):
    """wav: 16 kHz mono float32. Returns dict with `segments` [(speaker, start, end)] in
    seconds (speakers numbered 0.. by first appearance), `num_speakers`, `step`, `timeline`."""
    step = C.TIMELINE_STEP_SEC
    n_steps = int(np.ceil(len(wav) / SR / step))
    result = {"segments": [], "num_speakers": 0, "step": step, "timeline": np.full(n_steps, -1)}

    speech = rms_vad(wav)
    windows = _windows(speech)
    if not windows:
        return result
    x = normalize(extract_embeddings(wav, windows, model, logmel, device))
    dur = np.array([(b - a) / SR for a, b in windows])

    # Stage 5: spectral clustering on the reliable windows (long enough, not straddling a
    # speaker change); every window is then assigned to the nearest cluster centroid.
    core = dur >= C.MIN_CLUSTER_WINDOW_SEC
    if core.sum() >= 12:
        stab = _stable(x, windows)
        keep = core & (stab >= np.quantile(stab[core], 1 - C.STABLE_FRACTION))
        core = keep if keep.sum() >= 8 else core
    elif core.sum() < 4:
        core = np.ones(len(x), bool)
    labels_core = spectral_cluster(x[core], num_speakers)
    cents = centroids(x[core], labels_core)
    for _ in range(2):                                   # refine centroids on all windows
        labels = np.argmax(x @ cents.T, axis=1)
        good = core | (dur >= C.MIN_CLUSTER_WINDOW_SEC)
        cents = normalize(np.stack([
            x[good & (labels == c)].mean(0) if np.any(good & (labels == c)) else cents[c]
            for c in range(len(cents))]))
    labels = np.argmax(x @ cents.T, axis=1)
    k = len(cents)

    # Stage 6: majority voting over overlapping windows (votes weighted towards window centre)
    votes = np.zeros((n_steps, k))
    for (a, b), lab in zip(windows, labels):
        s, e = int(a / SR / step), min(n_steps, int(np.ceil(b / SR / step)))
        if e > s:
            votes[s:e, lab] += np.bartlett(e - s + 2)[1:-1] * min(1.0, (b - a) / C.WINDOW_SAMPLES)
    timeline = np.where(votes.sum(1) > 0, votes.argmax(1), -1)

    # Short pauses inside speech count as speech of the surrounding speaker.
    per = int(round(step / FRAME_HOP))
    lenient = _fill_short_runs(speech, False, int(C.VAD_BRIDGE_SEC / FRAME_HOP))
    pad = np.pad(lenient.astype(float), (0, max(0, n_steps * per - len(lenient))))[:n_steps * per]
    speech_steps = pad.reshape(n_steps, per).mean(axis=1) >= 0.5
    known = np.flatnonzero(timeline >= 0)
    for i in np.flatnonzero(speech_steps & (timeline < 0)):
        timeline[i] = timeline[known[np.argmin(np.abs(known - i))]]
    timeline[~speech_steps] = -1

    timeline = _smooth(timeline, int(round(C.MIN_SEGMENT_SEC / step)))

    # number speakers by first appearance
    order = []
    for v in timeline:
        if v >= 0 and v not in order:
            order.append(int(v))
    remap = {old: new for new, old in enumerate(order)}
    timeline = np.array([remap.get(int(v), -1) for v in timeline])

    segments = [(int(v), s * step, min(e * step, len(wav) / SR)) for s, e, v in _runs(timeline) if v >= 0]
    result.update(segments=segments, num_speakers=len(order), timeline=timeline)
    return result


def _smooth(timeline, min_steps):
    """Enforce a minimum speaker-turn duration: turns shorter than `min_steps` are absorbed
    by the neighbouring speaker (silence gaps are ignored when looking for neighbours)."""
    t = timeline.copy()
    for _ in range(3):
        speech_idx = np.flatnonzero(t >= 0)
        if len(speech_idx) == 0:
            break
        seq = t[speech_idx]
        runs = list(_runs(seq))
        changed = False
        for i, (s, e, v) in enumerate(runs):
            if e - s < min_steps and len(runs) > 1:
                prev_len = runs[i - 1][1] - runs[i - 1][0] if i > 0 else -1
                next_len = runs[i + 1][1] - runs[i + 1][0] if i < len(runs) - 1 else -1
                seq[s:e] = runs[i - 1][2] if prev_len >= next_len else runs[i + 1][2]
                changed = True
        t[speech_idx] = seq
        if not changed:
            break
    return t
