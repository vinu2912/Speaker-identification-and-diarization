"""Spectral clustering of speaker embeddings (paper III-D, Stage 5).

Cosine-similarity affinity, row normalisation, eigen-decomposition of the normalised graph
Laplacian, eigengap heuristic for the number of speakers, k-means in the spectral space.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

from . import config as C


def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _affinity(x, keep):
    """Cosine affinity, sparsified to each row's strongest neighbours, then row-normalised."""
    a = np.clip(x @ x.T, 0.0, 1.0)
    n = len(a)
    k = max(2, int(round(keep * n)))
    if k < n:
        thresh = np.partition(a, n - k, axis=1)[:, n - k][:, None]
        a = np.where(a >= thresh, a, a * 0.01)       # soft-suppress weak links
    a = 0.5 * (a + a.T)                               # keep it symmetric
    a = a / (a.max(axis=1, keepdims=True) + 1e-9)     # row normalisation
    return 0.5 * (a + a.T)


def centroids(x, labels):
    return normalize(np.stack([x[labels == c].mean(0) for c in range(int(labels.max()) + 1)]))


def merge_similar(x, labels, threshold):
    """Merge clusters whose centroids have cosine similarity above `threshold`
    (the same voice split in two). Returns relabelled 0..k-1."""
    labels = labels.copy()
    while labels.max() > 0:
        ids = np.unique(labels)
        if len(ids) < 2:
            break
        cents = normalize(np.stack([x[labels == c].mean(0) for c in ids]))
        sim = cents @ cents.T
        np.fill_diagonal(sim, -1.0)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        if sim[i, j] < threshold:
            break
        labels[labels == ids[j]] = ids[i]
    return np.unique(labels, return_inverse=True)[1]


def drop_tiny(x, labels, min_share=0.02):
    """Clusters holding under 2 % of the windows are noise, not a speaker: reassign them."""
    counts = np.bincount(labels)
    small = counts < max(2, min_share * len(labels))
    if not small.any() or small.all():
        return labels
    keep_ids = np.flatnonzero(~small)
    cents = normalize(np.stack([x[labels == c].mean(0) for c in keep_ids]))
    out = labels.copy()
    mask = small[labels]
    out[mask] = keep_ids[np.argmax(x[mask] @ cents.T, axis=1)]
    return np.unique(out, return_inverse=True)[1]


def spectral_cluster(emb, num_speakers=None, max_speakers=C.MAX_SPEAKERS, max_points=2500):
    """emb [N, D] -> labels [N] (0..k-1). With num_speakers=None the count is estimated
    by the eigengap heuristic, then clusters that are the same voice are merged."""
    n = len(emb)
    if n < 4 or num_speakers == 1:
        return np.zeros(n, dtype=int)
    x_all = normalize(emb)

    # Cluster an evenly spaced subset when the recording is very long, then assign the rest.
    idx = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype(int)
    x = x_all[idx]

    # Each window keeps its strongest neighbours: 30 % of the windows in a short clip, but
    # never more than AFFINITY_MAX_NEIGHBOURS, so that in a long multi-speaker recording a
    # window is not linked to other people's speech.
    a = _affinity(x, min(C.AFFINITY_KEEP, C.AFFINITY_MAX_NEIGHBOURS / len(x)))
    d_inv_sqrt = 1.0 / np.sqrt(a.sum(axis=1) + 1e-9)
    lap = np.eye(len(a)) - d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
    top = min(max_speakers + 1, len(a) - 1)
    vals, vecs = eigh(lap, subset_by_index=[0, top - 1])

    if num_speakers is None:
        k = max(2, int(np.argmax(np.diff(vals)[:max_speakers])) + 1)   # eigengap heuristic
    else:
        k = int(min(num_speakers, len(a)))

    labels = KMeans(k, n_init=10, random_state=0).fit_predict(normalize(vecs[:, :k]))
    if num_speakers is None:
        labels = merge_similar(x, labels, C.SAME_SPEAKER_COSINE)
        labels = drop_tiny(x, labels)

    if len(idx) == n:
        return labels
    return np.argmax(x_all @ centroids(x, labels).T, axis=1)
