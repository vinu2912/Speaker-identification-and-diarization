"""Evaluation as in the paper (Section IV/V):

  * speaker-verification EER from pairwise cosine similarity of embeddings
    (LibriSpeech held-out test split, and dev-clean = speakers never seen in training)
  * diarization error (missed / false alarm / confusion) on the sample conversations

    python evaluate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from sid import config as C
from sid.audio import load_audio
from sid.data import CropDataset, collate, list_utterances, split_utterances
from sid.diarize import diarize
from sid.model import load_encoder


@torch.no_grad()
def embed_items(items, model, logmel, device, crops=2):
    loader = DataLoader(CropDataset(items, C.WINDOW_SAMPLES, crops, False), batch_size=64,
                        collate_fn=collate, num_workers=6)
    embs, labels = [], []
    for wav, lab in loader:
        embs.append(model.embed(logmel(wav.to(device))).cpu())
        labels.append(lab)
    return torch.cat(embs), torch.cat(labels)


def eer_from_embeddings(emb, labels, device, plot_path=None, title=""):
    emb = torch.nn.functional.normalize(emb.to(device), dim=1)
    sim = emb @ emb.T
    iu = torch.triu_indices(len(emb), len(emb), offset=1, device=device)
    scores = sim[iu[0], iu[1]]
    same = (labels.to(device)[iu[0]] == labels.to(device)[iu[1]])
    thresholds = torch.linspace(-1, 1, 2001, device=device)
    genuine, impostor = scores[same], scores[~same]
    g_hist = torch.histc(genuine, bins=2000, min=-1, max=1)
    i_hist = torch.histc(impostor, bins=2000, min=-1, max=1)
    frr = torch.cat([torch.zeros(1, device=device), g_hist.cumsum(0)]) / len(genuine)   # genuine below thr
    far = 1 - torch.cat([torch.zeros(1, device=device), i_hist.cumsum(0)]) / len(impostor)
    k = int(torch.argmin((far - frr).abs()))
    eer, thr = float((far[k] + frr[k]) / 2), float(thresholds[k])
    if plot_path:
        plt.figure(figsize=(6, 4))
        plt.plot(thresholds.cpu(), far.cpu() * 100, "b", label="FAR")
        plt.plot(thresholds.cpu(), frr.cpu() * 100, "r", label="FRR")
        plt.axvline(thr, color="gray", ls="--", lw=1)
        plt.scatter([thr], [eer * 100], color="k", zorder=3, label=f"EER {eer * 100:.2f}% @ {thr:.3f}")
        plt.xlim(0, 1); plt.xlabel("Cosine similarity threshold"); plt.ylabel("Error rate (%)")
        plt.title(title); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(plot_path, dpi=150); plt.close()
    return eer, thr, int(len(scores))


def diarization_error(pred_timeline, truth, step):
    """Frame-level DER with optimal speaker mapping. -> dict of percentages."""
    n = len(pred_timeline)
    ref = np.full(n, -1)
    names = sorted({t["speaker"] for t in truth})
    for t in truth:
        ref[int(round(t["start"] / step)):int(round(t["end"] / step))] = names.index(t["speaker"])
    k_ref, k_hyp = len(names), int(pred_timeline.max()) + 1
    overlap = np.zeros((k_ref, max(k_hyp, 1)))
    for r in range(k_ref):
        for h in range(k_hyp):
            overlap[r, h] = np.sum((ref == r) & (pred_timeline == h))
    rows, cols = linear_sum_assignment(-overlap)
    correct = overlap[rows, cols].sum()
    total = np.sum(ref >= 0)
    miss = np.sum((ref >= 0) & (pred_timeline < 0))
    fa = np.sum((ref < 0) & (pred_timeline >= 0))
    conf = total - miss - correct
    return {"missed": 100 * miss / total, "false_alarm": 100 * fa / total,
            "confusion": 100 * conf / total, "DER": 100 * (miss + fa + conf) / total,
            "speakers_true": k_ref, "speakers_found": k_hyp}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, logmel = load_encoder(C.BEST_MODEL, device)
    C.REPORT_DIR.mkdir(exist_ok=True)
    report = {}

    if C.LIBRISPEECH_TRAIN.exists():
        items, _ = list_utterances(C.LIBRISPEECH_TRAIN)
        _, _, test_items = split_utterances(items)
        emb, lab = embed_items(test_items, model, logmel, device)
        eer, thr, pairs = eer_from_embeddings(emb, lab, device, C.REPORT_DIR / "eer_in_domain.png",
                                              "FAR / FRR - LibriSpeech test split (training speakers)")
        print(f"In-domain  EER {eer * 100:.2f}% | threshold {thr:.4f} | {pairs:,} pairs")
        report["eer_in_domain"] = {"eer_percent": eer * 100, "threshold": thr, "pairs": pairs}

    if C.LIBRISPEECH_DEV.exists():
        items, _ = list_utterances(C.LIBRISPEECH_DEV)
        emb, lab = embed_items(items, model, logmel, device)
        eer, thr, pairs = eer_from_embeddings(emb, lab, device, C.REPORT_DIR / "eer_unseen_speakers.png",
                                              "FAR / FRR - dev-clean (unseen speakers)")
        print(f"Unseen-spk EER {eer * 100:.2f}% | threshold {thr:.4f} | {pairs:,} pairs")
        report["eer_unseen_speakers"] = {"eer_percent": eer * 100, "threshold": thr, "pairs": pairs}

    for truth_file in sorted(Path(C.ROOT / "samples").glob("*.json")):
        if not truth_file.with_suffix(".wav").exists():
            print(f"{truth_file.stem:28s} audio not present - recreate it with tools/make_conversation.py")
            continue
        wav = load_audio(truth_file.with_suffix(".wav"))
        dia = diarize(wav, model, logmel, device)
        err = diarization_error(dia["timeline"], json.loads(truth_file.read_text()), dia["step"])
        print(f"{truth_file.stem:28s} DER {err['DER']:5.2f}% | miss {err['missed']:.2f} FA {err['false_alarm']:.2f} "
              f"conf {err['confusion']:.2f} | speakers {err['speakers_found']}/{err['speakers_true']}")
        report[f"der_{truth_file.stem}"] = err

    (C.REPORT_DIR / "evaluation.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
