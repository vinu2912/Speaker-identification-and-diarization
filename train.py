"""Train the CNN-GRU speaker encoder on LibriSpeech train-clean-100 (paper Section III-C).

    python train.py            # full run, 10 epochs
    python train.py --epochs 1 --limit 400   # quick smoke test
"""
import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sid import config as C
from sid.data import CropDataset, collate, list_utterances, split_utterances
from sid.features import LogMel
from sid.model import CNNGRU

# Raw crop is longer than 3 s so that a time-stretched / pitch-shifted crop still fills 3 s.
RAW_CROP = int(C.WINDOW_SAMPLES * (1 + C.TIME_STRETCH) * 1.03)


def augment_params():
    """Random pitch shift (+-2 semitones) and time stretch (+-10 %); ~40 % left clean."""
    if random.random() < 0.4:
        return 0.0, 1.0
    return (random.uniform(-C.PITCH_SEMITONES, C.PITCH_SEMITONES),
            random.uniform(1 - C.TIME_STRETCH, 1 + C.TIME_STRETCH))


def run_epoch(model, logmel, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    count, step, t0 = 0, 0, time.time()
    for wav_all, labels_all in loader:
        wav_all = wav_all.to(device, non_blocking=True)
        labels_all = labels_all.to(device, non_blocking=True)
        # The loader returns CROPS_PER_UTTERANCE crops for each utterance. For training, each
        # mini-batch takes ONE crop from each of 32 different utterances, so batch-norm
        # statistics are not dominated by a handful of speakers.
        k = C.CROPS_PER_UTTERANCE if train else 1
        for j in range(k):
            wav, labels = wav_all[j::k], labels_all[j::k]
            if train:
                # four groups per batch, each with its own random pitch shift / time stretch
                mels = []
                for part in wav.chunk(4):
                    pitch, stretch = augment_params()
                    if pitch == 0.0 and stretch == 1.0:
                        part = part[:, :C.WINDOW_SAMPLES]
                    mels.append(logmel(part, pitch_semitones=pitch, stretch=stretch))
                mel = torch.cat(mels)
            else:
                mel = logmel(wav)
            with torch.set_grad_enabled(train):
                logits = model(mel)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.detach() * labels.numel()
            correct += (logits.argmax(1) == labels).sum()
            count += labels.numel()
            step += 1
            if train and step % 500 == 0:
                print(f"    step {step} | loss {total_loss.item() / count:.3f} | "
                      f"{step / (time.time() - t0):.1f} it/s", flush=True)
    return total_loss.item() / count, 100.0 * correct.item() / count


@torch.no_grad()
def recalibrate_batchnorm(model, logmel, loader, device):
    """Recompute batch-norm running statistics on clean training audio with dropout off.
    During training the statistics are collected under dropout and augmentation, which
    does not match inference conditions; this makes eval-mode behaviour match."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    model.eval()
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None          # cumulative average over all calibration batches
        bn.train()
    for wav_all, _ in loader:
        wav_all = wav_all.to(device)
        for j in range(4):
            model(logmel(wav_all[j::4]))
    for bn in bns:
        bn.momentum = 0.1
    model.eval()


def plot_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    for key, ylabel, name in (("loss", "Loss", "loss_curves.png"), ("acc", "Accuracy (%)", "accuracy_curves.png")):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history[f"train_{key}"], "o-", label="Training")
        plt.plot(epochs, history[f"val_{key}"], "s-", label="Validation")
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.grid(alpha=0.3); plt.legend()
        plt.title(f"Training and validation {ylabel.lower()}")
        plt.tight_layout(); plt.savefig(out_dir / name, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="use only N utterances (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    items, speakers = list_utterances(C.LIBRISPEECH_TRAIN)
    if args.limit:
        random.Random(0).shuffle(items)
        items = items[:args.limit]
    train_items, val_items, test_items = split_utterances(items)
    print(f"{len(speakers)} speakers | utterances: train {len(train_items)}, val {len(val_items)}, "
          f"test {len(test_items)} | device {device}", flush=True)

    utts_per_batch = C.BATCH_SIZE          # x CROPS_PER_UTTERANCE crops -> 4 batches of 32
    kw = dict(collate_fn=collate, num_workers=args.workers, pin_memory=True,
              persistent_workers=args.workers > 0)
    train_loader = DataLoader(CropDataset(train_items, RAW_CROP, C.CROPS_PER_UTTERANCE, True),
                              batch_size=utts_per_batch, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(CropDataset(val_items, C.WINDOW_SAMPLES, 4, False), batch_size=32, **kw)
    calib_items = random.Random(1).sample(train_items, min(1600, len(train_items)))
    calib_loader = DataLoader(CropDataset(calib_items, C.WINDOW_SAMPLES, 4, False), batch_size=32,
                              shuffle=True, drop_last=True, collate_fn=collate, num_workers=4)
    test_loader =DataLoader(CropDataset(test_items, C.WINDOW_SAMPLES, 4, False), batch_size=32, **kw)

    model = CNNGRU(n_speakers=len(speakers)).to(device)
    logmel = LogMel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)

    C.CHECKPOINT_DIR.mkdir(exist_ok=True); C.REPORT_DIR.mkdir(exist_ok=True)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val, bad_epochs = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, logmel, train_loader, device, criterion, optimizer)
        train_bn = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k}
        recalibrate_batchnorm(model, logmel, calib_loader, device)
        va_loss, va_acc = run_epoch(model, logmel, val_loader, device, criterion)
        calibrated = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(train_bn, strict=False)       # keep training on its own statistics
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)
        print(f"epoch {epoch:2d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc:.2f}% | {time.time() - t0:.0f}s", flush=True)
        if va_loss < best_val:
            best_val, bad_epochs = va_loss, 0
            torch.save({"model": calibrated, "n_speakers": len(speakers),
                        "speakers": speakers, "epoch": epoch, "val_acc": va_acc}, C.BEST_MODEL)
        else:
            bad_epochs += 1
            if bad_epochs >= C.EARLY_STOP_PATIENCE:
                print("early stopping", flush=True)
                break
        (C.REPORT_DIR / "history.json").write_text(json.dumps(history, indent=1))
        plot_curves(history, C.REPORT_DIR)

    model.load_state_dict(torch.load(C.BEST_MODEL, map_location=device, weights_only=True)["model"])
    te_loss, te_acc = run_epoch(model, logmel, test_loader, device, criterion)
    print(f"TEST (best checkpoint) | loss {te_loss:.4f} acc {te_acc:.2f}%", flush=True)
    history["test_loss"], history["test_acc"] = te_loss, te_acc
    (C.REPORT_DIR / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
