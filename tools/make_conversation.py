"""Build a multi-speaker test conversation from LibriSpeech dev-clean (speakers the model
never saw in training), with ground-truth speaker turns.

Needs data/LibriSpeech/dev-clean (python tools/download_dataset.py dev-clean).
The eight-file test set used in the README was built with:

    python tools/make_conversation.py --speakers 1 --turns 6  --seed 11 --out samples/monologue_1spk
    python tools/make_conversation.py --speakers 2 --turns 10 --seed 1  --out samples/conversation_2spk
    python tools/make_conversation.py --speakers 2 --turns 16 --seed 12 --out samples/conversation_2spk_b
    python tools/make_conversation.py --speakers 2 --turns 40 --seed 15 --out samples/conversation_2spk_long
    python tools/make_conversation.py --speakers 3 --turns 14 --seed 2  --out samples/conversation_3spk
    python tools/make_conversation.py --speakers 3 --turns 20 --seed 13 --noise 0.003 --out samples/conversation_3spk_noisy
    python tools/make_conversation.py --speakers 4 --turns 18 --seed 3  --noise 0.004 --out samples/conversation_4spk_noisy
    python tools/make_conversation.py --speakers 5 --turns 30 --seed 14 --out samples/conversation_5spk
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sid import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--speakers", type=int, default=2)
ap.add_argument("--turns", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--noise", type=float, default=0.0, help="background noise level, e.g. 0.005")
ap.add_argument("--out", default="samples/conversation")
args = ap.parse_args()

rng = random.Random(args.seed)
speakers = rng.sample(sorted(d.name for d in C.LIBRISPEECH_DEV.iterdir() if d.is_dir()), args.speakers)
# one recording session (chapter) per speaker, like a real conversation
files = {}
for s in speakers:
    chapters = sorted((C.LIBRISPEECH_DEV / s).iterdir())
    files[s] = sorted(max(chapters, key=lambda c: len(list(c.glob('*.flac')))).glob('*.flac'))
for s in speakers:
    rng.shuffle(files[s])

audio, truth, t, prev = [], [], 0.0, None
for _ in range(args.turns):
    spk = rng.choice([s for s in speakers if s != prev] or speakers)
    prev = spk
    wav, sr = sf.read(files[spk].pop(), dtype="float32")
    wav = wav[: int(sr * rng.uniform(4, 11))]
    gap = np.zeros(int(sr * rng.uniform(0.2, 0.7)), dtype=np.float32)
    truth.append({"speaker": spk, "start": round(t, 2), "end": round(t + len(wav) / sr, 2)})
    audio += [wav, gap]
    t += (len(wav) + len(gap)) / sr

mix = np.concatenate(audio)
if args.noise:
    mix = mix + np.random.default_rng(args.seed).normal(0, args.noise, len(mix)).astype(np.float32)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out.with_suffix(".wav"), mix, C.SAMPLE_RATE)
out.with_suffix(".json").write_text(json.dumps(truth, indent=1))
print(f"wrote {out.with_suffix('.wav')}  ({t:.1f} s, {args.speakers} speakers, {args.turns} turns)")
