"""End-to-end: audio file -> speaker timeline + speaker-attributed transcript + summary."""
import re
import threading

import numpy as np
import torch

from . import config as C
from .audio import load_audio
from .diarize import diarize
from .model import load_encoder
from .summarize import summarize_dialogue
from .transcribe import transcribe

_encoder = None
_lock = threading.Lock()


def get_encoder():
    global _encoder
    if _encoder is None:
        if not C.BEST_MODEL.exists():
            raise RuntimeError("No trained CNN-GRU model found. Run `python train.py` first.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _encoder = (*load_encoder(C.BEST_MODEL, device), device)
    return _encoder


def _phrases(words, max_pause=0.5):
    """Split the word stream at sentence punctuation, commas and pauses."""
    phrase = []
    for i, w in enumerate(words):
        phrase.append(w)
        last = i == len(words) - 1
        pause = (not last) and words[i + 1]["start"] - w["end"] > max_pause
        if last or pause or re.search(r"[.?!,;:]$", w["text"].strip()):
            yield phrase
            phrase = []


def _speaker_for(timeline, step, start, end):
    """Speaker with the most talk time inside [start, end] (nearest labelled step if none)."""
    a = int(start / step)
    b = max(a + 1, int(np.ceil(end / step)))
    seg = timeline[a:b]
    seg = seg[seg >= 0]
    if len(seg):
        return int(np.bincount(seg).argmax())
    known = np.flatnonzero(timeline >= 0)
    if len(known) == 0:
        return 0
    # The word sits in a pause. Whisper tends to start a word slightly early, so a word in
    # the pause just before speech resumes belongs to the speaker who talks next.
    ahead = known[(known >= b) & (known < b + int(0.6 / step))]
    if len(ahead):
        return int(timeline[ahead[0]])
    return int(timeline[known[np.argmin(np.abs(known - b))]])


def _split_by_speaker(phrase, timeline, step, min_run_sec=1.0):
    """Yield (speaker, words) pieces of one phrase. A phrase normally has one speaker; if the
    timeline shows a real speaker change inside it (e.g. Whisper gave no punctuation), it is
    cut at the word where the speaker changes. Runs shorter than `min_run_sec` are absorbed."""
    whole = _speaker_for(timeline, step, phrase[0]["start"], phrase[-1]["end"])
    if phrase[-1]["end"] - phrase[0]["start"] < 2 * min_run_sec:
        yield whole, phrase
        return
    # Label each word by its last 0.3 s: Whisper can stretch a word's start back across a
    # pause, but the word's end is reliable.
    labels = [_speaker_for(timeline, step, max(w["start"], w["end"] - 0.3), w["end"]) for w in phrase]
    for _ in range(len(phrase)):
        runs, start = [], 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                runs.append((start, i))
                start = i
        short = [(phrase[e - 1]["end"] - phrase[s]["start"], s, e) for s, e in runs
                 if phrase[e - 1]["end"] - phrase[s]["start"] < min_run_sec]
        if len(runs) == 1 or not short:
            break
        _, s, e = min(short)                          # absorb the shortest run into a neighbour
        labels[s:e] = [labels[s - 1] if s > 0 else labels[e]] * (e - s)
    for s, e in runs:
        yield labels[s], phrase[s:e]


def attribute(words, dia):
    """Combine Whisper words with the speaker timeline -> utterances [{speaker, start, end, text}]."""
    utterances = []
    pieces = [p for phrase in _phrases(words)
              for p in _split_by_speaker(phrase, dia["timeline"], dia["step"])]
    for spk, phrase in pieces:
        text = "".join(w["text"] for w in phrase).strip()
        prev = utterances[-1] if utterances else None
        ends_sentence = prev and re.search(r"[.?!]$", prev["text"])
        long_turn = prev and len(prev["text"].split()) > 70 and ends_sentence
        if prev and prev["speaker"] == spk and not long_turn:
            prev["text"] += " " + text
            prev["end"] = phrase[-1]["end"]
        else:
            utterances.append({"speaker": spk, "start": phrase[0]["start"],
                               "end": phrase[-1]["end"], "text": text})
    return utterances


def process(path, num_speakers=None, language=None, progress=lambda stage, frac: None):
    """Run the whole system on one audio file. `progress(stage, fraction)` is optional."""
    with _lock:
        progress("Loading audio", 0.0)
        wav = load_audio(path)
        duration = len(wav) / C.SAMPLE_RATE

        progress("Identifying speakers (CNN-GRU + spectral clustering)", 0.0)
        model, logmel, device = get_encoder()
        dia = diarize(wav, model, logmel, device, num_speakers)

        progress("Transcribing (Whisper)", 0.0)
        words, lang = transcribe(wav, language, lambda f: progress("Transcribing (Whisper)", f))

        utterances = attribute(words, dia)
        # Speakers that never got a transcript line still appear in the timeline; number
        # speakers consistently as "Speaker N" (N = order of first appearance).
        n_speakers = max(dia["num_speakers"], 1 + max((u["speaker"] for u in utterances), default=0))
        talk = [0.0] * n_speakers
        for spk, s, e in dia["segments"]:
            talk[spk] += e - s

        progress("Summarising", 0.0)
        summary, sections = summarize_dialogue(utterances, lambda f: progress("Summarising", f))
        progress("Done", 1.0)

    return {
        "duration": duration,
        "language": lang,
        "num_speakers": n_speakers,
        "speakers": [{"id": i, "name": f"Speaker {i + 1}", "talk_time": round(talk[i], 2)}
                     for i in range(n_speakers)],
        "segments": [{"speaker": spk, "start": round(s, 2), "end": round(e, 2)}
                     for spk, s, e in dia["segments"]],
        "utterances": [{"speaker": u["speaker"], "start": round(u["start"], 2),
                        "end": round(u["end"], 2), "text": u["text"]} for u in utterances],
        "summary": summary,
        "summary_sections": sections,
    }
