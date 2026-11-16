"""Summarise a speaker-attributed transcript with a dialogue-summarisation BART model."""
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

_tok = _model = None
MAX_INPUT_TOKENS = 900


def _load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(C.SUMMARY_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(C.SUMMARY_MODEL)
        _model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        # lengths are set per call with max_new_tokens / min_new_tokens
        _model.generation_config.max_length = None
        _model.generation_config.min_length = None
    return _tok, _model


def _n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _tidy(text):
    """The model likes to open with a roll-call ('Speaker 1, Speaker 3, Speaker 4 and Speaker 2
    are ...'); replace lists of three or more speakers with 'The participants'."""
    return re.sub(r"Speaker \d+(?:, Speaker \d+)+,? and Speaker \d+", "The participants", text)


@torch.no_grad()
def _summarise(text, max_tokens=200):
    tok, model = _load()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    n_in = enc["input_ids"].shape[1]
    max_new = int(min(max_tokens, max(40, n_in * 0.45)))
    min_new = int(min(60, max(8, n_in * 0.12)))
    out = model.generate(**enc, num_beams=4, max_new_tokens=max_new, min_new_tokens=min_new,
                         no_repeat_ngram_size=3, length_penalty=1.0, early_stopping=True)
    return _tidy(tok.decode(out[0], skip_special_tokens=True).strip())


def _chunks(tok, utterances):
    """Group consecutive utterances into chunks that fit the model's input window.
    Yields (start_sec, end_sec, text)."""
    chunk, size = [], 0

    def flush():
        return chunk[0]["start"], chunk[-1]["end"], "\n".join(u["line"] for u in chunk)

    for u in utterances:
        n = _n_tokens(tok, u["line"]) + 1
        if n > MAX_INPUT_TOKENS:                       # one very long monologue: split by words
            if chunk:
                yield flush()
                chunk, size = [], 0
            words = u["line"].split()
            pieces = [words[i:i + 500] for i in range(0, len(words), 500)]
            span = (u["end"] - u["start"]) / len(pieces)
            for j, piece in enumerate(pieces):
                yield u["start"] + j * span, u["start"] + (j + 1) * span, " ".join(piece)
            continue
        if chunk and size + n > MAX_INPUT_TOKENS:
            yield flush()
            chunk, size = [], 0
        chunk.append(u)
        size += n
    if chunk:
        yield flush()


def summarize_dialogue(utterances, progress=None):
    """utterances: [{speaker, start, end, text}] -> (overview, sections).

    The transcript is cut into consecutive parts that fit the model; each part gets its own
    summary, so every stretch of the recording is covered. `sections` is
    [{start, end, text}] (empty for short recordings); `overview` condenses the beginning,
    middle and end separately so that later topics are not dropped."""
    utts = [dict(u, line=f"Speaker {u['speaker'] + 1}: {u['text']}") for u in utterances if u["text"].strip()]
    if not utts:
        return "", []
    if sum(len(u["text"].split()) for u in utts) < 25:
        return "The recording is too short to summarise: " + " ".join(u["line"] for u in utts), []
    tok, _ = _load()
    parts = list(_chunks(tok, utts))
    sections = []
    for i, (start, end, text) in enumerate(parts):
        sections.append({"start": round(start, 2), "end": round(end, 2), "text": _summarise(text)})
        if progress:
            progress((i + 1) / (len(parts) + 1))
    if len(sections) <= 3:
        return " ".join(s["text"] for s in sections), []
    # Overview: the lead sentence of every part, so no stretch of the recording is dropped
    # (for very long recordings, an evenly spaced selection of 12 parts).
    picked = sections if len(sections) <= 12 else [sections[round(i * (len(sections) - 1) / 11)] for i in range(12)]
    leads = [re.split(r"(?<=[.!?])\s+", s["text"].strip())[0] for s in picked]
    leads = leads[:1] + [re.sub(r"^The participants", "They", l) for l in leads[1:]]
    overview = " ".join(leads)
    return overview, sections