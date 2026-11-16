"""Speech-to-text with OpenAI Whisper (faster-whisper runtime), word-level timestamps."""
import os
from pathlib import Path

import torch

from . import config as C

_model = None


def _load():
    global _model
    if _model is None:
        # faster-whisper needs the cuBLAS / cuDNN DLLs; PyTorch already ships them.
        torch_lib = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        from faster_whisper import WhisperModel
        if torch.cuda.is_available():
            _model = WhisperModel(C.WHISPER_MODEL, device="cuda", compute_type="float16")
        else:
            _model = WhisperModel(C.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe(wav, language=None, progress=None):
    """wav: 16 kHz mono float32 -> (words [{start, end, text}], language code)."""
    model = _load()
    segments, info = model.transcribe(
        wav, language=language, word_timestamps=True, vad_filter=True,
        beam_size=5, condition_on_previous_text=False,
    )
    words = []
    total = max(len(wav) / C.SAMPLE_RATE, 1e-6)
    for seg in segments:
        for w in seg.words or []:
            if w.word.strip():
                words.append({"start": float(w.start), "end": float(w.end), "text": w.word})
        if progress:
            progress(min(seg.end / total, 1.0))
    return words, info.language
