"""Central configuration. Values follow the paper (Section III) unless noted."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LIBRISPEECH_TRAIN = DATA_DIR / "LibriSpeech" / "train-clean-100"
LIBRISPEECH_DEV = DATA_DIR / "LibriSpeech" / "dev-clean"
CHECKPOINT_DIR = ROOT / "checkpoints"
BEST_MODEL = CHECKPOINT_DIR / "cnn_gru_best.pt"
REPORT_DIR = ROOT / "reports"

# ---- Audio / features (III-A) ----
SAMPLE_RATE = 16_000
WINDOW_SEC = 3.0            # sliding window length
TRAIN_HOP_SEC = 1.5         # 50% overlap
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
N_FFT = 512                 # 512-point STFT, Hann window
HOP_LENGTH = 256            # 50% hop
N_MELS = 80                 # 80 mel bands, 0-8 kHz
N_FRAMES = WINDOW_SAMPLES // HOP_LENGTH + 1   # 188 frames per 3 s window

# ---- Model (III-B) ----
EMBEDDING_DIM = 768
GRU_HIDDEN = 256
N_TRAIN_SPEAKERS = 251

# ---- Training (III-C) ----
EPOCHS = 10
BATCH_SIZE = 32             # segments per batch
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 3
CROPS_PER_UTTERANCE = 4     # random 3 s crops taken from each utterance per epoch
TIME_STRETCH = 0.10         # +-10 %
PITCH_SEMITONES = 2.0       # +-2 semitones
SEED = 1234

# ---- Diarization (III-D) ----
INFER_HOP_SEC = 0.5         # denser than training hop -> finer speaker boundaries
VAD_RMS_THRESHOLD = 0.02    # paper's RMS threshold (applied after loudness normalisation)
VAD_REFERENCE_LEVEL = 0.4   # loud speech frames are scaled to this RMS before thresholding
VAD_MIN_PAUSE_SEC = 0.2     # pauses shorter than this do not split a stretch of speech
VAD_BRIDGE_SEC = 0.5        # pauses shorter than this stay attributed to the speaker
MIN_SEGMENT_SEC = 0.5
MAX_SPEAKERS = 8
AFFINITY_KEEP = 0.3         # each window keeps its strongest 30 % affinities ...
AFFINITY_MAX_NEIGHBOURS = 80  # ... but at most this many (matters for long recordings)
SAME_SPEAKER_COSINE = 0.87  # two cluster centroids closer than this = one speaker
MIN_CLUSTER_WINDOW_SEC = 1.5  # shorter speech stretches are assigned, not clustered
STABLE_FRACTION = 0.7       # share of windows (most self-consistent) used for clustering
TIMELINE_STEP_SEC = 0.1

# ---- Transcription / summarisation ----
WHISPER_MODEL = "large-v3-turbo"
SUMMARY_MODEL = "philschmid/bart-large-cnn-samsum"
