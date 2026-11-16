"""Download and unpack the LibriSpeech subsets used by this project into data/.

    python tools/download_dataset.py            # train-clean-100 (6.3 GB) + dev-clean (0.3 GB)
    python tools/download_dataset.py dev-clean  # only the small evaluation subset

Only needed for training (train.py) and evaluation (evaluate.py); the app itself runs
from the trained model in checkpoints/.
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
BASE = "https://www.openslr.org/resources/12/"
subsets = sys.argv[1:] or ["dev-clean", "train-clean-100"]

DATA.mkdir(exist_ok=True)
for name in subsets:
    target = DATA / "LibriSpeech" / name
    if target.exists():
        print(f"{name}: already present, skipping")
        continue
    archive = DATA / f"{name}.tar.gz"
    if not archive.exists():
        print(f"{name}: downloading {BASE}{name}.tar.gz")

        def report(blocks, block_size, total):
            done = blocks * block_size
            if blocks % 2000 == 0:
                print(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)

        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE}{name}.tar.gz", tmp, report)
        tmp.rename(archive)
    print(f"{name}: extracting")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(DATA, filter="data")
        except TypeError:                      # Python without extraction filters
            tar.extractall(DATA)
    archive.unlink()
    print(f"{name}: ready at {target}")
