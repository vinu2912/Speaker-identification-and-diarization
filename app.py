"""Web app: upload an audio file -> speaker-coloured transcript + summary.

    python app.py        then open http://127.0.0.1:8000
"""
import threading
import traceback
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from sid import config as C
from sid.pipeline import process

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="Speaker Identification and Summarization")
jobs = {}


def _run(job_id, path, num_speakers):
    job = jobs[job_id]

    def progress(stage, frac):
        job["stage"], job["progress"] = stage, round(float(frac), 3)

    try:
        job["result"] = process(path, num_speakers, progress=progress)
        job["status"] = "done"
    except Exception as exc:  # report the failure to the page instead of hanging
        traceback.print_exc()
        job["status"], job["error"] = "error", str(exc) or exc.__class__.__name__
    finally:
        path.unlink(missing_ok=True)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"model_trained": C.BEST_MODEL.exists()}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), num_speakers: int = Form(0)):
    if not C.BEST_MODEL.exists():
        raise HTTPException(503, "The CNN-GRU model has not been trained yet. Run `python train.py`.")
    job_id = uuid.uuid4().hex
    path = UPLOADS / (job_id + Path(file.filename or "audio").suffix[:8])
    with open(path, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    jobs[job_id] = {"status": "running", "stage": "Queued", "progress": 0.0}
    n = num_speakers if 1 <= num_speakers <= C.MAX_SPEAKERS else None
    threading.Thread(target=_run, args=(job_id, path, n), daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[job_id]


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
