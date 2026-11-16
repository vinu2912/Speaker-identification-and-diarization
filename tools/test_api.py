"""Send an audio file to the running app and print the result (python tools/test_api.py file [n_speakers])."""
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

path = Path(sys.argv[1])
n = sys.argv[2] if len(sys.argv) > 2 else "0"
base = "http://127.0.0.1:8000"
b = uuid.uuid4().hex
body = (f"--{b}\r\nContent-Disposition: form-data; name=\"num_speakers\"\r\n\r\n{n}\r\n"
        f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        "Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{b}--\r\n".encode()
req = urllib.request.Request(base + "/api/jobs", data=body,
                             headers={"Content-Type": f"multipart/form-data; boundary={b}"})
job = json.load(urllib.request.urlopen(req))["id"]
t0 = time.time()
while True:
    r = json.load(urllib.request.urlopen(f"{base}/api/jobs/{job}"))
    if r["status"] != "running":
        break
    time.sleep(1)
print(f"status {r['status']} in {time.time() - t0:.1f}s")
if r["status"] == "error":
    sys.exit(r["error"])
res = r["result"]
print(f"duration {res['duration']:.1f}s | language {res['language']} | speakers {res['num_speakers']}")
for u in res["utterances"]:
    print(f"[{u['start']:6.1f}] Speaker {u['speaker'] + 1}: {u['text']}")
print("\nSUMMARY:", res["summary"])
for s in res.get("summary_sections", []):
    print(f"  [{s['start'] / 60:5.1f}-{s['end'] / 60:5.1f} min] {s['text']}")
print("TALK TIME:", {s["name"]: round(s["talk_time"] / 60, 1) for s in res["speakers"]})
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(res))
