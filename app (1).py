"""
AIGVDet Web Interface - Flask Backend (Google Colab version)
- Uses GPU (CUDA) — Colab T4/A100
- Served publicly via pyngrok tunnel
- Run this file from a Colab cell: exec(open('app.py').read())
  OR:  import app  (after %cd AIGVDet)
"""

import os
import sys
import uuid
import subprocess
import re
import shutil
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# ── Colab: start ngrok tunnel automatically ───────────────────────────────────
def start_ngrok(port: int = 5000):
    try:
        from pyngrok import ngrok, conf
        # If you have an ngrok auth token, set it here:
        # conf.get_default().auth_token = "YOUR_NGROK_TOKEN"
        tunnel = ngrok.connect(port)
        print("\n" + "="*60)
        print(f"  ✅  PUBLIC URL:  {tunnel.public_url}")
        print(f"  Share this link to access the detector from anywhere.")
        print("="*60 + "\n")
        return tunnel.public_url
    except Exception as e:
        print(f"[ngrok] Could not start tunnel: {e}")
        print(f"[ngrok] App still running at http://localhost:{port}")
        return None

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR       = Path(__file__).parent
UPLOAD_DIR     = BASE_DIR / "web_uploads"
FRAMES_DIR     = BASE_DIR / "web_frames"
OPTICAL_DIR    = BASE_DIR / "web_optical"

CHECKPOINT_OPT = BASE_DIR / "checkpoints" / "optical.pth"
CHECKPOINT_RGB = BASE_DIR / "checkpoints" / "original.pth"

MAX_UPLOAD_MB  = 500
USE_CPU        = False   # ← Colab has GPU; set True only if runtime has no GPU

# ── Check GPU availability ────────────────────────────────────────────────────
try:
    import torch
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available — switching to CPU mode.")
        USE_CPU = True
    else:
        print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    USE_CPU = True

# ── Create directories ────────────────────────────────────────────────────────
for d in [UPLOAD_DIR, FRAMES_DIR, OPTICAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# ── Job store ─────────────────────────────────────────────────────────────────
jobs: dict = {}
jobs_lock = threading.Lock()

def set_job(jid, **kwargs):
    with jobs_lock:
        jobs.setdefault(jid, {}).update(kwargs)

def get_job(jid):
    with jobs_lock:
        return dict(jobs.get(jid, {}))

def new_job_id():
    return uuid.uuid4().hex[:12]

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTS

# ── Parse demo.py output ──────────────────────────────────────────────────────
def parse_demo_output(stdout, stderr):
    combined = stdout + "\n" + stderr
    m = re.search(
        r"(?i)(predict(?:ion)?)[:\s]+(\w+).*?score[:\s]+([\d.]+)",
        combined,
    )
    if m:
        label = m.group(2).upper()
        score = float(m.group(3))
        return {
            "label": label,
            "score": round(score, 4),
            "is_ai": label == "FAKE",
            "confidence": round(score if label == "FAKE" else 1 - score, 4),
            "raw": combined.strip()[-3000:],
        }
    low = combined.lower()
    if "fake" in low:
        return {"label": "FAKE", "score": None, "is_ai": True,  "confidence": None, "raw": combined.strip()[-3000:]}
    if "real" in low:
        return {"label": "REAL", "score": None, "is_ai": False, "confidence": None, "raw": combined.strip()[-3000:]}
    return     {"label": "UNKNOWN","score": None,"is_ai": None, "confidence": None, "raw": combined.strip()[-3000:]}

# ── Core analysis worker ──────────────────────────────────────────────────────
def run_analysis(jid: str, video_path: Path):
    frame_dir   = FRAMES_DIR  / jid
    optical_dir = OPTICAL_DIR / jid
    frame_dir.mkdir(parents=True, exist_ok=True)
    optical_dir.mkdir(parents=True, exist_ok=True)

    set_job(jid, status="running", progress="Extracting frames & optical flow…")

    try:
        cmd = [
            sys.executable, "demo.py",          # uses same Python as Flask
            "--path",                    str(video_path),
            "--folder_original_path",    str(frame_dir),
            "--folder_optical_flow_path",str(optical_dir),
            "-mop", str(CHECKPOINT_OPT),
            "-mor", str(CHECKPOINT_RGB),
        ]
        if USE_CPU:
            cmd.append("--use_cpu")

        set_job(jid, progress="Running AIGVDet model on GPU…" if not USE_CPU else "Running AIGVDet model on CPU…")

        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,   # 5 min — GPU is fast
        )

        result = parse_demo_output(proc.stdout, proc.stderr)

        if proc.returncode != 0 and result["label"] == "UNKNOWN":
            set_job(jid,
                    status="error",
                    error=f"demo.py error (code {proc.returncode}): {proc.stderr[-400:]}")
        else:
            set_job(jid, status="done", result=result)

    except subprocess.TimeoutExpired:
        set_job(jid, status="error", error="Analysis timed out (>5 min).")
    except Exception as exc:
        set_job(jid, status="error", error=str(exc))
    finally:
        shutil.rmtree(frame_dir,   ignore_errors=True)
        shutil.rmtree(optical_dir, ignore_errors=True)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze/upload", methods=["POST"])
def analyze_upload():
    if "video" not in request.files:
        return jsonify(error="No file part"), 400
    f = request.files["video"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify(error="Unsupported file type"), 400

    jid  = new_job_id()
    ext  = Path(f.filename).suffix.lower()
    dest = UPLOAD_DIR / f"{jid}{ext}"
    f.save(dest)

    set_job(jid, status="queued", filename=f.filename)
    threading.Thread(target=run_analysis, args=(jid, dest), daemon=True).start()
    return jsonify(job_id=jid), 202

@app.route("/analyze/url", methods=["POST"])
def analyze_url():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify(error="No URL provided"), 400

    jid  = new_job_id()
    dest = UPLOAD_DIR / f"{jid}.mp4"
    set_job(jid, status="downloading", url=url)

    def download_and_run():
        set_job(jid, progress="Downloading video…")
        try:
            r = subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--merge-output-format", "mp4",
                    "--output", str(dest),
                    "--no-part",
                    url,
                ],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr[:500])
        except (FileNotFoundError, RuntimeError):
            import urllib.request
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                set_job(jid, status="error", error=f"Download failed: {e}")
                return

        # yt-dlp may append format ID — find the actual file
        if not dest.exists() or dest.stat().st_size == 0:
            actual = next(UPLOAD_DIR.glob(f"{jid}*"), None)
            if actual and actual != dest:
                actual.rename(dest)

        if not dest.exists() or dest.stat().st_size == 0:
            set_job(jid, status="error", error="Downloaded file is empty. Check the URL.")
            return

        run_analysis(jid, dest)

    threading.Thread(target=download_and_run, daemon=True).start()
    return jsonify(job_id=jid), 202

@app.route("/job/<jid>")
def job_status(jid):
    job = get_job(jid)
    if not job:
        return jsonify(error="Job not found"), 404
    return jsonify(job)

@app.route("/health")
def health():
    import torch
    missing = [str(p) for p in [CHECKPOINT_OPT, CHECKPOINT_RGB] if not p.exists()]
    return jsonify(
        status="ok" if not missing else "missing_checkpoints",
        missing_files=missing,
        use_cpu=USE_CPU,
        cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    ), 200 if not missing else 503

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_ngrok(5000)
    # use_reloader=False is critical in Colab — reloader breaks threading
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
