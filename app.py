# app.py
"""
PestAPI - Flask + YOLOv8 inference server
- /health            GET  -> basic health check
- /predict           POST -> multipart/form-data with "image" -> returns detections + absolute image_url
- /static/<file>     GET  -> serve annotated images
Background thread: periodic cleanup of old files in UPLOAD_DIR.
"""

from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import os
import uuid
import time
import threading
import logging

# Optional: limit torch threads on small/free hosts
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# --- Configuration (via env vars) ---
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
UPLOAD_DIR = os.environ.get("UPLOAD_FOLDER", "static")
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("CLEANUP_INTERVAL_SECONDS", 60 * 60 * 6))  # default 6 hours
FILE_MAX_AGE_SECONDS = int(os.environ.get("FILE_MAX_AGE_SECONDS", 60 * 60 * 24 * 2))  # default 48 hours

os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Flask app ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PestAPI")

# --- Load model once at startup ---
logger.info("Loading model from: %s", MODEL_PATH)
model = YOLO(MODEL_PATH)
logger.info("Model loaded.")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided to API"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        logger.exception("Invalid image uploaded")
        return jsonify({"error": "Invalid image"}), 400

    # Run inference
    try:
        results = model(image)
    except Exception as e:
        logger.exception("Model inference failed")
        return jsonify({"error": "Model inference failed", "detail": str(e)}), 500

    # Draw bounding boxes and collect detections
    draw = ImageDraw.Draw(image)
    detections = []
    try:
        for result in results:
            # Each result.boxes.data is tensor Nx6 with [x1,y1,x2,y2,conf,cls]
            for box in result.boxes.data.tolist():
                x_min, y_min, x_max, y_max, conf, cls = box
                label = model.names[int(cls)]
                det = {
                    "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                    "confidence": float(conf),
                    "class": int(cls),
                    "label": label
                }
                detections.append(det)

                # Draw on image
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                draw.text((x_min, y_min), f"{label} {conf:.2f}", fill="red")
    except Exception:
        # If drawing fails, continue but log
        logger.exception("Failed while parsing/drawing results")

    # Save annotated image with unique filename
    filename = f"{uuid.uuid4()}.jpg"
    out_path = os.path.join(UPLOAD_DIR, filename)
    try:
        image.save(out_path)
    except Exception:
        logger.exception("Failed to save annotated image")
        return jsonify({"error": "Failed to save image"}), 500

    # Build absolute URL (uses request host)
    image_url = url_for("serve_static", filename=filename, _external=True)

    response = {"detections": detections, "image_url": image_url}
    return jsonify(response), 200

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# --- Background cleanup thread ---
def cleanup_worker():
    logger.info("Cleanup worker started. interval=%s seconds, max_age=%s seconds",
                CLEANUP_INTERVAL_SECONDS, FILE_MAX_AGE_SECONDS)
    while True:
        try:
            now = time.time()
            removed = 0
            for fname in os.listdir(UPLOAD_DIR):
                fpath = os.path.join(UPLOAD_DIR, fname)
                if not os.path.isfile(fpath):
                    continue
                age = now - os.path.getmtime(fpath)
                if age > FILE_MAX_AGE_SECONDS:
                    try:
                        os.remove(fpath)
                        removed += 1
                    except Exception:
                        logger.exception("Failed to remove file: %s", fpath)
            if removed:
                logger.info("Cleanup removed %d files", removed)
        except Exception:
            logger.exception("Cleanup worker error")
        time.sleep(CLEANUP_INTERVAL_SECONDS)

def start_cleanup_thread():
    t = threading.Thread(target=cleanup_worker, daemon=True)
    t.start()

# Start cleanup thread when app starts
start_cleanup_thread()

if __name__ == "__main__":
    logger.info("Starting dev server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
