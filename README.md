# PestAPI (Flask + YOLOv8)

This repository contains a small Flask API that runs a YOLOv8 model and returns detections for uploaded images.

## Files
- `app.py` - main Flask app with `/predict` endpoint and background file cleanup
- `wsgi.py` - gunicorn entrypoint
- `requirements.txt` - Python dependencies
- `Procfile` - for Render/Gunicorn
- `runtime.txt` - Python runtime pin
- `best.pt` - your trained YOLOv8 weights (place here)
- `static/` - annotated images saved here

## Endpoints
- `GET /health` - health check
- `POST /predict` - accepts `multipart/form-data` with key `image` (JPEG/PNG). Returns JSON:
  ```json
  {
    "detections": [
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.98,
        "class": 0,
        "label": "pest_label"
      }
    ],
    "image_url": "https://your-host/static/<uuid>.jpg"
  }
