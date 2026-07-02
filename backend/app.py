import json
import time
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from predict import predict_category

BASE_DIR = Path(__file__).resolve().parent.parent
GRAPH_DIR = BASE_DIR / "graphs"
BACKEND_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
CORS(app)

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "NewsMind AI"})

@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if len(text) < 30:
        return jsonify({"error": "Please enter a news article with at least 30 characters."}), 400
    start = time.perf_counter()
    try:
        result = predict_category(text)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"category": result["category"], "confidence": result["confidence"], "prediction_time_ms": round((time.perf_counter() - start) * 1000, 2)})

@app.get("/api/metrics")
def metrics():
    path = BACKEND_DIR / "metrics.json"
    if not path.exists():
        return jsonify({"error": "Run python train.py first."}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))

@app.get("/api/report")
def report():
    path = GRAPH_DIR / "classification_report.txt"
    if not path.exists():
        return jsonify({"error": "Run python evaluate.py first."}), 404
    return jsonify({"report": path.read_text(encoding="utf-8")})

@app.get("/graphs/<path:filename>")
def graphs(filename):
    return send_from_directory(GRAPH_DIR, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
