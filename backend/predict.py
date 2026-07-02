from pathlib import Path
import joblib
from preprocess import preprocess_text

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "model.pkl"
VECTORIZER_PATH = BACKEND_DIR / "vectorizer.pkl"

def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Run python train.py before making predictions.")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

def predict_category(text):
    model, vectorizer = load_artifacts()
    features = vectorizer.transform([preprocess_text(text)])
    category = model.predict(features)[0]
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(features)[0].max() * 100)
    elif hasattr(model, "decision_function"):
        import numpy as np
        scores = model.decision_function(features)
        scores = scores[0] if getattr(scores, "ndim", 1) > 1 else scores
        exp_scores = np.exp(scores - np.max(scores))
        confidence = float((exp_scores / exp_scores.sum()).max() * 100)
    return {"category": str(category), "confidence": round(confidence, 2)}
