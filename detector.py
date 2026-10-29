
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "baseline_sms_classifier.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


def analyze_message(message):
    if not message or not message.strip():
        raise ValueError("Please enter a message to analyze.")

    probabilities = model.predict_proba([message])[0]
    classes = list(model.classes_)

    spam_index = classes.index(1)
    spam_probability = float(probabilities[spam_index])

    prediction = "spam" if spam_probability >= 0.5 else "ham"

    if spam_probability >= 0.70:
        risk = "High"
    elif spam_probability >= 0.35:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "prediction": prediction,
        "spam_probability": spam_probability,
        "risk": risk
    }
