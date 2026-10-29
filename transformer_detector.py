
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = Path(__file__).parent / "models" / "distilbert_scam"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.to(DEVICE)
model.eval()

def analyze_with_distilbert(message: str) -> dict:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please provide a non-empty message.")

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_class = int(torch.argmax(probabilities).item())
    labels = {0: "Legitimate (ham)", 1: "Spam-like"}

    return {
        "prediction": labels[predicted_class],
        "spam_probability": round(float(probabilities[1].item()), 4),
        "legitimate_probability": round(float(probabilities[0].item()), 4)
    }

if __name__ == "__main__":
    examples = [
        "URGENT: Your bank account will be blocked. Click this link to verify your password.",
        "Hi, our project meeting is tomorrow at 11 AM. Please bring your notes."
    ]

    for message in examples:
        print("\nMessage:", message)
        print("Result:", analyze_with_distilbert(message))
