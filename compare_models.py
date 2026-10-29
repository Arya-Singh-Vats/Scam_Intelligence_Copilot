
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_PATH = Path("data/SMSSpamCollection")
BASELINE_PATH = Path("models/baseline_sms_classifier.joblib")
TRANSFORMER_PATH = Path("models/distilbert_scam")
RESULTS_PATH = Path("models/model_comparison.json")

SEED = 42


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="latin-1",
    )

    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df.drop_duplicates(subset=["text"])
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["target"],
    )

    return train_df, test_df


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "spam_precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "spam_recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "spam_f1": round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
    }


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_PATH}")

    if not TRANSFORMER_PATH.exists():
        raise FileNotFoundError(f"DistilBERT model not found: {TRANSFORMER_PATH}")

    train_df, test_df = load_data()

    print("Training split:", len(train_df))
    print("Held-out test split:", len(test_df))
    print("\nIMPORTANT: verify that the baseline model was trained only")
    print("on this training split before interpreting this comparison.")

    # Load baseline classifier
    baseline = joblib.load(BASELINE_PATH)
    texts = test_df["text"].tolist()
    y_true = test_df["target"].to_numpy()

    baseline_raw = baseline.predict(texts)
    # Expected baseline labels: ham/spam or 0/1
    baseline_pred = np.array([
        1 if str(value).strip().lower() in ("spam", "1", "true")
        else 0
        for value in baseline_raw
    ])

    # Load fine-tuned DistilBERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(TRANSFORMER_PATH)
    ).to(device)
    model.eval()

    transformer_pred = []

    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            predictions = torch.argmax(logits, dim=1).cpu().tolist()

        transformer_pred.extend(predictions)

    results = {
        "test_samples": len(y_true),
        "baseline": calculate_metrics(y_true, baseline_pred),
        "distilbert": calculate_metrics(y_true, transformer_pred),
    }

    print("\nBASELINE CLASSIFIER")
    print(classification_report(
        y_true, baseline_pred,
        target_names=["Legitimate (ham)", "Spam-like"],
        zero_division=0,
    ))

    print("\nDISTILBERT CLASSIFIER")
    print(classification_report(
        y_true, transformer_pred,
        target_names=["Legitimate (ham)", "Spam-like"],
        zero_division=0,
    ))

    print("\nMETRICS COMPARISON")
    print(json.dumps(results, indent=2))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
