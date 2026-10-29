
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_PATH = BASE_DIR / "models" / "baseline_sms_classifier.joblib"


def get_evaluation_results():
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "sms"],
        encoding="utf-8",
        on_bad_lines="skip",
    )

    df = df.dropna(subset=["label", "sms"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    model = joblib.load(MODEL_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["sms"],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        "test_size": len(y_test),
    }
