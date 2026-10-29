
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
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_PATH = BASE_DIR / "models" / "baseline_sms_classifier.joblib"


def main():
    # Load the dataset: each row contains a label and an SMS.
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "sms"],
        encoding="utf-8",
        on_bad_lines="skip",
    )

    df = df.dropna(subset=["label", "sms"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()

    # Match the numeric labels used by the saved classifier.
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    model = joblib.load(MODEL_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["sms"],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    # Evaluate on the held-out split.
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    print("\n===== SCAM INTELLIGENCE COPILOT =====")
    print("===== BASELINE MODEL EVALUATION =====")
    print(f"Dataset messages: {len(df)}")
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
    print(f"Spam precision: {precision:.4f}")
    print(f"Spam recall:    {recall:.4f}")
    print(f"Spam F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Ham", "Spam"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Ham", "Spam"],
    )
    display.plot(values_format="d")
    plt.title("Baseline SMS Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png") 


if __name__ == "__main__":
    main()
