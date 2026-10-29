
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# Configuration
MODEL_NAME = "distilbert/distilbert-base-uncased"
DATA_PATH = Path("data/SMSSpamCollection")
OUTPUT_DIR = Path("models/distilbert_scam")
SEED = 42

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}. "
            "Check the location of SMSSpamCollection."
        )

    # Load the tab-separated SMS Spam Collection dataset
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="latin-1"
    )
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df.drop_duplicates(subset=["text"])
    df["labels"] = df["label"].map({"ham": 0, "spam": 1})

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["labels"]
    )

    print(f"Total messages: {len(df)}")
    print(f"Training messages: {len(train_df)}")
    print(f"Test messages: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class SMSDataset(torch.utils.data.Dataset):
        def __init__(self, frame):
            self.texts = frame["text"].astype(str).tolist()
            self.labels = frame["labels"].astype(int).tolist()

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            item = tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=128
            )
            item["labels"] = self.labels[idx]
            return item

    train_dataset = SMSDataset(train_df)
    test_dataset = SMSDataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    def compute_metrics(pred):
        logits = pred.predictions
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            pred.label_ids,
            predictions,
            average="binary",
            zero_division=0
        )
        return {
            "accuracy": accuracy_score(pred.label_ids, predictions),
            "spam_precision": precision,
            "spam_recall": recall,
            "spam_f1": f1
        }

    args = TrainingArguments(
        output_dir="models/distilbert_checkpoints",
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=SEED,
        use_cpu=not torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    print("\nStarting DistilBERT fine-tuning...")
    trainer.train()

    print("\nEvaluating on held-out test messages...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = test_df["labels"].to_numpy()

    print(classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate (ham)", "Spam-like"],
        digits=4,
        zero_division=0
    ))

    metrics = compute_metrics(predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model to:", OUTPUT_DIR.resolve())
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
