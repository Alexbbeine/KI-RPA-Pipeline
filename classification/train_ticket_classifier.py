
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


TEXT_COLUMNS_DEFAULT = ["Titel", "Beschreibung"]


def read_table(path: Path, sheet_name: str = "Tabelle") -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Nicht unterstütztes Dateiformat: {suffix}")


def build_text(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    return (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg("\n\n".join, axis=1)
        .str.strip()
    )


def prepare_dataframe(df: pd.DataFrame, label_col: str, text_cols: list[str]) -> pd.DataFrame:
    required = text_cols + [label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten: {missing}")

    work = df.copy()
    work["text"] = build_text(work, text_cols)
    work[label_col] = work[label_col].fillna("").astype(str).str.strip()

    work = work[(work["text"] != "") & (work[label_col] != "")].copy()
    work = work.rename(columns={label_col: "label_text"})[["text", "label_text"]]

    # Exakte Dubletten entfernen, damit identische Tickets nicht über Train/Val/Test verteilt werden.
    work = work.drop_duplicates(subset=["text", "label_text"]).reset_index(drop=True)

    counts = work["label_text"].value_counts()
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise ValueError(
            "Zu kleine Klassen gefunden. Mindestens 3 Beispiele pro Klasse nötig.\n"
            + too_small.to_string()
        )

    return work


def make_splits(df: pd.DataFrame, seed: int):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label_text"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label_text"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, label2id: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label_text"].map(label2id)
    return out


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(y: pd.Series, label2id: dict[str, int]) -> torch.Tensor:
    counts = y.value_counts().to_dict()
    total = len(y)
    num_classes = len(label2id)
    weights = []

    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        c = counts[label]
        # balanced weight
        weights.append(total / (num_classes * c))

    return torch.tensor(weights, dtype=torch.float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Pfad zu CSV/XLSX mit historischen Tickets")
    parser.add_argument("--label-col", required=True, help="Zielspalte, z. B. Typ, Bereich, Prio oder Schweregrad")
    parser.add_argument("--text-cols", nargs="+", default=TEXT_COLUMNS_DEFAULT, help="Textspalten, standardmäßig Titel Beschreibung")
    parser.add_argument("--model", default="deepset/gbert-base", help="z. B. deepset/gbert-base oder distilbert/distilbert-base-german-cased")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sheet-name", default="Tabelle")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_table(data_path, sheet_name=args.sheet_name)
    raw_rows = len(raw_df)
    df = prepare_dataframe(raw_df, label_col=args.label_col, text_cols=args.text_cols)
    prepared_rows = len(df)

    train_df, val_df, test_df = make_splits(df, seed=args.seed)

    labels = sorted(df["label_text"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_df = encode_labels(train_df, label2id)
    val_df = encode_labels(val_df, label2id)
    test_df = encode_labels(test_df, label2id)

    train_ds = Dataset.from_pandas(train_df[["text", "label_text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label_text", "label"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[["text", "label_text", "label"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text", "label_text"])
    tokenized_val = val_ds.map(tokenize, batched=True, remove_columns=["text", "label_text"])
    tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=["text", "label_text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_df["label_text"], label2id)

    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels_np, preds),
            "macro_f1": f1_score(labels_np, preds, average="macro"),
            "weighted_f1": f1_score(labels_np, preds, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=True,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_metrics = trainer.evaluate(tokenized_test, metric_key_prefix="test")
    pred_output = trainer.predict(tokenized_test)
    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True,
        zero_division=0,
    )

    (output_dir / "metrics_test.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
                "raw_rows": raw_rows,
                "prepared_rows_after_cleaning_and_dedup": prepared_rows,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "use_class_weights": bool(args.use_class_weights),
                "base_model": args.model,
                "max_length": args.max_length,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Training abgeschlossen.")
    print(f"Modell gespeichert in: {output_dir}")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
