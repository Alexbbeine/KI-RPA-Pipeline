import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from train_ticket_classifier import read_table, prepare_dataframe, make_splits


TEXT_COLUMNS_DEFAULT = ["Titel", "Beschreibung"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sheet-name", default="Tabelle")
    parser.add_argument("--text-cols", nargs="+", default=TEXT_COLUMNS_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    raw_df = read_table(Path(args.data), sheet_name=args.sheet_name)
    df = prepare_dataframe(raw_df, label_col=args.label_col, text_cols=args.text_cols)
    _, _, test_df = make_splits(df, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    label2id = {str(k): int(v) for k, v in model.config.label2id.items()}
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    test_df = test_df.copy()
    test_df["label"] = test_df["label_text"].map(label2id)

    if test_df["label"].isna().any():
        missing = test_df.loc[test_df["label"].isna(), "label_text"].unique().tolist()
        raise ValueError(f"Labels im Testset fehlen im Modellmapping: {missing}")

    test_ds = Dataset.from_pandas(
        test_df[["text", "label_text", "label"]],
        preserve_index=False
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_test = test_ds.map(
        tokenize,
        batched=True,
        remove_columns=["text", "label_text"]
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir / "_eval_tmp"),
            per_device_eval_batch_size=args.eval_batch_size,
            report_to="none",
            use_cpu=True,
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    eval_metrics = trainer.evaluate(tokenized_test, metric_key_prefix="test")
    pred_output = trainer.predict(tokenized_test)

    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=1)

    metrics = {
        **eval_metrics,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "test_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True,
        zero_division=0,
    )

    (model_dir / "metrics_test.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (model_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nGespeichert in: {model_dir}")


if __name__ == "__main__":
    main()