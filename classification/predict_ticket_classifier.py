import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer


# Lädt ein trainiertes Modell und gibt für einen Text die vorhergesagte Klasse sowie die zugehörogen Wahrscheinlichkeiten zurück.
def predict(model_dir: str, text: str, max_length: int = 192):
    model_path = Path(model_dir)
    model_dir_str = str(model_path).lower()

    # Passenden Tokenizer zum gespeicherten Modell laden.
    if "gbert" in model_dir_str or "bert-base-german" in model_dir_str:
        tokenizer = BertTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Den Eingabetext in dasselbe Tokenformat umwandeln wie beim Training.
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())

    # Die numerische Vorhersage wieder in das fachliche Label zurückübersetzen.
    label = model.config.id2label[pred_id]
    score = float(probs[pred_id].item())

    return {
        "label": label,
        "score": score,
        "all_scores": {
            model.config.id2label[i]: float(probs[i].item())
            for i in range(len(probs))
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=3000)
    args = parser.parse_args()

    # Ergebnis als JSON ausgeben, damit es leicht ausgelesen werden kann.
    result = predict(args.model_dir, args.text, max_length=args.max_length)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
