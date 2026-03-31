import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer

from config import CLASSIFIER_MODELS


def _load_tokenizer(model_path: Path):
    model_dir_str = str(model_path).lower()
    if 'gbert' in model_dir_str or 'bert-base-german' in model_dir_str:
        return BertTokenizer.from_pretrained(model_path, local_files_only=True)
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


class TicketClassifier:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_path = Path(model_dir).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f'Modellpfad nicht gefunden: {self.model_path}')
        if not (self.model_path / 'config.json').exists():
            raise FileNotFoundError(f'config.json nicht gefunden im Modellordner: {self.model_path}')

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = _load_tokenizer(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            int(label_id): label_name
            for label_id, label_name in self.model.config.id2label.items()
        }

    def predict(self, text: str, max_length: int = 256) -> dict:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities_tensor = torch.softmax(outputs.logits, dim=1)[0]
            predicted_id = int(torch.argmax(probabilities_tensor).item())

        sorted_items = sorted(
            [
                (self.id2label[index], float(probabilities_tensor[index].item()))
                for index in range(len(probabilities_tensor))
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        probabilities = {label: probability for label, probability in sorted_items}
        predicted_label = self.id2label[predicted_id]
        softmax_confidence = probabilities[predicted_label]

        return {
            'label': predicted_label,
            'softmax_confidence': softmax_confidence,
            'probabilities': probabilities,
            'top_3': [
                {'label': label, 'score': probability}
                for label, probability in sorted_items[:3]
            ],
        }


@lru_cache(maxsize=None)
def get_classifier(model_dir: str, device: str | None = None) -> TicketClassifier:
    resolved_path = str(Path(model_dir).resolve())
    return TicketClassifier(resolved_path, device=device)


def predict(model_dir: str, text: str, max_length: int = 256) -> dict:
    classifier = get_classifier(model_dir)
    return classifier.predict(text=text, max_length=max_length)


def load_classifiers() -> list[dict]:
    loaded_configs = []

    for key, config in CLASSIFIER_MODELS.items():
        model_dir = Path(config['model_dir']).resolve()
        max_length = int(config.get('max_length', 256))

        if not model_dir.exists():
            raise FileNotFoundError(
                f'Konfigurierter Modellordner fuer {key} nicht gefunden: {model_dir}'
            )

        loaded_configs.append(
            {
                'key': key,
                'model_dir': model_dir,
                'max_length': max_length,
                'classifier': get_classifier(str(model_dir)),
            }
        )

    return loaded_configs


def classify_email_text(text_for_classification: str, classifiers: list[dict]) -> dict:
    predictions = {}

    for classifier_config in classifiers:
        result = classifier_config['classifier'].predict(
            text=text_for_classification,
            max_length=classifier_config['max_length'],
        )

        predictions[classifier_config['key']] = {
            'label': result['label'],
            'softmax_confidence': result['softmax_confidence'],
            'top_3': result['top_3'],
            'probabilities': result['probabilities'],
            'model_dir': str(classifier_config['model_dir']),
        }

    return predictions


def build_predicted_ticket(classifications: dict) -> dict:
    return {
        'type': classifications['ticket_type']['label'],
        'area': classifications['ticket_area']['label'],
        'impact': classifications['ticket_impact']['label'],
        'priority': classifications['ticket_priority']['label'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--max-length', type=int, default=256)
    args = parser.parse_args()

    result = predict(args.model_dir, args.text, max_length=args.max_length)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
