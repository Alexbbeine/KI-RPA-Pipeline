import json
import hashlib
from pathlib import Path
from typing import Any

from config import EMAILS_DIR, STATE_DIR

# Datei zur Ablage aller bereits verarbeiteten Message-IDs.
PROCESSED_IDS_FILE = STATE_DIR / "_processed_ids.txt"


def ensure_directories() -> None:
    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

# Message-IDs als Menge laden, um eine schnelle Duplikatsprüfungen zu ermöglichen.
def load_processed_ids() -> set[str]:
    ensure_directories()

    if not PROCESSED_IDS_FILE.exists():
        return set()

    return {
        line.strip()
        for line in PROCESSED_IDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

# Message-ID am Ende der Statusdatei ergänzen, sobald eine E-Mail erfolgreich gespeichert wurde.
def append_processed_id(message_id: str) -> None:
    ensure_directories()

    with PROCESSED_IDS_FILE.open("a", encoding="utf-8") as f:
        f.write(message_id + "\n")

# Aus Empfangsdatum und gehashter Message-ID einen stabilen Dateinamen erzeugen.
def safe_filename(received_utc: str, message_id: str) -> str:
    hash_part = hashlib.sha1(message_id.encode("utf-8", errors="ignore")).hexdigest()[:16]
    date_part = received_utc[:10].replace("-", "")
    return f"EMAIL-{date_part}-{hash_part}.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """
    Schreibt JSON zunächst in eine temporäre Datei und benennt diese
    anschließend auf den Zielnamen um. Dadurch werden unvollständige
    Dateien bei Abbrüchen oder Fehlern vermieden.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)

# Vorverarbeitete E-Mail als JSON-datei persistent ablegen.
def save_email_json(record: dict[str, Any]) -> Path:
    ensure_directories()

    received_utc = record["email"]["received_utc"]
    message_id = record["meta"]["message_id"]

    target_path = EMAILS_DIR / safe_filename(received_utc, message_id)
    write_json_atomic(target_path, record)

    return target_path
