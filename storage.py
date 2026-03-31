import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config import EMAILS_DIR, ERRORS_DIR, PROCESSED_DIR, STATE_DIR, TICKETS_DIR

STORED_EMAIL_IDS_FILE = STATE_DIR / "_processed_ids.txt"
TICKETED_IDS_FILE = STATE_DIR / "_ticketed_ids.txt"


def ensure_directories() -> None:
    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _load_id_set(path: Path) -> set[str]:
    ensure_directories()
    if not path.exists():
        return set()

    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _append_id(path: Path, value: str) -> None:
    ensure_directories()
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(value + "\n")


def load_stored_email_ids() -> set[str]:
    return _load_id_set(STORED_EMAIL_IDS_FILE)


def append_stored_email_id(message_id: str) -> None:
    _append_id(STORED_EMAIL_IDS_FILE, message_id)


def load_ticketed_ids() -> set[str]:
    return _load_id_set(TICKETED_IDS_FILE)


def append_ticketed_id(message_id: str) -> None:
    _append_id(TICKETED_IDS_FILE, message_id)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_hash(value: str) -> str:
    return hashlib.sha1((value or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def safe_stem(received_utc: str, message_id: str) -> str:
    date_part = (received_utc or utc_now_iso())[:10].replace("-", "")
    return f"{date_part}-{safe_hash(message_id)}"


def build_email_filename(received_utc: str, message_id: str) -> str:
    return f"EMAIL-{safe_stem(received_utc, message_id)}.json"


def build_ticket_filename(received_utc: str, message_id: str) -> str:
    return f"TICKET-{safe_stem(received_utc, message_id)}.json"


def build_error_filename(stage: str, message_id: str | None = None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hash_part = safe_hash(message_id or timestamp)
    stage_part = stage.replace(" ", "_").replace("/", "_")
    return f"ERROR-{timestamp}-{stage_part}-{hash_part}.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_directories()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def save_email_json(record: dict[str, Any]) -> Path:
    received_utc = record["email"]["received_utc"]
    message_id = record["meta"]["message_id"]
    target_path = EMAILS_DIR / build_email_filename(received_utc, message_id)
    write_json_atomic(target_path, record)
    return target_path


def save_ticket_json(record: dict[str, Any], *, received_utc: str, message_id: str) -> Path:
    target_path = TICKETS_DIR / build_ticket_filename(received_utc, message_id)
    write_json_atomic(target_path, record)
    return target_path


def save_error_report(report: dict[str, Any]) -> Path:
    stage = str(report.get("stage", "unknown"))
    message_id = report.get("message_id")
    target_path = ERRORS_DIR / build_error_filename(stage=stage, message_id=message_id)
    write_json_atomic(target_path, report)
    return target_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_email_json_files() -> Iterable[Path]:
    ensure_directories()
    return sorted(EMAILS_DIR.glob("EMAIL-*.json"))
