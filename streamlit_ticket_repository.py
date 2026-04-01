from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from config import TICKETS_DIR
from storage import load_json, utc_now_iso, write_json_atomic

EDITABLE_TICKET_FIELDS = (
    "Title",
    "Area",
    "Iteration",
    "Description",
    "Ticket-Type",
    "Environment",
    "Prio",
    "Impact",
)

CLASSIFICATION_LABELS = {
    "ticket_type": "Ticket-Typ",
    "ticket_area": "Bereich",
    "ticket_priority": "Prioritaet",
    "ticket_impact": "Schweregrad",
}

INDEX_FIELD_MAP = {
    "Title": "title",
    "Area": "area",
    "Iteration": "iteration",
    "Description": "description",
    "Ticket-Type": "ticket_type",
    "Environment": "environment",
    "Prio": "priority",
    "Impact": "impact",
}

CANONICAL_AREA_OPTIONS = [
    "SEU\\ALH\\Analytics",
    "SEU\\ALH\\Architektur",
    "SEU\\ALH\\Kranken",
    "SEU\\ALH\\Leben",
    "SEU\\ALH\\Sach",
    "SEU\\ALH\\Vertrieb",
    "SEU\\ALH\\Zentrale Systeme",
]

FIXED_SELECT_OPTIONS = {
    "ticket_type": ["ChangeRequest", "Problem"],
    "area": CANONICAL_AREA_OPTIONS,
    "priority": ["1", "2", "3", "4"],
    "impact": ["1 - Kritisch", "2 - Hoch", "3 - Mittel", "4 - Niedrig"],
    "environment": ["PROD", "RFRG"],
    "iteration": [],
}

IMPACT_LABEL_MAP = {
    "1": "1 - Kritisch",
    "2": "2 - Hoch",
    "3": "3 - Mittel",
    "4": "4 - Niedrig",
}

TICKET_TYPE_LABEL_MAP = {
    "problem": "Problem",
    "change request": "ChangeRequest",
    "changerequest": "ChangeRequest",
    "change-request": "ChangeRequest",
}

FIELD_TO_OPTION_KEY = {
    "Area": "area",
    "Ticket-Type": "ticket_type",
    "Environment": "environment",
    "Prio": "priority",
    "Impact": "impact",
}


def ensure_ticket_directory(ticket_dir: Path | None = None) -> Path:
    directory = Path(ticket_dir or TICKETS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def iter_ticket_files(ticket_dir: Path | None = None) -> list[Path]:
    directory = ensure_ticket_directory(ticket_dir)
    return sorted(directory.glob("TICKET-*.json"), reverse=True)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_spaces(value: str) -> str:
    return " ".join(value.split())


def _normalize_area(value: Any) -> str:
    text = _normalize_spaces(_as_text(value).strip())
    if not text:
        return ""

    normalized = text.replace("/", "\\")
    alias_map = {
        "SEU\\ALH\\ZentraleSysteme": "SEU\\ALH\\Zentrale Systeme",
        "SEU\\ALH\\ZENTRALESYSTEME": "SEU\\ALH\\Zentrale Systeme",
    }

    if normalized in alias_map:
        return alias_map[normalized]

    for canonical in CANONICAL_AREA_OPTIONS:
        if normalized.lower() == canonical.lower():
            return canonical

    return normalized


def _normalize_ticket_type(value: Any) -> str:
    text = _normalize_spaces(_as_text(value).strip())
    if not text:
        return ""

    return TICKET_TYPE_LABEL_MAP.get(text.lower(), text)


def _normalize_priority(value: Any) -> str:
    text = _normalize_spaces(_as_text(value).strip())
    if not text:
        return ""

    leading = text.split("-", 1)[0].strip()
    if leading in {"1", "2", "3", "4"}:
        return leading

    return text


def _normalize_impact(value: Any) -> str:
    text = _normalize_spaces(_as_text(value).strip())
    if not text:
        return ""

    leading = text.split("-", 1)[0].strip()
    if leading in IMPACT_LABEL_MAP:
        return IMPACT_LABEL_MAP[leading]

    return text


def _normalize_environment(value: Any) -> str:
    text = _normalize_spaces(_as_text(value).strip())
    if not text:
        return ""

    return text.upper()


def normalize_option_value(option_key: str, value: Any) -> str:
    if option_key == "area":
        return _normalize_area(value)
    if option_key == "ticket_type":
        return _normalize_ticket_type(value)
    if option_key == "priority":
        return _normalize_priority(value)
    if option_key == "impact":
        return _normalize_impact(value)
    if option_key == "environment":
        return _normalize_environment(value)
    if option_key == "iteration":
        return _normalize_spaces(_as_text(value).strip())
    return _normalize_spaces(_as_text(value).strip())


def _extract_confidence_map(classification: dict[str, Any]) -> dict[str, float]:
    confidences: dict[str, float] = {}

    for classifier_key, payload in classification.items():
        score = payload.get("softmax_confidence")
        if isinstance(score, (int, float)):
            confidences[classifier_key] = float(score)

    return confidences


def _truncate(value: str, limit: int = 160) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def normalize_ticket_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    email = record.get("email", {})
    ticket = record.get("ticket", {})
    meta = record.get("meta", {})
    manual_review = record.get("manual_review", {})

    ticket_id = _as_text(meta.get("message_id") or source_path.stem)
    received_utc = _as_text(email.get("received_utc"))
    created_utc = _as_text(meta.get("ticket_created_at_utc"))
    description = _as_text(ticket.get("Description") or email.get("body_cleaned") or email.get("body"))

    confidence_map = _extract_confidence_map(record.get("classification", {}))
    confidence_values = list(confidence_map.values())
    average_confidence = mean(confidence_values) if confidence_values else None
    minimum_confidence = min(confidence_values) if confidence_values else None

    history = manual_review.get("history", [])
    manually_edited = isinstance(history, list) and len(history) > 0

    return {
        "ticket_id": ticket_id,
        "file_name": source_path.name,
        "file_path": str(source_path),
        "title": _as_text(ticket.get("Title") or email.get("subject")),
        "sender": _as_text(email.get("sender")),
        "received_utc": received_utc,
        "ticket_created_at_utc": created_utc,
        "ticket_type": normalize_option_value("ticket_type", ticket.get("Ticket-Type")),
        "area": normalize_option_value("area", ticket.get("Area")),
        "iteration": normalize_option_value("iteration", ticket.get("Iteration")),
        "environment": normalize_option_value("environment", ticket.get("Environment")),
        "priority": normalize_option_value("priority", ticket.get("Prio")),
        "impact": normalize_option_value("impact", ticket.get("Impact")),
        "description": description,
        "description_preview": _truncate(description),
        "average_confidence": average_confidence,
        "minimum_confidence": minimum_confidence,
        "confidence_map": confidence_map,
        "manually_edited": manually_edited,
        "edit_count": len(history) if isinstance(history, list) else 0,
        "last_manual_save_utc": _as_text(manual_review.get("last_saved_at_utc")),
    }


def load_ticket_index(ticket_dir: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for ticket_file in iter_ticket_files(ticket_dir):
        try:
            record = load_json(ticket_file)
            rows.append(normalize_ticket_record(record, ticket_file))
        except Exception as error:
            rows.append(
                {
                    "ticket_id": ticket_file.stem,
                    "file_name": ticket_file.name,
                    "file_path": str(ticket_file),
                    "title": "Fehler beim Laden",
                    "sender": "",
                    "received_utc": "",
                    "ticket_created_at_utc": "",
                    "ticket_type": "",
                    "area": "",
                    "iteration": "",
                    "environment": "",
                    "priority": "",
                    "impact": "",
                    "description": "",
                    "description_preview": f"Datei konnte nicht gelesen werden: {error}",
                    "average_confidence": None,
                    "minimum_confidence": None,
                    "confidence_map": {},
                    "manually_edited": False,
                    "edit_count": 0,
                    "last_manual_save_utc": "",
                    "load_error": str(error),
                }
            )

    rows.sort(
        key=lambda row: (
            row.get("received_utc", ""),
            row.get("ticket_created_at_utc", ""),
            row.get("file_name", ""),
        ),
        reverse=True,
    )
    return rows


def load_ticket_record_by_id(
    ticket_id: str,
    ticket_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    for ticket_file in iter_ticket_files(ticket_dir):
        record = load_json(ticket_file)
        message_id = _as_text(record.get("meta", {}).get("message_id") or ticket_file.stem)
        if message_id == ticket_id:
            return ticket_file, record
    return None


def build_editable_ticket(record: dict[str, Any]) -> dict[str, str]:
    ticket = record.get("ticket", {})
    email = record.get("email", {})

    editable = {
        "Title": _as_text(ticket.get("Title") or email.get("subject")),
        "Area": normalize_option_value("area", ticket.get("Area")),
        "Iteration": normalize_option_value("iteration", ticket.get("Iteration")),
        "Description": _as_text(ticket.get("Description") or email.get("body_cleaned") or email.get("body")),
        "Ticket-Type": normalize_option_value("ticket_type", ticket.get("Ticket-Type")),
        "Environment": normalize_option_value("environment", ticket.get("Environment")),
        "Prio": normalize_option_value("priority", ticket.get("Prio")),
        "Impact": normalize_option_value("impact", ticket.get("Impact")),
    }
    return editable


def build_classification_overview(record: dict[str, Any]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []

    for classifier_key, payload in record.get("classification", {}).items():
        top_3 = payload.get("top_3", []) or []
        alternatives = [item.get("label", "") for item in top_3[1:3] if isinstance(item, dict)]
        prediction = _as_text(payload.get("label"))

        if classifier_key == "ticket_type":
            prediction = normalize_option_value("ticket_type", prediction)
        elif classifier_key == "ticket_area":
            prediction = normalize_option_value("area", prediction)
        elif classifier_key == "ticket_priority":
            prediction = normalize_option_value("priority", prediction)
        elif classifier_key == "ticket_impact":
            prediction = normalize_option_value("impact", prediction)

        rows.append(
            {
                "Modell": CLASSIFICATION_LABELS.get(classifier_key, classifier_key),
                "Vorhersage": prediction,
                "Konfidenz": float(payload.get("softmax_confidence", 0.0) or 0.0),
                "Alternative 1": alternatives[0] if len(alternatives) > 0 else "",
                "Alternative 2": alternatives[1] if len(alternatives) > 1 else "",
                "Modellpfad": _as_text(payload.get("model_dir")),
            }
        )

    return rows


def collect_options(index_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {
        key: list(values) for key, values in FIXED_SELECT_OPTIONS.items()
    }

    for row in index_rows:
        for key in options:
            value = normalize_option_value(key, row.get(key))
            if value and value not in options[key]:
                options[key].append(value)

    return options


def option_list(values: list[str], current_value: str) -> list[str]:
    unique_values = [value for value in values if value]
    current_value = _as_text(current_value).strip()

    if current_value and current_value not in unique_values:
        unique_values.append(current_value)
    if not unique_values:
        unique_values = [""]

    return unique_values


def update_ticket_record(
    ticket_id: str,
    updated_ticket: dict[str, Any],
    ticket_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    loaded = load_ticket_record_by_id(ticket_id, ticket_dir=ticket_dir)
    if loaded is None:
        raise FileNotFoundError(f"Ticket mit der ID {ticket_id} wurde nicht gefunden.")

    target_path, record = loaded
    ticket = record.setdefault("ticket", {})
    original_ticket = {field: _as_text(ticket.get(field)) for field in EDITABLE_TICKET_FIELDS}
    changed_fields: dict[str, dict[str, str]] = {}

    for field in EDITABLE_TICKET_FIELDS:
        new_value = _as_text(updated_ticket.get(field))

        if field != "Description":
            new_value = new_value.strip()
        else:
            new_value = new_value.strip("\n")

        option_key = FIELD_TO_OPTION_KEY.get(field)
        if option_key:
            new_value = normalize_option_value(option_key, new_value)

        old_value_raw = original_ticket.get(field, "")
        if field != "Description":
            old_value_compare = old_value_raw.strip()
        else:
            old_value_compare = old_value_raw.strip("\n")

        if option_key:
            old_value_compare = normalize_option_value(option_key, old_value_compare)

        if old_value_compare != new_value:
            changed_fields[field] = {"old": old_value_raw, "new": new_value}
            ticket[field] = new_value

    if not changed_fields:
        return changed_fields

    manual_review = record.setdefault("manual_review", {})
    manual_review.setdefault("initial_ticket", original_ticket)
    history = manual_review.setdefault("history", [])
    history.append(
        {
            "edited_at_utc": utc_now_iso(),
            "source": "streamlit_ui",
            "changed_fields": changed_fields,
        }
    )
    manual_review["last_saved_at_utc"] = utc_now_iso()

    meta = record.setdefault("meta", {})
    meta["last_updated_by_ui_at_utc"] = utc_now_iso()

    write_json_atomic(target_path, record)
    return changed_fields
