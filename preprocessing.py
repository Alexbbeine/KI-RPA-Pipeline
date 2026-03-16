import re
import unicodedata

GREETING_RE = re.compile(
    r"^\s*(?:"
    r"hallo|hi|hey|moin|servus|guten\s+tag|guten\s+morgen|guten\s+abend|"
    r"sehr\s+geehrte(?:r|n)?|liebe(?:r|n|s)?"
    r")\b.*$",
    re.IGNORECASE,
)

FORMAL_CLOSING_RE = re.compile(
    r"^\s*(?:"
    r"mit\s+freundlichen\s+gr[uü][ßs]en|"
    r"freundliche\s+gr[uü][ßs]e|"
    r"viele\s+gr[uü][ßs]e|"
    r"beste\s+gr[uü][ßs]e|"
    r"mit\s+besten\s+gr[uü][ßs]en|"
    r"sch[oö]ne\s+gr[uü][ßs]e|"
    r"herzliche\s+gr[uü][ßs]e|"
    r"mit\s+herzlichen\s+gr[uü][ßs]en|"
    r"best\s+regards|kind\s+regards|regards"
    r")\s*[!,.]*\s*$",
    re.IGNORECASE,
)

SHORT_CLOSING_RE = re.compile(
    r"^\s*(?:vg|lg|mfg)\b.*$",
    re.IGNORECASE,
)

TEAM_CLOSING_RE = re.compile(
    r"^\s*(?:ihr|euer|dein)\s+[\wÄÖÜäöüß ._-]*team\s*$",
    re.IGNORECASE,
)

SIGNATURE_START_RE = re.compile(
    r"^\s*(?:i\s*\.\s*a\s*\.|i\s*\.\s*v\s*\.|--+|__)\s*.*$",
    re.IGNORECASE,
)

TECHNICAL_SUBJECT_TAG_RE = re.compile(
    r"\[[^\]]*(?:s/mime|verschluesselt|verschlüsselt|signiert)[^\]]*\]",
    re.IGNORECASE,
)

SUBJECT_PREFIX_RE = re.compile(r"^\s*(?:aw|wg|re|fw|fwd)\s*:\s*", re.IGNORECASE)


def normalize_linebreaks(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _trim_edge_blank_lines(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _is_closing_start(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    return any(
        pattern.match(stripped)
        for pattern in (FORMAL_CLOSING_RE, SHORT_CLOSING_RE, TEAM_CLOSING_RE, SIGNATURE_START_RE)
    )


def strip_salutation_and_closing(body: str) -> tuple[str, dict]:
    text = normalize_linebreaks(body)
    lines = _trim_edge_blank_lines(text.split("\n"))

    salutation_removed = False
    closing_removed = False

    # Anrede tolerant in den ersten 5 nicht-leeren Zeilen suchen
    first_non_empty_indices = [i for i, line in enumerate(lines) if line.strip()][:5]
    for idx in first_non_empty_indices:
        if GREETING_RE.match(lines[idx].strip()):
            del lines[idx]
            salutation_removed = True
            break

    lines = _trim_edge_blank_lines(lines)

    # Schlussformel/Signaturanfang in den letzten 30 Zeilen suchen.
    search_start = max(0, len(lines) - 30)
    closing_index = None

    for i in range(search_start, len(lines)):
        if _is_closing_start(lines[i]):
            closing_index = i
            break

    if closing_index is not None:
        lines = lines[:closing_index]
        closing_removed = True

    lines = _trim_edge_blank_lines(lines)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned, {
        "salutation_removed": salutation_removed,
        "closing_removed": closing_removed,
    }


def normalize_subject(subject: str) -> str:
    subject = (subject or "").strip()

    while SUBJECT_PREFIX_RE.match(subject):
        subject = SUBJECT_PREFIX_RE.sub("", subject).strip()

    subject = TECHNICAL_SUBJECT_TAG_RE.sub(" ", subject)
    subject = re.sub(r"\s{2,}", " ", subject).strip()

    return subject


def normalize_for_classification(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = normalize_linebreaks(text)

    # Technische Artefakte und Kontakt-/Link-Rauschen entfernen
    text = TECHNICAL_SUBJECT_TAG_RE.sub(" ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", text)

    # Nur Buchstaben/Zahlen/Leerzeichen/Umlaute/ß sowie - und / behalten
    text = re.sub(r"[^0-9A-Za-zÄÖÜäöüß\s\-/]", " ", text)

    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_email(subject: str, body: str) -> dict:
    subject_cleaned = normalize_subject(subject)
    body_cleaned, info = strip_salutation_and_closing(body)
    text_for_classification = normalize_for_classification(
        f"{subject_cleaned}\n{body_cleaned}"
    )

    return {
        "subject_cleaned": subject_cleaned,
        "body_cleaned": body_cleaned,
        "text_for_classification": text_for_classification,
        "preprocessing": info,
    }
