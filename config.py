from pathlib import Path

# Projektwurzel dynamisch relativ zu dieser Datei bestimmen.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

EMAILS_DIR = DATA_DIR / "emails_inbox"
TICKETS_DIR = DATA_DIR / "tickets"
ERRORS_DIR = DATA_DIR / "errors"
STATE_DIR = DATA_DIR / "state"
PROCESSED_DIR = DATA_DIR / "processed"

MAILBOX_SMTP = "alexander.beine@hallesche.de"
TARGET_FOLDER = "KI-RPA-Pipeline"

MAX_MESSAGES = 500
UNREAD_ONLY = True
MARK_AS_READ = False

CLASSIFIER_MODELS = {
    "ticket_type": {
        "model_dir": MODELS_DIR / "ticket-type" / "deepset-gbert",
        "max_length": 256,
    },
    "ticket_area": {
        "model_dir": MODELS_DIR / "ticket-area" / "hyperparameter/dbmz-german",
        "max_length": 256,
    },
    "ticket_impact": {
        "model_dir": MODELS_DIR / "ticket-impact" / "dbmz-german",
        "max_length": 256,
    },
    "ticket_priority": {
        "model_dir": MODELS_DIR / "ticket-prio" / "deepset-gbert",
        "max_length": 256,
    },
}

# Zielstruktur für Azure DevOps.
DEFAULT_ENVIRONMENT = "PROD"
ITERATION_ROOT = "SEU"
RELEASE_CUTOFF_DAYS = 14

# Es wird die Produktivsetzung als Release-Datum verwendet.
RELEASE_PRODUCTIVE_DATES = {
    2026: [
        "2026-03-21",
        "2026-06-13",
        "2026-09-12",
        "2026-11-28",
    ],
}
