from pathlib import Path

BASE_DIR = Path(r"C:\KI-RPA-Pipeline")
DATA_DIR = BASE_DIR / "data"

EMAILS_DIR = DATA_DIR / "emails_inbox"
STATE_DIR = DATA_DIR / "state"

MAILBOX_SMTP = "alexander.beine@hallesche.de"
TARGET_FOLDER = "KI-RPA-Pipeline"

MAX_MESSAGES = 500
UNREAD_ONLY = True
MARK_AS_READ = False
