from datetime import datetime, timezone

import win32com.client

from config import MAILBOX_SMTP, TARGET_FOLDER, MAX_MESSAGES, UNREAD_ONLY

PR_INTERNET_MESSAGE_ID = "http://schemas.microsoft.com/mapi/proptag/0x1035001E"
UNREAD_ONLY = False


def to_utc_iso(dt) -> str:
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        dt = dt.replace(tzinfo=local_tz)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def get_sender_smtp(item) -> str:
    """
    Liefert möglichst die echte SMTP-Adresse.
    Bei internen Exchange-Absendern kommt sonst oft nur eine Exchange-DN zurück.
    """
    try:
        sender_type = str(getattr(item, "SenderEmailType", "") or "").upper()

        if sender_type == "SMTP":
            return str(getattr(item, "SenderEmailAddress", "") or "")

        if sender_type == "EX" and getattr(item, "Sender", None):
            ex_user = item.Sender.GetExchangeUser()
            if ex_user and getattr(ex_user, "PrimarySmtpAddress", None):
                return str(ex_user.PrimarySmtpAddress or "")
    except Exception:
        pass

    return str(getattr(item, "SenderEmailAddress", "") or "")


def get_message_id(item) -> str:
    try:
        return item.PropertyAccessor.GetProperty(PR_INTERNET_MESSAGE_ID)
    except Exception:
        return str(item.EntryID)


def fetch_emails() -> list[dict]:
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

    # Orientierung am Ursprung:
    # direkt das Postfach per angezeigtem Namen/Mailadresse holen
    mailbox = outlook.Folders.Item(MAILBOX_SMTP)
    target_folder = mailbox.Folders.Item(TARGET_FOLDER)

    items = target_folder.Items
    items.Sort("[ReceivedTime]", True)

    if UNREAD_ONLY:
        items = items.Restrict("[Unread] = True")

    emails = []
    count = 0

    for item in items:
        # 43 = MailItem
        if getattr(item, "Class", None) != 43:
            continue

        try:
            email_data = {
                "subject": str(getattr(item, "Subject", "") or ""),
                "sender": get_sender_smtp(item),
                "received_utc": to_utc_iso(item.ReceivedTime),
                "body": str(getattr(item, "Body", "") or ""),
                "message_id": get_message_id(item),
            }

            emails.append(email_data)

            count += 1
            if count >= MAX_MESSAGES:
                break

        except Exception as ex:
            print(f"Fehler bei Mail: {ex}")

    return emails
