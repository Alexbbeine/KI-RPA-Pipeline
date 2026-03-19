from outlook_reader import fetch_emails
from preprocessing import preprocess_email
from storage import load_processed_ids, append_processed_id, save_email_json


def main():
    # Bereits verarbeitete Message_IDs laden, damit keine Mail doppelt verarbeitet wird.
    processed_ids = load_processed_ids()

    # E-Mails aus dem vordefinierten Outlook-Ordner auslesen.
    emails = fetch_emails()

    # Zähler initialisieren.
    total_read = len(emails)
    processed_count = 0
    skipped_count = 0
    error_count = 0

    print(f"[INFO] {total_read} Mail(s) aus Outlook ausgelesen.")

    # Jede aus Outlook geladene E-Mails nacheinander verarbeiten.
    for email in emails:
        # Die Message-ID dient als technischer Schlüssel zur eindeutigen Identifikation.
        message_id = email.get("message_id", "UNKNOWN_MESSAGE_ID")

        try:
            # Bereits bekannte Nachrichten werden übersprungen, um Duplikate in der weiteren Verarbeitung zur vermeiden.
            if message_id in processed_ids:
                skipped_count += 1
                print(f"[SKIP] Bereits verarbeitet: {message_id}")
                continue

            # Betreff und Nachrichtentext für die spätere Klassifkation bereinigen.
            processed = preprocess_email(email["subject"], email["body"])

            # Einheitliches JSON-Objekt aufbauen, das als Übergabeformat für die Ticketanalage dient.
            record = {
                "email": {
                    "subject": email["subject"],
                    "subject_cleaned": processed["subject_cleaned"],
                    "sender": email["sender"],
                    "received_utc": email["received_utc"],
                    "body": email["body"],
                    "body_cleaned": processed["body_cleaned"],
                    # Zusammengeführter und bereinigter Text für die ML-Modelle.
                    "text_for_classification": processed["text_for_classification"],
                },
                # Metainformationen für die Auditierbarkeit.
                "preprocessing": processed["preprocessing"],
                "meta": {
                    "source": "outlook_desktop",
                    "message_id": message_id,
                },
            }

            save_email_json(record)
            append_processed_id(message_id)
            processed_ids.add(message_id)

            processed_count += 1
            print(f"[OK] Mail verarbeitet: {message_id}")

        except Exception as ex:
            error_count += 1
            print(f"[ERROR] Fehler bei Mail {message_id}: {ex}")

    # Zusammenfassung des aktuellen Ausführungslaufs geben.
    print("\n--- Zusammenfassung ---")
    print(f"Ausgelesen:   {total_read}")
    print(f"Verarbeitet:  {processed_count}")
    print(f"Übersprungen: {skipped_count}")
    print(f"Fehler:       {error_count}")


if __name__ == "__main__":
    main()
