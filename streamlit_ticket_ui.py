from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from config import RPA_INBOX_DIR, TICKETS_DIR
from streamlit_ticket_repository import (
    build_classification_overview,
    build_editable_ticket,
    collect_options,
    format_area_display,
    load_ticket_index,
    load_ticket_record_by_id,
    move_tickets_to_rpa_inbox,
    update_ticket_record,
)

OVERVIEW_PAGE = None
DETAIL_PAGE = None


@st.cache_data(show_spinner=False)
def get_ticket_index_cached(signature: tuple[tuple[str, int], ...]) -> list[dict[str, Any]]:
    del signature
    return load_ticket_index(TICKETS_DIR)


def build_inventory_signature() -> tuple[tuple[str, int], ...]:
    directory = Path(TICKETS_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    return tuple(
        sorted((file_path.name, file_path.stat().st_mtime_ns) for file_path in directory.glob("TICKET-*.json"))
    )


def clear_ticket_cache() -> None:
    get_ticket_index_cached.clear()


def get_ticket_index() -> list[dict[str, Any]]:
    return get_ticket_index_cached(build_inventory_signature())


def format_timestamp(value: str) -> str:
    if not value:
        return "-"

    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return value

    return parsed.tz_convert("Europe/Berlin").strftime("%d.%m.%Y %H:%M")


def format_confidence(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"



def build_display_dataframe(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raw_df = pd.DataFrame(
            columns=[
                "ticket_id",
                "title",
                "sender",
                "received_utc",
                "ticket_type",
                "area",
                "priority",
                "impact",
                "average_confidence",
                "manually_edited",
                "description_preview",
            ]
        )

    display_df = pd.DataFrame(
        {
            "Titel": raw_df["title"],
            "Absender": raw_df["sender"],
            "Empfangen": raw_df["received_utc"].apply(format_timestamp),
            "Typ": raw_df["ticket_type"],
            "Bereich": raw_df["area"].apply(format_area_display),
            "Priorität": raw_df["priority"],
            "Schweregrad": raw_df["impact"],
            "Ø Konfidenz": raw_df["average_confidence"].apply(format_confidence),
            "Manuell geändert": raw_df["manually_edited"].map({True: "Ja", False: "Nein"}),
            "Beschreibung": raw_df["description_preview"],
        }
    )

    return raw_df.reset_index(drop=True), display_df.reset_index(drop=True)


def apply_filters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    st.sidebar.header("Filter")
    search_value = st.sidebar.text_input("Suche", placeholder="Titel, Absender oder Beschreibung")

    all_types = sorted({row.get("ticket_type", "") for row in rows if row.get("ticket_type")})
    all_areas = sorted({row.get("area", "") for row in rows if row.get("area")})
    all_priorities = sorted({row.get("priority", "") for row in rows if row.get("priority")})

    selected_types = st.sidebar.multiselect("Ticket-Typ", options=all_types)
    selected_areas = st.sidebar.multiselect(
        "Bereich",
        options=all_areas,
        format_func=format_area_display,
    )
    selected_priorities = st.sidebar.multiselect("Priorität", options=all_priorities)
    only_manual = st.sidebar.toggle("Nur manuell geänderte Tickets", value=False)
    min_confidence = st.sidebar.slider("Minimale Ø Konfidenz", 0.0, 1.0, 0.0, 0.05)

    filtered_rows: list[dict[str, Any]] = []
    normalized_search = search_value.strip().lower()

    for row in rows:
        haystack = " ".join(
            [
                str(row.get("title", "")),
                str(row.get("sender", "")),
                str(row.get("area", "")),
                str(row.get("ticket_type", "")),
                str(row.get("description", "")),
            ]
        ).lower()

        if normalized_search and normalized_search not in haystack:
            continue
        if selected_types and row.get("ticket_type") not in selected_types:
            continue
        if selected_areas and row.get("area") not in selected_areas:
            continue
        if selected_priorities and row.get("priority") not in selected_priorities:
            continue
        if only_manual and not row.get("manually_edited"):
            continue

        avg_confidence = row.get("average_confidence")
        if avg_confidence is not None and float(avg_confidence) < min_confidence:
            continue

        filtered_rows.append(row)

    return filtered_rows


def render_distribution_chart(series: pd.Series, title: str, *, label_formatter=None) -> None:
    st.subheader(title)

    if series.empty:
        st.info("Keine Daten vorhanden.")
        return

    chart_df = series.reset_index()
    chart_df.columns = ["Kategorie", "Anzahl"]
    if label_formatter is not None:
        chart_df["Kategorie"] = chart_df["Kategorie"].apply(label_formatter)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Kategorie:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Anzahl:Q", title="Anzahl", axis=alt.Axis(format="d", tickMinStep=1)),
            tooltip=[alt.Tooltip("Kategorie:N", title="Kategorie"), alt.Tooltip("Anzahl:Q", title="Anzahl")],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)


def build_select_state(base_options: list[str], current_value: str) -> tuple[list[str], int]:
    options = [option for option in base_options if option]
    current_value = str(current_value or "").strip()

    if current_value and current_value not in options:
        options = [*options, current_value]

    if not options:
        options = [""]

    index = options.index(current_value) if current_value in options else 0
    return options, index


def execute_pipeline(mode: str) -> tuple[dict[str, Any], str]:
    from main import run_pipeline

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = run_pipeline(mode=mode)

    return result, buffer.getvalue().strip()


def render_pipeline_controls() -> None:
    st.subheader("Mailbox und Klassifikation")

    flash_message = st.session_state.pop("pipeline_flash_message", None)
    error_message = st.session_state.pop("pipeline_error_message", None)

    if flash_message:
        st.success(flash_message)
    if error_message:
        st.error(error_message)

    action_col_1, action_col_2, action_col_3 = st.columns(3)
    run_all = action_col_1.button("Mails holen und klassifizieren", type="primary", use_container_width=True)
    fetch_only = action_col_2.button("Nur Mails holen", use_container_width=True)
    classify_only = action_col_3.button("Nur klassifizieren", use_container_width=True)

    st.caption(f"RPA-Inbox für die Ticketanlage: {Path(RPA_INBOX_DIR)}")

    requested_mode = None
    if run_all:
        requested_mode = "all"
    elif fetch_only:
        requested_mode = "fetch"
    elif classify_only:
        requested_mode = "classify"

    if requested_mode:
        with st.spinner("Pipeline wird ausgeführt..."):
            try:
                pipeline_result, pipeline_log = execute_pipeline(requested_mode)
                st.session_state["last_pipeline_result"] = pipeline_result
                st.session_state["last_pipeline_log"] = pipeline_log
                st.session_state["pipeline_flash_message"] = "Pipeline erfolgreich abgeschlossen."
                clear_ticket_cache()
                st.rerun()
            except Exception as error:
                st.session_state["pipeline_error_message"] = f"Pipeline konnte nicht ausgeführt werden: {error}"
                st.rerun()

    last_pipeline_result = st.session_state.get("last_pipeline_result")
    last_pipeline_log = st.session_state.get("last_pipeline_log", "")
    if not last_pipeline_result:
        return

    summary_left, summary_right = st.columns(2)
    fetch = last_pipeline_result.get("fetch", {})
    classification = last_pipeline_result.get("classification", {})

    with summary_left:
        st.markdown("**Fetch-Stufe**")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Ausgelesen", int(fetch.get("read", 0)))
        metric_cols[1].metric("Gespeichert", int(fetch.get("stored", 0)))
        metric_cols[2].metric("Übersprungen", int(fetch.get("skipped", 0)))
        metric_cols[3].metric("Fehler", int(fetch.get("errors", 0)))

    with summary_right:
        st.markdown("**Klassifikations-Stufe**")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Geprüft", int(classification.get("checked", 0)))
        metric_cols[1].metric("Ticket erstellt", int(classification.get("ticketed", 0)))
        metric_cols[2].metric("Übersprungen", int(classification.get("skipped", 0)))
        metric_cols[3].metric("Fehler", int(classification.get("errors", 0)))

    st.caption(
        f"Letzte Ausführung: {format_timestamp(last_pipeline_result.get('finished_at_utc', ''))} | Modus: {last_pipeline_result.get('mode', '-') }"
    )
    if last_pipeline_log:
        with st.expander("Pipelinelog anzeigen", expanded=False):
            st.code(last_pipeline_log, language="text")


def render_overview_messages() -> None:
    overview_flash_message = st.session_state.pop("overview_flash_message", None)
    overview_error_message = st.session_state.pop("overview_error_message", None)

    if overview_flash_message:
        st.success(overview_flash_message)
    if overview_error_message:
        st.error(overview_error_message)


def render_overview_page() -> None:
    rows = get_ticket_index()

    st.title("KI Ticket Pilot")
    st.caption("Übersicht über alle klassifizierten Tickets sowie Startpunkt für Abruf, Klassifikation und Ticketanlage.")

    render_pipeline_controls()
    render_overview_messages()

    if not rows:
        st.info(f"Im Verzeichnis {Path(TICKETS_DIR)} wurden noch keine TICKET-JSON-Dateien gefunden.")
        return

    filtered_rows = apply_filters(rows)
    raw_df, display_df = build_display_dataframe(filtered_rows)

    metric_columns = st.columns(4)
    average_confidence = raw_df["average_confidence"].dropna().mean() if not raw_df.empty else None
    low_confidence_count = int((raw_df["average_confidence"].fillna(1.0) < 0.85).sum()) if not raw_df.empty else 0
    manual_count = int(raw_df["manually_edited"].sum()) if not raw_df.empty else 0

    metric_columns[0].metric("Gefilterte Tickets", len(raw_df))
    metric_columns[1].metric("Ø Konfidenz", format_confidence(average_confidence) if average_confidence is not None else "-")
    metric_columns[2].metric("Manuell geändert", manual_count)
    metric_columns[3].metric("Unter 85% Konfidenz", low_confidence_count)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        type_counts = raw_df["ticket_type"].replace("", "Unbekannt").value_counts()
        render_distribution_chart(type_counts, "Verteilung nach Ticket-Typ")

    with chart_right:
        area_counts = raw_df["area"].replace("", "Unbekannt").value_counts().head(10)
        render_distribution_chart(area_counts, "Verteilung nach Bereich", label_formatter=format_area_display)

    st.subheader("Ticketliste")
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="ticket_overview_table",
        column_config={
            "Titel": st.column_config.TextColumn(width="large"),
            "Absender": st.column_config.TextColumn(width="medium"),
            "Empfangen": st.column_config.TextColumn(width="small"),
            "Typ": st.column_config.TextColumn(width="medium"),
            "Bereich": st.column_config.TextColumn(width="medium"),
            "Priorität": st.column_config.TextColumn(width="small"),
            "Schweregrad": st.column_config.TextColumn(width="medium"),
            "Ø Konfidenz": st.column_config.TextColumn(width="small"),
            "Manuell geändert": st.column_config.TextColumn(width="small"),
            "Beschreibung": st.column_config.TextColumn(width="large"),
        },
    )

    selected_rows = list(event.selection.rows)
    if not selected_rows:
        return

    selected_df = raw_df.iloc[selected_rows]
    selected_ids = selected_df["ticket_id"].tolist()
    selected_count = len(selected_ids)

    info_col, open_col, create_col = st.columns([5, 1, 1])
    if selected_count == 1:
        selected_row = selected_df.iloc[0]
        info_col.info(
            f"Ausgewählt: {selected_row['title']} | Typ: {selected_row['ticket_type'] or '-'} | Bereich: {format_area_display(selected_row['area']) or '-'}"
        )
    else:
        info_col.info(f"{selected_count} Tickets ausgewählt.")

    if open_col.button(
        "Ticket öffnen",
        use_container_width=True,
        type="primary",
        disabled=selected_count != 1,
    ):
        st.session_state["selected_ticket_id"] = selected_ids[0]
        st.switch_page(DETAIL_PAGE)

    create_label = "Ticket anlegen" if selected_count == 1 else f"{selected_count} Tickets anlegen"
    if create_col.button(create_label, use_container_width=True, disabled=selected_count == 0):
        move_result = move_tickets_to_rpa_inbox(selected_ids)
        clear_ticket_cache()

        moved_count = len(move_result["moved"])
        error_count = len(move_result["errors"])

        if moved_count:
            st.session_state["overview_flash_message"] = (
                f"{moved_count} Ticket(s) wurden in die RPA-Inbox verschoben: {Path(RPA_INBOX_DIR)}"
            )
        if error_count:
            error_lines = [
                f"{entry['ticket_id']}: {entry['error']}"
                for entry in move_result["errors"]
            ]
            st.session_state["overview_error_message"] = "Fehler bei der Ticketanlage:\n" + "\n".join(error_lines)

        if st.session_state.get("selected_ticket_id") in selected_ids:
            st.session_state.pop("selected_ticket_id", None)

        st.rerun()


def render_detail_page() -> None:
    rows = get_ticket_index()
    flash_message = st.session_state.pop("ticket_flash_message", None)

    selected_ticket_id = st.session_state.get("selected_ticket_id")

    title_col, action_col = st.columns([4, 1])
    title_col.title("Ticketdetail und Nachbearbeitung")
    title_col.caption("Pflichtfelder prüfen, bei Bedarf korrigieren und abspeichern.")

    if action_col.button("Zur Übersicht", use_container_width=True):
        st.switch_page(OVERVIEW_PAGE)

    if flash_message:
        st.success(flash_message)

    if not selected_ticket_id:
        st.warning("Es wurde noch kein Ticket aus der Übersicht ausgewählt.")
        return

    loaded = load_ticket_record_by_id(selected_ticket_id, TICKETS_DIR)
    if loaded is None:
        st.error(f"Das Ticket mit der ID {selected_ticket_id} wurde nicht gefunden.")
        return

    source_path, record = loaded
    editable_ticket = build_editable_ticket(record)
    option_map = collect_options(rows)

    email = record.get("email", {})
    meta = record.get("meta", {})
    ticket = record.get("ticket", {})

    meta_top = st.columns(2)
    meta_top[0].text_input("Message-ID", value=meta.get("message_id", "-"), disabled=True)
    meta_top[1].text_input("Dateiname", value=source_path.name, disabled=True)

    meta_bottom = st.columns(2)
    meta_bottom[0].text_input("Empfangen", value=format_timestamp(email.get("received_utc", "")), disabled=True)
    meta_bottom[1].text_input("Absender", value=email.get("sender", "-"), disabled=True)

    confidence_rows = build_classification_overview(record)
    if confidence_rows:
        confidence_df = pd.DataFrame(confidence_rows)
        score_columns = st.columns(len(confidence_rows))
        for column, confidence_row in zip(score_columns, confidence_rows):
            column.metric(confidence_row["Modell"], format_confidence(float(confidence_row["Konfidenz"])))
    else:
        confidence_df = pd.DataFrame(columns=["Modell", "Vorhersage", "Konfidenz", "Alternative 1", "Alternative 2", "Modellpfad"])

    area_options, area_index = build_select_state(option_map["area"], editable_ticket["Area"])
    ticket_type_options, ticket_type_index = build_select_state(option_map["ticket_type"], editable_ticket["Ticket-Type"])
    environment_options, environment_index = build_select_state(option_map["environment"], editable_ticket["Environment"])
    priority_options, priority_index = build_select_state(option_map["priority"], editable_ticket["Prio"])
    impact_options, impact_index = build_select_state(option_map["impact"], editable_ticket["Impact"])

    st.subheader("Vorausgefüllte Pflichtfelder")
    with st.form("ticket_edit_form"):
        first_row = st.columns(2)
        title_value = first_row[0].text_input("Titel", value=editable_ticket["Title"])
        area_value = first_row[1].selectbox(
            "Bereichspfad",
            options=area_options,
            index=area_index,
            format_func=format_area_display,
        )

        second_row = st.columns(2)
        ticket_type_value = second_row[0].selectbox(
            "Ticket-Typ",
            options=ticket_type_options,
            index=ticket_type_index,
        )
        iteration_value = second_row[1].text_input("Iteration", value=editable_ticket["Iteration"])

        third_row = st.columns(3)
        environment_value = third_row[0].selectbox(
            "Umgebung",
            options=environment_options,
            index=environment_index,
        )
        priority_value = third_row[1].selectbox(
            "Priorität",
            options=priority_options,
            index=priority_index,
        )
        impact_value = third_row[2].selectbox(
            "Schweregrad",
            options=impact_options,
            index=impact_index,
        )

        description_value = st.text_area("Beschreibung", value=editable_ticket["Description"], height=260)

        submitted = st.form_submit_button("Änderungen speichern", type="primary", use_container_width=True)

    if submitted:
        changed_fields = update_ticket_record(
            ticket_id=selected_ticket_id,
            updated_ticket={
                "Title": title_value,
                "Area": area_value,
                "Iteration": iteration_value,
                "Description": description_value,
                "Ticket-Type": ticket_type_value,
                "Environment": environment_value,
                "Prio": priority_value,
                "Impact": impact_value,
            },
            ticket_dir=TICKETS_DIR,
        )
        clear_ticket_cache()
        if changed_fields:
            field_list = ", ".join(changed_fields.keys())
            st.session_state["ticket_flash_message"] = f"Ticket gespeichert. Geänderte Felder: {field_list}"
        else:
            st.session_state["ticket_flash_message"] = "Es wurden keine Änderungen erkannt."
        st.rerun()

    expander_left, expander_right = st.columns(2)

    with expander_left:
        with st.expander("Modellvorhersagen und Konfidenzen", expanded=True):
            display_confidence_df = confidence_df.copy()
            if "Konfidenz" in display_confidence_df:
                display_confidence_df["Konfidenz"] = display_confidence_df["Konfidenz"].apply(format_confidence)
            for column_name in ["Vorhersage", "Alternative 1", "Alternative 2"]:
                if column_name in display_confidence_df:
                    display_confidence_df[column_name] = display_confidence_df[column_name].apply(
                        lambda value: format_area_display(value) if isinstance(value, str) and "SEU\\" in value else value
                    )
            st.dataframe(
                display_confidence_df,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Originale Mail", expanded=False):
            st.markdown(f"**Betreff:** {email.get('subject', '-')}")
            st.markdown(f"**Absender:** {email.get('sender', '-')}")
            st.markdown(f"**Empfangen:** {format_timestamp(email.get('received_utc', ''))}")
            st.text_area("Mail-Body", value=email.get("body", ""), height=320, disabled=True)

    with expander_right:
        with st.expander("Bereinigter Text für die Klassifikation", expanded=False):
            st.text_area(
                "Text for classification",
                value=email.get("text_for_classification", ""),
                height=320,
                disabled=True,
            )

        with st.expander("Aktueller Ticketzustand", expanded=False):
            display_ticket = dict(ticket)
            if "Area" in display_ticket:
                display_ticket["Area"] = format_area_display(display_ticket["Area"])
            st.json(display_ticket)

        manual_review = record.get("manual_review", {})
        history = manual_review.get("history", [])
        if history:
            with st.expander("Manuelle Änderungshistorie", expanded=False):
                history_rows = []
                for entry in history:
                    for field_name, payload in entry.get("changed_fields", {}).items():
                        history_rows.append(
                            {
                                "Zeitpunkt": format_timestamp(entry.get("edited_at_utc", "")),
                                "Feld": field_name,
                                "Alt": payload.get("old", ""),
                                "Neu": payload.get("new", ""),
                            }
                        )
                history_df = pd.DataFrame(history_rows)
                if not history_df.empty:
                    history_df["Alt"] = history_df["Alt"].apply(
                        lambda value: format_area_display(value) if isinstance(value, str) and "SEU\\" in value else value
                    )
                    history_df["Neu"] = history_df["Neu"].apply(
                        lambda value: format_area_display(value) if isinstance(value, str) and "SEU\\" in value else value
                    )
                st.dataframe(history_df, use_container_width=True, hide_index=True)


st.set_page_config(
    page_title="KI Ticket Pilot",
    page_icon=":material/confirmation_number:",
    layout="wide",
    initial_sidebar_state="expanded",
)

OVERVIEW_PAGE = st.Page(render_overview_page, title="Ticketübersicht", icon=":material/dashboard:", default=True)
DETAIL_PAGE = st.Page(
    render_detail_page,
    title="Ticketdetail",
    icon=":material/edit_note:",
    url_path="ticketdetail",
    visibility="hidden",
)

navigation = st.navigation([OVERVIEW_PAGE, DETAIL_PAGE])
navigation.run()
