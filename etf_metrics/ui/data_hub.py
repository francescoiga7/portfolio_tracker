# -*- coding: utf-8 -*-
"""UI: Gestione Dati di Mercato (download incrementale nel DB SQLite).

Scarica una sola volta lo storico dei ticker richiesti; ai giri successivi
aggiorna SOLO il delta (le righe successive all'ultima data salvata).
"""
import pandas as pd
import streamlit as st

from etf_metrics.core.data_manager import MarketDataManager
from etf_metrics.core.data_updater import update_tickers
from etf_metrics.clients.yahoo_client import (
    is_rate_limited,
    rate_limit_wait_seconds,
)


def _parse_tickers(text: str) -> list:
    """Esegue il parsing del testo della textarea: un ticker per riga, o separati da virgola/spazi."""
    raw = (text or "").replace(",", " ").replace(";", " ").split()
    return list(dict.fromkeys(t.strip().upper() for t in raw if t.strip()))


def _render_rate_limit_banner():
    if is_rate_limited():
        wait = rate_limit_wait_seconds()
        st.warning(
            f"⏳ Yahoo sta limitando le richieste (HTTP 429): i download sono sospesi "
            f"per altri ~{max(1, round(wait / 60))} minuti. Le letture dal DB continuano "
            f"a funzionare normalmente."
        )


def _render_report(report: dict):
    """Rende il report di update_tickers: metriche, tabella risultato, saltati."""
    results = report.get('results') or {}
    skipped = report.get('skipped') or {}

    updated = len(results)
    new_rows = sum(r.get('new_rows', 0) for r in results.values())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker aggiornati", f"{updated}/{report.get('requested', 0)}")
    c2.metric("Righe nuove totali", f"{new_rows:,}")
    c3.metric("Saltati", len(skipped))
    c4.metric("Tempo", f"{report.get('elapsed_sec', 0):.1f}s")

    if results:
        rows = []
        for t, r in sorted(results.items()):
            rows.append({
                "Ticker": t,
                "Modalità": "⚡ Delta" if r.get('mode') == 'delta' else "🔄 Completo",
                "Righe nuove": r.get('new_rows', 0),
                "Righe totali": r.get('total_rows', 0),
                "Dal": str(r.get('first', ''))[:10] if r.get('first') else "—",
                "Al": str(r.get('last', ''))[:10] if r.get('last') else "—",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if skipped:
        sk = pd.DataFrame(
            [{"Ticker": t, "Motivo": m} for t, m in sorted(skipped.items())])
        with st.expander(f"⏭️ {len(skipped)} ticker saltati", expanded=True):
            st.dataframe(sk, width="stretch", hide_index=True)

    if report.get('rate_limited'):
        st.error(
            "Yahoo ha risposto 429 (Too Many Requests): il download è stato interrotto. "
            f"Riprova tra ~{max(1, round(rate_limit_wait_seconds() / 60))} minuti: i dati "
            "già scaricati sono comunque stati salvati nel DB."
        )


def _run_update(tickers: list, period: str, force_full: bool, skip_updated_today: bool):
    """Esegue l'aggiornamento con barra di avanzamento e rende il report."""
    bar = st.progress(0.0, text="Preparazione...")
    status = st.empty()

    def cb(done, total, label):
        frac = min(1.0, done / total) if total else 1.0
        bar.progress(frac, text=f"{label} — {int(frac * 100)}%")
        status.caption(label)

    try:
        report = update_tickers(
            tickers, period=period, force_full=force_full,
            skip_updated_today=skip_updated_today, progress_callback=cb)
    finally:
        bar.empty()
        status.empty()

    _render_report(report)


def render_data_hub_ui():
    st.title("📥 Gestione Dati di Mercato")
    st.markdown(
        "Scarica **una sola volta** lo storico dei tuoi ticker nel database locale "
        "(SQLite, ottimizzato in modalità WAL): ai giri successivi viene aggiornato "
        "**solo il delta** — le righe successive all'ultima data già salvata."
    )

    with st.expander("ℹ️ Come funziona", expanded=False):
        st.markdown(
            """
- **Prima volta** (ticker non nel DB): download completo dello storico scelto,
  in batch paralleli, poi salvataggio nel DB.
- **Giri successivi**: per ogni ticker viene scaricata solo la parte mancante
  (dall'ultima data salvata a oggi, ultima giornata inclusa per correggere
  eventuali prezzi parziali) e fusa nel DB con un upsert.
- **Ticker senza dati** ("no data found"): memorizzati in cache negativa e non
  riscaricati; puoi ispezionarli e resettarli in fondo alla pagina.
- **Rate limit Yahoo (429)**: il download si sospende per ~10 minuti invece di
  martellare il server; i dati già scaricati vengono comunque salvati.
- Il DB è un unico file `market_data.db` (percorso configurabile con la
  variabile d'ambiente `ETF_METRICS_DB`).
            """
        )

    _render_rate_limit_banner()

    dm = MarketDataManager()
    stats = dm.get_db_stats()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ticker nel DB", f"{stats['tickers']:,}")
    c2.metric("Righe di prezzi", f"{stats['rows']:,}")
    c3.metric("Ultima sincronizzazione", str(stats['last_sync'] or "—"))
    c4.metric("Dimensione DB", f"{stats['db_mb']:.1f} MB")
    c5.metric("Cache negativa", stats['failed'])

    st.divider()

    # ------------------------------------------------------------------
    # Download / aggiornamento per lista di ticker
    # ------------------------------------------------------------------
    st.subheader("⬇️ Scarica / Aggiorna ticker")

    col_input, col_opts = st.columns([3, 2])
    with col_input:
        tickers_text = st.text_area(
            "Ticker (uno per riga, o separati da virgola)",
            placeholder="EXSA\nE909\nBMW\nSPY\nBRK-B",
            height=120,
        )
    with col_opts:
        mode = st.radio(
            "Modalità",
            ["⚡ Delta (solo righe mancanti)", "🔄 Riscaricamento completo"],
            help="Delta: scarica solo i dati successivi all'ultima data nel DB "
                 "(consigliato). Completo: riscarica tutto lo storico.",
        )
        period = st.selectbox(
            "Storico per ticker nuovi",
            ["5y", "10y", "2y", "max"],
            index=0,
            help="Usato solo per i ticker non ancora presenti nel DB.",
        )

    if st.button("⬇️ Scarica / Aggiorna", type="primary", disabled=is_rate_limited()):
        tickers = _parse_tickers(tickers_text)
        if not tickers:
            st.warning("Inserisci almeno un ticker.")
        else:
            st.info(f"Richiesti {len(tickers)} ticker: {', '.join(tickers[:20])}"
                    + (" …" if len(tickers) > 20 else ""))
            _run_update(
                tickers,
                period=period,
                force_full=(mode.startswith("🔄")),
                skip_updated_today=False,
            )

    # ------------------------------------------------------------------
    # Aggiornamento di tutto il DB
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🔄 Aggiorna tutto il DB")

    with st.expander("Aggiornamento dell'intero database (solo delta)", expanded=False):
        force_all = st.checkbox(
            "Forza anche i ticker già aggiornati oggi",
            value=False,
            help="Di default i ticker sincronizzati oggi vengono saltati per non "
                 "sprecare richieste (utile con migliaia di ticker).",
        )
        if st.button(
            f"🌍 Aggiorna tutti i {stats['tickers']:,} ticker nel DB",
            disabled=is_rate_limited() or stats['tickers'] == 0,
        ):
            all_tickers = dm.get_all_tickers()
            if not all_tickers:
                st.warning("Il DB non contiene ticker.")
            else:
                st.caption(f"Aggiornamento delta di {len(all_tickers)} ticker…")
                _run_update(
                    all_tickers,
                    period=period,
                    force_full=False,
                    skip_updated_today=not force_all,
                )

    # ------------------------------------------------------------------
    # Contenuto del DB
    # ------------------------------------------------------------------
    st.divider()
    with st.expander(f"🗄️ Contenuto del DB ({stats['tickers']:,} ticker)", expanded=False):
        info = dm.get_all_tickers_info()
        if info.empty:
            st.caption("Il DB è ancora vuoto: scarica i primi ticker qui sopra.")
        else:
            st.dataframe(info, width="stretch", hide_index=True)

    # ------------------------------------------------------------------
    # Cache negativa
    # ------------------------------------------------------------------
    with st.expander(f"🚫 Ticker in cache negativa ({stats['failed']})", expanded=False):
        failed_info = dm.get_failed_tickers_info()
        if failed_info.empty:
            st.caption("Nessun ticker in cache negativa.")
        else:
            st.dataframe(failed_info, width="stretch", hide_index=True)
            if st.button("♻️ Resetta la cache negativa", key="dh_reset_blacklist"):
                removed = dm.clear_failed_tickers()
                st.success(f"Rimossi {removed} ticker dalla cache negativa.")
                st.rerun()
