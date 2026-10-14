# -*- coding: utf-8 -*-
"""UI: Gestione Dati di Mercato (download incrementale nello store Parquet).

Scarica una sola volta lo storico dei ticker richiesti; ai giri successivi
aggiorna SOLO il delta (le righe successive all'ultima data salvata).
"""
import os

import pandas as pd
import streamlit as st

from etf_metrics.core.data_manager import MarketDataManager
from etf_metrics.core.data_updater import update_tickers
from etf_metrics.core.etf_search_engine import (
    load_universe_to_db, etf_universe_info)
from etf_metrics.clients.yahoo_client import (
    is_rate_limited,
    rate_limit_wait_seconds,
)
from etf_metrics.ui.components import render_page_help


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


def _render_migrate_to_parquet(dm):
    """Pulsante di migrazione dello storage SQLite legacy → Parquet."""
    with st.expander("🚀 Migra lo storage a Parquet (partizioni mensili)",
                     expanded=False):
        st.markdown(
            "Il DB SQLite legacy può essere convertito nello store **Parquet** "
            "partizionato per mese: formato colonnare aperto, letture più veloci "
            "e aggiornamenti incrementali che riscrivono solo i mesi coinvolti. "
            "Il file `.db` **non viene modificato**: resta come backup, ma l'app "
            "userà da subito il nuovo store `market_data_parquet/`."
        )
        if os.environ.get("ETF_METRICS_DB"):
            st.warning(
                "La variabile d'ambiente ETF_METRICS_DB è impostata e continua a "
                "puntare al DB SQLite: rimuovila (o puntala alla directory "
                "Parquet) dopo la migrazione, altrimenti l'app continuerà a usare "
                "il vecchio backend."
            )
        if st.button("🚀 Migra a Parquet", key="dh_migrate_parquet"):
            from etf_metrics.core.store_migration import migrate_sqlite_to_parquet
            try:
                with st.spinner("Migrazione in corso…"):
                    report = migrate_sqlite_to_parquet(dm.db_path)
                st.success(
                    f"Migrazione completata in {report['elapsed_sec']:.1f}s: "
                    f"{report['price_rows']:,} righe di prezzi, "
                    f"{report['tickers']} ticker, {report['months']} mesi → "
                    f"`{report['dest']}`"
                )
                st.rerun()
            except Exception as e:
                st.error(f"Migrazione non riuscita: {e}")


def _render_download_tab(dm, stats):
    """Tab ⬇️: download per lista di ticker + aggiornamento di tutto il DB."""
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

    st.divider()

    with st.expander(f"🔄 Aggiorna tutto il DB (solo delta) — {stats['tickers']:,} ticker"):
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


def _render_universe_tab():
    """Tab 🌍: discovery dell'universo ETF e download centralizzato nel DB."""
    uni_info = etf_universe_info()
    if uni_info['last_discovery']:
        st.caption(
            f"Ultima discovery: **{uni_info['last_discovery']}** — "
            f"**{uni_info['count']:,}** strumenti ETF/ETP/ETN "
            f"(listato preferito: Xetra `.DE`) salvati in `{uni_info['file']}`. "
            "Questo universo alimenta i consigli di acquisto della pagina "
            "🧭 Satelliti PAC.")
    else:
        st.caption(
            "Universo non ancora scoperto. La discovery cerca gli ETF/ETP/ETN europei "
            "con le query seed dell'app, deduplica per ISIN tenendo il listato "
            "tedesco (`.DE`) e salva tutto nel DB: è l'universo usato dalla sezione "
            "🧭 Satelliti PAC per i consigli di acquisto.")

    if st.button("🌍 Scopri / Aggiorna universo ETF", disabled=is_rate_limited()):
        with st.status("🌍 Discovery universo ETF...", expanded=True) as status:
            last = [0.0]

            def _prog(done, total, msg=""):
                try:
                    frac = (done / total) if total else 1.0
                    if frac - last[0] >= 0.1 or frac >= 1.0:
                        last[0] = frac
                        st.write(msg if msg else f"{done}/{total}")
                except Exception:
                    pass

            rep = load_universe_to_db(progress=_prog)
            status.update(label="Discovery completata", state="complete")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Strumenti trovati", f"{rep['found']:,}")
        m2.metric("Unici (dedup ISIN)", f"{rep['unique']:,}")
        m3.metric("Nuovi scaricati", f"{rep['downloaded']:,}")
        m4.metric("Già a punto", f"{rep['cached_today']:,}")
        m5.metric("Senza dati", f"{rep['failed']:,}")
        st.caption(f"Universo totale persistito: **{rep['universe_count']:,}** strumenti "
                   f"(ultima discovery: {rep['last_discovery'] or '—'}).")
        if rep.get('rate_limited'):
            st.warning("Rate limit Yahoo: il download è stato interrotto. Rilancia la "
                       "discovery più tardi per completare i ticker mancanti.")


def _render_db_content_tab(dm, stats):
    """Tab 🗄️: contenuto del DB, cache negativa e migrazione dello storage."""
    info = dm.get_all_tickers_info()
    if info.empty:
        st.caption("Il DB è ancora vuoto: scarica i primi ticker dalla tab "
                   "⬇️ Scarica / Aggiorna.")
    else:
        display = info.rename(columns={
            'ticker': 'Ticker', 'name': 'Nome',
            'first_added': 'Aggiunto il', 'last_updated': 'Ultimo agg.',
            'rows': 'Righe', 'first_date': 'Dal', 'last_date': 'Al',
            'last_price': 'Ultimo prezzo',
        }).fillna("—")
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption("🆕 **Aggiunto il** = prima volta che il ticker è entrato nel DB "
                   "(i ticker precedenti all'introduzione di questa colonna "
                   "mostrano — perché la data è sconosciuta).")

    with st.expander(f"🚫 Ticker in cache negativa ({stats['failed']})"):
        failed_info = dm.get_failed_tickers_info()
        if failed_info.empty:
            st.caption("Nessun ticker in cache negativa.")
        else:
            st.dataframe(failed_info, width="stretch", hide_index=True)
            if st.button("♻️ Resetta la cache negativa", key="dh_reset_blacklist"):
                removed = dm.clear_failed_tickers()
                st.success(f"Rimossi {removed} ticker dalla cache negativa.")
                st.rerun()

    if stats.get('backend') != 'parquet':
        _render_migrate_to_parquet(dm)


def render_data_hub_ui():
    st.title("📥 Gestione Dati di Mercato")
    st.caption("Scarica una sola volta lo storico dei ticker nel DB locale; ai giri "
               "successivi aggiorna solo il delta. Nessun'altra pagina scarica dati.")

    render_page_help(
        "**Cosa fa**: scarica lo storico dei prezzi nel DB locale (Parquet, partizionato per mese in `market_data_parquet/`).",
        "**A cosa serve**: rendere autonome tutte le altre pagine (indicatori e backtest senza download).",
        "Prima volta = storico completo; poi solo il **delta** fino a oggi, ultima candela inclusa.",
        "🌍 **Universo ETF**: scopre gli ETF/ETP/ETN europei (listato preferito Xetra `.DE`) e li scarica nel DB.",
        "Rate limit Yahoo (429) = pausa ~10 min; i ticker senza dati finiscono in cache negativa (resettabile in Contenuto del DB).",
    )

    _render_rate_limit_banner()

    dm = MarketDataManager()
    stats = dm.get_db_stats()

    # --- Panoramica del DB ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Ticker nel DB", f"{stats['tickers']:,}")
    c2.metric("Righe di prezzi", f"{stats['rows']:,}")
    c3.metric("Ultima sincronizzazione", str(stats['last_sync'] or "—"))
    c4, c5, c6 = st.columns(3)
    c4.metric("Dimensione DB", f"{stats['db_mb']:.1f} MB")
    c5.metric("Cache negativa", stats['failed'])
    c6.metric("Backend", "🧊 Parquet" if stats.get('backend') == 'parquet' else "SQLite")

    if stats.get('backend') == 'parquet':
        st.caption(
            f"🧊 Store Parquet in `{stats['path']}` — "
            f"{stats.get('partitions') or 0} partizioni mensili "
            "(`prices/month=YYYY-MM/`)"
        )

    # --- Tre aree: download, universo ETF, ispezione del DB ---
    tab_download, tab_universe, tab_db = st.tabs(
        ["⬇️ Scarica / Aggiorna", "🌍 Universo ETF", "🗄️ Contenuto del DB"])

    with tab_download:
        _render_download_tab(dm, stats)

    with tab_universe:
        _render_universe_tab()

    with tab_db:
        _render_db_content_tab(dm, stats)
