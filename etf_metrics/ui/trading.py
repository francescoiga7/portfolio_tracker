# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

from etf_metrics.core.pac_screener import fetch_screener_data
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.metrics import get_monday_buy_signal, monitor_weekly_trade
from etf_metrics.shared.config import TACTICAL_TRADING_SEED_QUERIES

TRADES_JSON_FILE = "simulated_trades.json"


def save_simulated_trades():
    """Salva le operazioni simulate in un file JSON."""
    try:
        path = Path(TRADES_JSON_FILE)
        trades = st.session_state.get('simulated_trades', {})
        path.write_text(json.dumps(trades, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Errore durante il salvataggio delle operazioni: {e}")


def load_simulated_trades_once():
    """Carica le operazioni simulate da un file JSON una sola volta per sessione."""
    if st.session_state.get("_trades_loaded_once"):
        return
    path = Path(TRADES_JSON_FILE)
    if path.exists():
        try:
            trades = json.loads(path.read_text(encoding="utf-8"))
            st.session_state.simulated_trades = trades
        except Exception as e:
            st.error(f"Errore durante il caricamento delle operazioni: {e}")
            st.session_state.simulated_trades = {}
    st.session_state["_trades_loaded_once"] = True


def render_trading_ui():
    st.title("💹 Trading Tattico Settimanale")
    st.caption("Identifica segnali di acquisto al lunedì e simula operazioni con vendita al venerdì, basato su un modello quantitativo.")

    # Inizializzazione e caricamento dello stato
    if 'simulated_trades' not in st.session_state:
        st.session_state.simulated_trades = {}
    load_simulated_trades_once()

    if 'tactical_universe_data' not in st.session_state:
        st.session_state.tactical_universe_data = None
    if 'tactical_log' not in st.session_state:
        st.session_state.tactical_log = []

    with st.expander("📖 Metodologia e Funzionamento"):
        st.markdown("""
        Questa sezione implementa una strategia di **swing trading settimanale** basata sull'anomalia del "weekend effect", arricchita con filtri quantitativi per aumentare le probabilità di successo.

        **Fase 1: Caricamento dell'Universo**
        Puoi scoprire un vasto universo di azioni ed ETF tramite la **Scoperta Automatica** o analizzare una tua **lista manuale** di Ticker.

        **Fase 2: Ricerca dei Segnali di "Monday Buy"**
        L'applicazione cerca segnali di acquisto solo di **lunedì**, basandosi su una **confluenza di 4 fattori**:
        1.  **Regime di Mercato Favorevole**: Il VIX (indice di volatilità) deve essere basso, per evitare di comprare durante fasi di panico.
        2.  **Trend di Fondo Rialzista**: Il prezzo dell'asset deve trovarsi al di sopra della sua media mobile a 50 giorni (SMA50).
        3.  **Condizione "Weekend Effect"**: Il rendimento del venerdì precedente deve essere stato negativo, aumentando le probabilità di un prezzo di ingresso vantaggioso.
        4.  **Conferma dei Volumi**: I volumi di scambio del lunedì devono essere superiori alla media per confermare l'interesse degli acquirenti.

        **Fase 3: Simulazione e Gestione della Posizione**
        Una volta simulato un acquisto, la posizione viene monitorata con strategie di uscita chiare:
        - **Vendita il Venerdì**: L'obiettivo è chiudere la posizione il venerdì, ma solo se il profitto atteso copre i costi di transazione (commissioni) e le tasse.
        - **Stop Loss**: Protezione dal ribasso con un limite di perdita predefinito.
        - **Uscita Tecnica Anticipata**: Se il trend di breve termine si inverte (es. prezzo sotto la media a 10 giorni), viene suggerita un'uscita per proteggere il capitale.
        """)

    # --- Sidebar ---
    st.sidebar.header("⚙️ Parametri di Trading")

    with st.sidebar.expander("1. Carica Universo di Analisi", expanded=True):
        source_mode = st.radio("Modalità di Ricerca", ["Scoperta Automatica", "Inserimento Manuale"])
        specific_items_input = ""
        if source_mode == "Inserimento Manuale":
            specific_items_input = st.text_area(
                "Elenco Ticker (uno per riga)",
                "ALAB\nHOOD\nCLS\nSOFI\nANET\nNVDA\nCRWV\nAPLD\nNBIS\nORCL",
                height=150
            )

        if st.button("Carica/Aggiorna Dati Universo"):
            st.session_state.tactical_universe_data, st.session_state.tactical_signals, st.session_state.tactical_log = None, [], []

            specific_items = None
            if source_mode == "Inserimento Manuale" and specific_items_input.strip():
                specific_items = [item.strip().upper() for item in specific_items_input.split('\n') if item.strip()]

            with st.spinner("Caricamento dati storici... L'operazione potrebbe richiedere alcuni minuti."):
                log_list = []
                instrument_types = ["EQUITY", "ETF", "ETP", "ETN"]
                queries = TACTICAL_TRADING_SEED_QUERIES if source_mode == "Scoperta Automatica" else []

                st.session_state.tactical_universe_data = fetch_screener_data(
                    log_area=log_list,
                    specific_isins=specific_items,
                    instrument_types=instrument_types,
                    queries=queries
                )
            st.success(f"Universo di {len(st.session_state.tactical_universe_data)} strumenti caricato!")
            st.session_state.tactical_log.extend(log_list)
            st.rerun()

    st.sidebar.header("2. Strategia di Uscita")
    stop_loss = st.sidebar.number_input("Stop Loss (%)", 1.0, 100.0, 10.0, 1.0)
    commission = st.sidebar.number_input("Commissioni per operazione (€)", 0.0, 100.0, 1.0, 0.5)
    tax_rate = st.sidebar.number_input("Tassazione plusvalenze (%)", 0.0, 100.0, 26.0, 1.0)

    if st.session_state.tactical_universe_data is not None:
        if st.sidebar.button("Cerca Segnali 'Monday Buy'"):
            st.session_state.tactical_log = []
            universe_data = st.session_state.tactical_universe_data
            signals = []
            progress_bar = st.progress(0, text="Analisi segnali in corso...")

            with st.spinner("Caricamento dati di mercato (VIX)..."):
                vix_series = get_series("^VIX", period="1y")

            if vix_series is None or vix_series.empty:
                st.error("Impossibile caricare i dati del VIX. L'analisi non può procedere.")
                return

            for i, data in enumerate(universe_data):
                ticker = data['ticker']
                series_df = data['series']
                series_df.attrs['ticker'] = ticker

                progress_bar.progress((i + 1) / len(universe_data), text=f"Analisi: {ticker}")

                if series_df is None or not isinstance(series_df, pd.DataFrame) or series_df.empty or len(
                        series_df) < 51:
                    st.session_state.tactical_log.append(f"⚠️ **{ticker}**: Dati storici insufficienti. Saltato.")
                    continue

                buy_signal, log_messages = get_monday_buy_signal(series_df, vix_series)
                st.session_state.tactical_log.extend(log_messages)

                if buy_signal:
                    signals.append({"ticker": ticker, "price": series_df['Close'].iloc[-1], **buy_signal})

            progress_bar.empty()
            st.session_state.tactical_signals = signals
            st.rerun()

    # --- Area Principale ---
    if not st.session_state.tactical_universe_data and not st.session_state.simulated_trades:
        st.info("👈 Inizia caricando un universo di strumenti dalla barra laterale.")

    if 'tactical_signals' in st.session_state:
        st.subheader("🚨 Segnali di Acquisto 'Monday Buy' Identificati")
        signals = st.session_state.tactical_signals
        if signals:
            for signal in signals:
                col1, col2, col3, col4 = st.columns([2, 2, 4, 2])
                with col1:
                    st.metric("Ticker", signal['ticker'])
                with col2:
                    st.metric("Prezzo", f"{signal['price']:.2f}")
                with col3:
                    st.info(f"**Motivazione:** {signal['reason']}")
                with col4:
                    if st.button("Simula Acquisto", key=f"buy_{signal['ticker']}"):
                        st.session_state.simulated_trades[signal['ticker']] = {
                            "purchase_price": signal['price'],
                            "purchase_date": datetime.now().strftime("%Y-%m-%d"),
                            "status": "Aperta"
                        }
                        save_simulated_trades()
                        st.success(f"Acquisto di {signal['ticker']} simulato a {signal['price']:.2f}!")
                        st.rerun()
        elif st.session_state.tactical_universe_data:
            st.info("Nessun segnale di acquisto 'Monday Buy' identificato oggi nell'universo caricato.")

    st.subheader("📈 Posizioni Simulate Aperte")
    open_trades_data = []
    if st.session_state.simulated_trades:
        for ticker, trade_info in st.session_state.simulated_trades.items():
            if trade_info.get('status') == "Aperta":
                series_df = get_series(ticker, period="1y", as_dataframe=True)
                if series_df is None or series_df.empty:
                    st.warning(f"Dati non disponibili per monitorare {ticker}.")
                    continue

                current_price = series_df['Close'].iloc[-1]
                pnl_pct = (current_price / trade_info['purchase_price'] - 1) * 100
                sell_signal = monitor_weekly_trade(
                    trade_info['purchase_price'], series_df['Close'],
                    commission, tax_rate, stop_loss
                )

                open_trades_data.append({
                    "Ticker": ticker,
                    "Data Acquisto": trade_info['purchase_date'],
                    "Prezzo Acquisto": trade_info['purchase_price'],
                    "Prezzo Attuale": current_price,
                    "P&L (%)": pnl_pct,
                    "Segnale Vendita": sell_signal['reason'] if sell_signal else "Mantieni"
                })

    if open_trades_data:
        df_display = pd.DataFrame(open_trades_data)

        def style_pnl(val):
            color = 'green' if val >= 0 else 'red'
            return f'color: {color};'

        def style_sell_signal(val):
            val_lower = val.lower()
            if "mantieni" in val_lower:
                return 'background-color: #28a745; color: white; font-weight: bold;'
            elif "vendi" in val_lower:
                return 'background-color: #dc3545; color: white; font-weight: bold;'
            return ''

        styler = df_display.style.map(style_pnl, subset=['P&L (%)']) \
            .map(style_sell_signal, subset=['Segnale Vendita']) \
            .format({
            "Prezzo Acquisto": "€{:,.2f}",
            "Prezzo Attuale": "€{:,.2f}",
            "P&L (%)": "{:,.2f}%"
        })

        st.dataframe(styler, hide_index=True, width="stretch")

        # Logica per chiudere le posizioni
        sell_candidates = [trade['Ticker'] for trade in open_trades_data if "Vendi" in trade['Segnale Vendita']]
        if sell_candidates:
            trades_to_close = st.multiselect(
                "Seleziona posizioni da chiudere",
                options=sell_candidates,
                label_visibility="collapsed"
            )
            if st.button("Chiudi Selezionate"):
                if trades_to_close:
                    for ticker_to_close in trades_to_close:
                        st.session_state.simulated_trades[ticker_to_close]['status'] = "Chiusa"
                    save_simulated_trades()
                    st.success(f"Posizioni chiuse per: {', '.join(trades_to_close)}")
                    st.rerun()
                else:
                    st.warning("Nessuna posizione selezionata per la chiusura.")
    else:
        st.info("Nessuna posizione simulata aperta.")

    # Visualizzazione del log
    if st.session_state.tactical_log:
        with st.expander("🔍 Log Dettagliato dell'Analisi"):
            st.markdown("\n".join(f"- {entry}" for entry in st.session_state.tactical_log))