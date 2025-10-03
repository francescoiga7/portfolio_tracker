# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import datetime

# Import dalle utility esistenti
from etf_metrics.core.pac_screener import fetch_screener_data
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.metrics import calculate_short_term_indicators, get_tactical_buy_signal, monitor_simulated_trade
from etf_metrics.shared.config import TACTICAL_TRADING_SEED_QUERIES  # Importa le nuove query


def render_trading_ui():
    st.title("🎯 Trading Tattico di Breve Periodo")
    st.caption("Identifica segnali di acquisto su azioni ed ETF e simula operazioni di trading.")

    # Inizializzazione dello stato
    if 'simulated_trades' not in st.session_state:
        st.session_state.simulated_trades = {}
    if 'tactical_universe_data' not in st.session_state:
        st.session_state.tactical_universe_data = None
    if 'tactical_log' not in st.session_state:
        st.session_state.tactical_log = []

    with st.expander("📖 Metodologia e Funzionamento"):
        st.markdown("""
        Questa sezione combina la scoperta di un vasto universo di strumenti con l'analisi di segnali di trading a breve termine.

        **Fase 1: Caricamento dell'Universo**
        Puoi scegliere tra due modalità:
        - **Scoperta Automatica**: Sfrutta un elenco di query mirate per scoprire le azioni e gli ETF più rilevanti sui mercati USA ed europei.
        - **Inserimento Manuale**: Fornisci una tua lista di ISIN o Ticker specifici.

        **Fase 2: Ricerca dei Segnali**
        Sull'universo caricato, l'applicazione cerca segnali di "Compra" basati su una **confluenza di tre indicatori**:
        1.  **Crossover Rialzista delle Medie Mobili Esponenziali (EMA 12/26)**
        2.  **Uscita dall'Ipervenduto (Oscillatore Stocastico)**
        3.  **Conferma dei Volumi**

        Puoi quindi simulare l'acquisto e monitorare la posizione virtuale con strategie di uscita personalizzabili (Take Profit, Stop Loss, etc.).
        """)

    # --- Sidebar ---
    st.sidebar.header("⚙️ Parametri di Trading")

    with st.sidebar.expander("1. Carica Universo di Analisi", expanded=True):
        source_mode = st.radio("Modalità di Ricerca", ["Scoperta Automatica", "Inserimento Manuale"])
        specific_items_input = ""
        if source_mode == "Inserimento Manuale":
            specific_items_input = st.text_area(
                "Elenco ISIN/Ticker (uno per riga)",
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

    if st.session_state.tactical_universe_data is not None:
        st.sidebar.header("2. Strategia di Uscita")
        take_profit = st.sidebar.number_input("Take Profit (%)", 1.0, value=10.0, step=1.0)
        stop_loss = st.sidebar.number_input("Stop Loss (%)", 1.0, value=5.0, step=1.0)
        trailing_stop = st.sidebar.number_input("Trailing Stop Loss (%)", 0.0, value=7.0, step=1.0,
                                                help="Imposta a 0 per disattivarlo.")

        if st.sidebar.button("Cerca Segnali Tattici"):
            st.session_state.tactical_log = []  # Pulisce il log per la nuova analisi
            universe_data = st.session_state.tactical_universe_data
            signals = []
            progress_bar = st.progress(0, text="Analisi segnali in corso...")

            for i, data in enumerate(universe_data):
                ticker = data['ticker']
                series_df = data['series']

                progress_bar.progress((i + 1) / len(universe_data), text=f"Analisi: {ticker}")

                if series_df is None or not isinstance(series_df, pd.DataFrame) or series_df.empty or len(
                        series_df) < 30:
                    st.session_state.tactical_log.append(f"⚠️ **{ticker}**: Dati storici insufficienti. Saltato.")
                    continue

                indicators = calculate_short_term_indicators(series_df)
                buy_signal = get_tactical_buy_signal(indicators)

                if buy_signal:
                    signals.append({"ticker": ticker, "price": series_df['Close'].iloc[-1], **buy_signal})
                    st.session_state.tactical_log.append(f"✅ **{ticker}**: Segnale di ACQUISTO trovato!")
                else:
                    st.session_state.tactical_log.append(f"⚪ **{ticker}**: Nessuna condizione di acquisto soddisfatta.")

            progress_bar.empty()
            st.session_state.tactical_signals = signals
            st.rerun()

    # --- Area Principale ---

    if not st.session_state.tactical_universe_data:
        st.info("👈 Inizia caricando un universo di strumenti dalla barra laterale.")

    if 'tactical_signals' in st.session_state:
        st.subheader("🚨 Segnali di Acquisto Identificati Oggi")
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
                        st.success(f"Acquisto di {signal['ticker']} simulato a {signal['price']:.2f}!")
                        st.rerun()
        # Messaggio mostrato solo se la ricerca è stata fatta e non ha prodotto segnali
        elif st.session_state.tactical_universe_data:
            st.info("Nessun segnale di acquisto tattico identificato oggi nell'universo caricato.")

    st.subheader("📈 Posizioni Simulate Aperte")
    if st.session_state.simulated_trades:
        trades_to_close = []
        for ticker, trade_info in st.session_state.simulated_trades.items():
            if trade_info['status'] == "Aperta":
                series_df = get_series(ticker, period="1y", as_dataframe=True)
                if series_df is None or series_df.empty:
                    st.warning(f"Dati non disponibili per monitorare {ticker}.")
                    continue

                current_price = series_df['Close'].iloc[-1]
                pnl_pct = (current_price / trade_info['purchase_price'] - 1) * 100

                sell_signal = monitor_simulated_trade(
                    trade_info['purchase_price'], series_df['Close'],
                    take_profit, stop_loss, trailing_stop
                )

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Ticker", ticker)
                col2.metric("Prezzo Acquisto", f"{trade_info['purchase_price']:.2f}")
                col3.metric("Prezzo Attuale", f"{current_price:.2f}")
                col4.metric("P&L Non Realizzato", f"{pnl_pct:.2f}%")

                if sell_signal:
                    with col5:
                        st.error(f"**{sell_signal['signal']}**: {sell_signal['reason']}")
                        if st.button("Chiudi Posizione", key=f"close_{ticker}"):
                            trades_to_close.append(ticker)
                else:
                    col5.success("Mantieni")

        for ticker_to_close in trades_to_close:
            st.session_state.simulated_trades[ticker_to_close]['status'] = "Chiusa"
            st.info(f"Posizione su {ticker_to_close} chiusa.")
            st.rerun()
    else:
        st.info("Nessuna posizione simulata aperta.")

    # Visualizzazione del log
    if st.session_state.tactical_log:
        with st.expander("🔍 Log Dettagliato dell'Analisi"):
            st.markdown("\n".join(f"- {entry}" for entry in st.session_state.tactical_log))