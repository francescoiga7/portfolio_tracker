# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import date
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.trading import analyze_ticker
from etf_metrics.core.automated_backtest import run_market_aware_backtest

DEFAULT_TRADING_LIST = """NVDA
TSLA
AMD
COIN
MARA
PLTR
META
AMZN
NFLX
QQQ
TQQQ
SQQQ"""


def render_trading_ui():
    st.title("💹 Trading & Algorithmic Strategy")
    st.caption("Scanner di segnali operativi e backtest di strategie di Breakout.")

    with st.expander("🧠 Logica Operativa"):
        st.markdown("""
        Questa sezione identifica setup di **Breakout** e **Reversion** basati su volatilità e volumi.

        **1. Scanner Segnali (Trading)**
        Analizza il mercato alla data odierna (o passata) per trovare opportunità immediate.
        * 🟢 **LONG_BREAKOUT:** Prezzo rompe la Banda di Bollinger superiore con volumi.
        * 🔵 **LONG_DIP:** Trend rialzista ma prezzo in ritracciamento (RSI basso).

        **2. Automated Backtest**
        Simula l'esecuzione della strategia nel tempo gestendo un portafoglio virtuale.
        * **Gestione Uscita:** Esclusivamente **Trailing Stop Dinamico** basato sull'ATR (Volatility Stop).
        * **Money Management:** Dynamic Position Sizing basato sulla volatilità (VIX).
        """)

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Configurazione")

    # Scelta Modalità
    mode = st.sidebar.radio(
        "Modalità Operativa",
        ["📡 Scanner Segnali (Live/Storico)", "🤖 Automated Backtest"],
        help="Scegli se cercare segnali puntuali o simulare una strategia nel tempo."
    )

    st.sidebar.divider()

    # Input Tickers (Comune a entrambe le modalità)
    tickers_input = st.sidebar.text_area("Watchlist (Ticker Yahoo)", DEFAULT_TRADING_LIST, height=200)
    tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

    # Configurazione Specifica per Modalità
    analysis_date = date.today()
    start_date_backtest = date(2021, 1, 1)

    if mode == "📡 Scanner Segnali (Live/Storico)":
        st.sidebar.subheader("⏳ Time Travel")
        enable_time_travel = st.sidebar.checkbox("Abilita Analisi Storica", value=False)
        if enable_time_travel:
            analysis_date = st.sidebar.date_input(
                "Data Analisi:",
                date.today(),
                min_value=date(2020, 1, 1),
                max_value=date.today()
            )
            st.sidebar.warning(f"Analisi congelata al: {analysis_date.strftime('%d/%m/%Y')}")
        else:
            st.sidebar.caption("Analisi su dati Odierni.")

    elif mode == "🤖 Automated Backtest":
        st.sidebar.subheader("⏳ Periodo Simulazione")
        start_date_backtest = st.sidebar.date_input("Data Inizio:", pd.to_datetime("2021-01-01"))
        st.sidebar.info("La simulazione utilizzerà solo Trailing Stop dinamico come uscita.")

    # --- MAIN CONTENT ---

    if mode == "📡 Scanner Segnali (Live/Storico)":
        st.subheader("📡 Scanner di Mercato")

        if st.button("🔥 Scansiona Watchlist"):
            if not tickers:
                st.warning("Inserisci almeno un ticker nella sidebar.")
                return

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            cutoff_date = pd.to_datetime(analysis_date)

            for i, ticker in enumerate(tickers):
                status_text.text(f"Analisi {ticker}...")
                # Scarica un po' più di dati per calcolare le medie mobili correttamente
                full_df = get_series(ticker, period="5y", as_dataframe=True)

                if full_df is not None and not full_df.empty:
                    df_slice = full_df[full_df.index <= cutoff_date].copy()
                    if len(df_slice) > 200:
                        signal_data = analyze_ticker(ticker, df_slice)
                        if signal_data and signal_data['signal'] != "NEUTRAL":
                            results.append(signal_data)

                progress_bar.progress((i + 1) / len(tickers))

            status_text.empty()
            progress_bar.empty()

            date_label = analysis_date.strftime('%d/%m/%Y')
            st.markdown(f"### Risultati al **{date_label}**")

            if not results:
                st.info(f"Nessun segnale operativo trovato.")
                return

            st.success(f"Trovate {len(results)} opportunità!")
            results.sort(key=lambda x: x['confidence'], reverse=True)

            for res in results:
                signal_color = "green" if "LONG" in res['signal'] else "red"
                with st.container():
                    st.markdown(
                        f"**{res['ticker']}** : :{signal_color}[{res['signal']}] (Confidenza: {res['confidence']}/100)")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prezzo", f"{res['price']:.2f}")
                    c2.metric("Stop Loss (ATR)", f"{res['stop_loss']:.2f}")
                    c3.metric("RSI (14)", f"{res['indicators']['RSI']:.1f}")

                    with st.expander("Dettagli"):
                        st.write(f"**Motivazione:** {res['reason']}")
                    st.divider()

    elif mode == "🤖 Automated Backtest":
        st.subheader("🤖 Simulazione Portafoglio Algoritmico")

        if st.button("▶️ Avvia Simulazione"):
            if not tickers:
                st.warning("Inserisci almeno un ticker nella sidebar.")
                return

            with st.spinner(f"Simulazione in corso dal {start_date_backtest}..."):
                # Chiama la funzione corretta senza use_tp_only
                df_trades, final_cap = run_market_aware_backtest(tickers, start_date=str(start_date_backtest))

            if not df_trades.empty:
                initial = 1000
                total_return = ((final_cap - initial) / initial) * 100

                c1, c2, c3 = st.columns(3)
                c1.metric("Capitale Finale", f"€{final_cap:,.2f}")
                c2.metric("Rendimento Totale", f"{total_return:.2f}%",
                          delta_color="normal" if total_return > 0 else "inverse")

                closed_trades = df_trades[df_trades['Action'] == 'SELL']
                if not closed_trades.empty:
                    wins = len(closed_trades[closed_trades['PnL_Net'] > 0])
                    total_closed = len(closed_trades)
                    win_rate = (wins / total_closed) * 100
                    c3.metric("Win Rate", f"{win_rate:.1f}% ({wins}/{total_closed})")

                st.subheader("Giornale delle Operazioni")

                def style_trades(row):
                    action = row['Action']
                    if "BUY" in action: return ['background-color: #f0f2f6'] * len(row)
                    if "SELL" in action:
                        if row['PnL_Net'] > 0: return ['background-color: #d1e7dd; color: #0f5132'] * len(row)  # Verde
                        return ['background-color: #f8d7da; color: #842029'] * len(row)  # Rosso
                    return [''] * len(row)

                st.dataframe(
                    df_trades.style.apply(style_trades, axis=1)
                    .format({"Price": "{:.2f}", "PnL_Net": "{:+.2f}", "PnL_Pct": "{:+.2f}%", "Capital": "€{:,.0f}"}),
                    width="stretch",
                )
            else:
                st.warning("Nessun trade generato nel periodo selezionato.")