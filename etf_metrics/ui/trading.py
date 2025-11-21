# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import date
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.trading import analyze_ticker_alpha

# Titoli di default ad alta volatilità
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
    st.title("💹 Trading")
    st.caption("Algoritmo quantitativo per Swing Trading e Breakout di Volatilità.")

    with st.expander("🧠 Logica Operativa e Backtest"):
        st.markdown("""
        Questa sezione identifica setup di **Breakout** e **Reversion** basati su volatilità e volumi.

        **Funzionalità Time Travel:**
        Attivando il Backtest nella sidebar, puoi "tornare indietro nel tempo".
        L'algoritmo vedrà SOLO i dati disponibili fino a quella data. È utile per rispondere alla domanda:
        *"Se avessi usato questo algoritmo il 15 ottobre scorso, mi avrebbe dato il segnale giusto?"*

        **Legenda Segnali:**
        * 🟢 **LONG_BREAKOUT:** Prezzo rompe la Banda di Bollinger superiore con volumi.
        * 🔵 **LONG_DIP:** Trend rialzista ma prezzo in ritracciamento (RSI basso).
        * 🔴 **SHORT_BREAKDOWN:** Rottura violenta al ribasso dei supporti.
        """)

    st.sidebar.header("⚙️ Radar Settings")

    # --- SEZIONE TIME TRAVEL ---
    st.sidebar.subheader("⏳ Macchina del Tempo")
    enable_time_travel = st.sidebar.checkbox("Abilita Backtest Storico", value=False)

    analysis_date = date.today()
    if enable_time_travel:
        analysis_date = st.sidebar.date_input(
            "Analizza come se fosse il:",
            date.today(),
            min_value=date(2020, 1, 1),
            max_value=date.today()
        )
        st.sidebar.warning(f"⚠️ Analisi congelata al: {analysis_date.strftime('%d/%m/%Y')}")
    else:
        st.sidebar.caption("Analisi in tempo reale (Dati odierni)")

    tickers_input = st.sidebar.text_area("Watchlist (Ticker Yahoo)", DEFAULT_TRADING_LIST, height=200)

    if st.sidebar.button("🔥 Scansiona Mercato"):
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Data di cutoff per il slicing
        cutoff_date = pd.to_datetime(analysis_date)

        for i, ticker in enumerate(tickers):
            status_text.text(f"Analisi {ticker} al {analysis_date}...")

            # 1. SCARICO STORICO AMPIO
            # Scarichiamo 5 anni per essere sicuri di avere dati anche se l'utente va indietro nel tempo
            # e per calcolare correttamente la SMA200 alla data del backtest.
            full_df = get_series(ticker, period="5y", as_dataframe=True)

            if full_df is not None and not full_df.empty:
                # 2. TIME TRAVEL SLICING (Il trucco è qui)
                # Prendiamo solo i dati FINO alla data selezionata inclusa
                df_slice = full_df[full_df.index <= cutoff_date].copy()

                # Verifica: abbiamo abbastanza dati ALLA data del backtest?
                # Servono almeno 200 giorni prima della data scelta per le medie mobili
                if len(df_slice) > 200:
                    # L'algoritmo analizzerà l'ultima riga di df_slice (che è la data del backtest)
                    signal_data = analyze_ticker_alpha(ticker, df_slice)

                    if signal_data and signal_data['signal'] != "NEUTRAL":
                        results.append(signal_data)
                else:
                    # Se non ci sono dati sufficienti a quella data (es. titolo non ancora quotato)
                    pass  # Ignora silenziosamente o logga

            progress_bar.progress((i + 1) / len(tickers))

        status_text.empty()
        progress_bar.empty()

        # --- VISUALIZZAZIONE RISULTATI ---
        date_label = analysis_date.strftime('%d/%m/%Y')
        if enable_time_travel:
            st.subheader(f"📅 Risultati Storici al {date_label}")
        else:
            st.subheader(f"📅 Segnali Live ({date_label})")

        if not results:
            st.info(f"Nessun segnale operativo trovato alla data {date_label}.")
            if enable_time_travel:
                st.caption(
                    "Suggerimento: Prova a cambiare data. I segnali di breakout sono rari e durano pochi giorni.")
            return

        st.success(f"Trovate {len(results)} opportunità operative!")

        # Ordina per "Confidenza"
        results.sort(key=lambda x: x['confidence'], reverse=True)

        for res in results:
            # Colore dinamico in base al tipo di segnale
            signal_color = "green" if "LONG" in res['signal'] else "red"

            with st.container():
                # Intestazione con Ticker e Segnale
                st.markdown(f"### {res['ticker']} : :{signal_color}[{res['signal']}]")
                st.caption(f"Prezzo alla data {date_label}: **{res['price']:.2f}** | Confidence: {res['confidence']}/5")

                # Metriche Chiave
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prezzo Ingresso", f"{res['price']:.2f}")

                # Stop Loss e Take Profit
                sl_delta = res['stop_loss'] - res['price']
                tp_delta = res['take_profit'] - res['price']

                c2.metric("Stop Loss (ATR)", f"{res['stop_loss']:.2f}", delta=f"{sl_delta:.2f}", delta_color="inverse")
                c3.metric("Take Profit", f"{res['take_profit']:.2f}", delta=f"{tp_delta:.2f}", delta_color="normal")

                # Indicatore RSI
                rsi_val = res['indicators']['RSI']
                rsi_state = "Ipercomprato" if rsi_val > 70 else "Ipervenduto" if rsi_val < 30 else "Neutrale"
                c4.metric("RSI (14)", f"{rsi_val:.1f}", delta=rsi_state, delta_color="off")

                # Spiegazione
                with st.expander("Dettagli Strategici"):
                    st.write("**Motivazioni del Segnale:**")
                    for reason in res['reasons']:
                        st.markdown(f"- {reason}")

                    st.markdown("---")
                    st.markdown(f"**Dati Tecnici al {date_label}:**")
                    st.markdown(f"- *Bande Bollinger Width:* {res['indicators']['BB_Width']:.4f} (Compressione)")
                    st.markdown(f"- *Volume Relativo:* {res['indicators']['Vol_Rel']:.1f}x media")

                st.markdown("---")

    else:
        if enable_time_travel:
            st.info(f"👈 Imposta la data storica ({analysis_date.strftime('%d/%m/%Y')}) e premi 'Scansiona'.")
        else:
            st.info("👈 Inserisci i ticker e premi 'Scansiona Mercato' per l'analisi live.")