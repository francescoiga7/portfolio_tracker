# -*- coding: utf-8 -*-
import streamlit as st
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.trading import analyze_ticker_alpha

# Titoli di default ad alta volatilità per trading breve termine
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

    with st.expander("🧠 Come funziona"):
        st.markdown("""
        Questo algoritmo non cerca investimenti sicuri a 10 anni. Cerca **anomalie statistiche** a breve termine.

        1.  **Bollinger Breakout:** Cerca prezzi che escono violentemente dalle bande statistiche standard.
        2.  **Volume Confirmation:** Ignora i movimenti senza "benzina" (volumi bassi).
        3.  **Smart Risk:** Calcola automaticamente Stop Loss basati sulla volatilità (ATR), non percentuali fisse a caso.

        **Legenda Segnali:**
        * 🟢 **LONG_BREAKOUT:** Il prezzo sta esplodendo al rialzo. Momentum forte.
        * 🔵 **LONG_DIP:** Il prezzo è in trend ma ha stornato. Occasione di ingresso a sconto.
        * 🔴 **SHORT_BREAKDOWN:** Il prezzo sta crollando. Opportunità di vendita allo scoperto.
        """)

    st.sidebar.header("⚙️ Radar Settings")
    tickers_input = st.sidebar.text_area("Watchlist (Ticker Yahoo)", DEFAULT_TRADING_LIST, height=200)

    if st.sidebar.button("🔥 Scansiona Mercato"):
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(tickers):
            status_text.text(f"Analisi {ticker}...")
            # Scarichiamo dati giornalieri, 1 anno di storico basta per gli indicatori
            df = get_series(ticker, period="1y", as_dataframe=True)

            if df is not None and not df.empty:
                signal_data = analyze_ticker_alpha(ticker, df)
                if signal_data and signal_data['signal'] != "NEUTRAL":
                    results.append(signal_data)

            progress_bar.progress((i + 1) / len(tickers))

        status_text.empty()
        progress_bar.empty()

        if not results:
            st.warning(
                "Nessun segnale operativo trovato al momento. Il mercato è laterale o i criteri sono troppo stretti.")
            return

        # Visualizzazione Risultati
        st.success(f"Trovate {len(results)} opportunità operative!")

        # Ordina per "Confidenza"
        results.sort(key=lambda x: x['confidence'], reverse=True)

        for res in results:
            signal_color = "green" if "LONG" in res['signal'] else "red"
            with st.container():
                st.markdown(
                    f"### {res['ticker']} : :{signal_color}[{res['signal']}] (Confidence: {res['confidence']}/5)")

                cols = st.columns(4)
                cols[0].metric("Prezzo Attuale", f"{res['price']:.2f}")
                cols[1].metric("Stop Loss (ATR)", f"{res['stop_loss']:.2f}", delta_color="inverse")
                cols[2].metric("Take Profit", f"{res['take_profit']:.2f}")

                rsi_val = res['indicators']['RSI']
                cols[3].metric("RSI", f"{rsi_val:.1f}", delta=f"{rsi_val - 50:.1f}")

                st.write("**Motivazioni Strategiche:**")
                for reason in res['reasons']:
                    st.markdown(f"- {reason}")

                st.markdown("---")

    else:
        st.info("👈 Inserisci i ticker nella sidebar e premi 'Scansiona Mercato' per attivare l'algoritmo.")