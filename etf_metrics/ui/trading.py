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
    st.title("💹 Trading Reale & Tasse")
    st.caption("Simulazione professionale con impatto Fiscale (26%) e Commissionale (1€).")

    with st.expander("🧠 Come ottimizzare per 1.000€ vs 10.000€"):
        st.markdown("""
        **La sfida dei piccoli capitali (1.000€):**
        - Ogni trade costa **2€** (1€ acquisto + 1€ vendita).
        - Su un trade da 200€, il 2€ è l'**1%** di perdita secca immediata.
        - **Strategia:** Devi fare **MENO trade** e più concentrati (All-In su 1-2 titoli) per diluire i costi fissi.

        **La gestione dei grandi capitali (10.000€+):**
        - Il costo commissionale è irrisorio (0.02% su 10k).
        - **Strategia:** Puoi permetterti di **DIVERSIFICARE** su 4-5 titoli per ridurre il rischio specifico senza preoccuparti delle commissioni.

        **Nota Fiscale:** Il backtest calcola il 26% di tasse su ogni profitto netto e usa lo zainetto fiscale per compensare le perdite.
        """)

    st.sidebar.header("⚙️ Configurazione")

    initial_capital = st.sidebar.number_input(
        "💰 Capitale Iniziale (€)",
        min_value=500, value=5000, step=500
    )

    enable_time_travel = st.sidebar.checkbox("Backtest Storico (Time Travel)")
    analysis_date = date.today()
    if enable_time_travel:
        analysis_date = st.sidebar.date_input("Data Analisi:", date.today())
        st.sidebar.warning(f"Data congelata: {analysis_date}")

    st.sidebar.subheader("📋 Watchlist")
    tickers_input = st.sidebar.text_area("Ticker", DEFAULT_TRADING_LIST, height=150)

    if st.sidebar.button("🔥 Scansiona Mercato"):
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
        results = []
        cutoff = pd.to_datetime(analysis_date)

        with st.spinner("Analisi in corso..."):
            for t in tickers:
                df = get_series(t, "10y", True)
                if df is not None:
                    df_slice = df[df.index <= cutoff].copy()
                    if len(df_slice) > 200:
                        res = analyze_ticker(t, df_slice)
                        if res and res['signal'] != "NEUTRAL": results.append(res)

        if not results:
            st.info("Nessun segnale trovato.")
        else:
            st.success(f"Trovati {len(results)} segnali!")
            results.sort(key=lambda x: x['confidence'], reverse=True)

            for res in results:
                with st.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(res['ticker'], f"{res['price']:.2f}", res['signal'])

                    costo_fisso_incidenza = (2.0 / (initial_capital * 0.2)) * 100

                    rec_size = "Concentrata (1-2 asset)" if initial_capital < 2500 else "Diversificata (4-5 asset)"
                    c2.metric("Strategia Consigliata", rec_size,
                              help=f"Incidenza commissioni stimate: {costo_fisso_incidenza:.2f}% per trade")
                    c3.metric("Stop Loss", f"{res['stop_loss']:.2f}")
                    c4.metric("Confidence", f"{res['confidence']}/100")
                    st.markdown("---")

    else:
        st.markdown("---")
        st.subheader("🤖 Backtest Fiscale (Commissioni + Tax 26%)")

        col_b1, col_b2 = st.columns(2)
        commission = col_b1.number_input("Commissione Fissa (€)", 0.0, 10.0, 1.0, step=0.5)
        tax_rate = col_b2.number_input("Aliquota Tasse (%)", 0.0, 50.0, 26.0, step=1.0)

        if st.button("Avvia Backtest Reale"):
            t_list = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
            start_dt = pd.to_datetime(analysis_date)

            with st.spinner("Simulazione fiscale in corso..."):
                df_trades, final = run_market_aware_backtest(
                    t_list, start_date=start_dt,
                    initial_capital=initial_capital,
                    commission=commission, tax_rate=tax_rate
                )

            if not df_trades.empty:
                net_profit = final - initial_capital
                ret_pct = (net_profit / initial_capital) * 100

                tot_comm = df_trades['Comm'].sum()
                tot_tax = df_trades['Tax'].sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Capitale Netto", f"€{final:,.2f}", delta=f"{ret_pct:.2f}%")
                m2.metric("Profitto Netto", f"€{net_profit:,.2f}")
                m3.metric("Tasse Pagate (26%)", f"€{tot_tax:,.2f}", delta_color="inverse")
                m4.metric("Commissioni Totali", f"€{tot_comm:,.2f}", delta_color="inverse")

                st.dataframe(
                    df_trades.style.format({
                        "Price": "{:.2f}", "PnL_Net": "{:+.2f}", "Tax": "{:.2f}",
                        "Comm": "{:.2f}", "Capital": "€{:,.0f}"
                    }),
                    width="stretch"
                )
            else:
                st.warning("Nessun trade eseguito.")