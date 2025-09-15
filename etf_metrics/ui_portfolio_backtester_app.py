# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from .portfolio_backtester import parse_portfolio_input, get_all_portfolios_for_backtest
from .config import FAMOUS_PORTFOLIOS
from .ui_portfolio_tracker_app import _display_backtest_results


def render_portfolio_backtester_ui():
    st.title("🔙 Portfolio Backtester")
    st.caption("Simula e confronta le performance di portafogli personalizzati nel tempo.")

    st.sidebar.header("⚙️ Impostazioni Backtest")

    portfolio_text = st.sidebar.text_area(
        "Definizione Portafoglio (ISIN: Peso%)",
        "IE00BK5BQT80: 80\nIE00BDBRDM35: 20",
        height=150,
        help="Inserisci un ISIN o Ticker per riga, seguito da ':' e dal peso percentuale (es. 'VWCE.MI: 80'). La somma dei pesi verrà normalizzata a 100."
    )

    initial_investment = st.sidebar.number_input("Investimento Iniziale (€)", min_value=100, max_value=1000000,
                                                 value=10000, step=100)
    strategy = st.sidebar.selectbox("Strategia", ["Lump Sum (PIC)", "PAC"])

    monthly_investment = 0
    if strategy == "PAC":
        monthly_investment = st.sidebar.number_input("Investimento mensile (€)", min_value=50, max_value=10000,
                                                     value=500, step=50)

    rebalancing = st.sidebar.selectbox("Frequenza di Ribilanciamento", ["Mai", "Annuale"])

    try:
        famous_all = list(FAMOUS_PORTFOLIOS.keys())
    except Exception:
        famous_all = []

    famous_selection = st.sidebar.multiselect(
        "Confronta con portafogli modello",
        famous_all,
        default=(["Classic 60/40"] if "Classic 60/40" in famous_all else [])
    )

    rf_ann_backtest = st.sidebar.number_input("Risk-free annuo (%) per Sharpe", min_value=-5.0, max_value=10.0,
                                              value=1.0, step=0.25)

    if st.sidebar.button("▶️ Esegui Backtest"):
        user_portfolio_def = parse_portfolio_input(portfolio_text)
        if not user_portfolio_def:
            st.error("La definizione del portafoglio non è valida. Controlla il formato.")
        else:
            with st.spinner("Esecuzione backtest in corso..."):
                all_series = get_all_portfolios_for_backtest(
                    user_portfolio_def,
                    famous_selection,
                    FAMOUS_PORTFOLIOS if famous_all else {},
                    initial_investment=initial_investment,
                    strategy=strategy.lower().replace(" ", "_"),
                    monthly_investment=monthly_investment,
                    rebalancing=rebalancing.lower()
                )
                st.session_state.backtest_results = {'series': all_series, 'rf_ann': rf_ann_backtest}
            st.success("Backtest completato!")

    # Visualizzazione risultati
    if st.session_state.get('backtest_results') and st.session_state.backtest_results.get('series'):
        st.subheader("📊 Risultati del Backtest")
        _display_backtest_results(
            st.session_state.backtest_results['series'],
            st.session_state.backtest_results['rf_ann'],
            key_prefix="main_bt"
        )
    else:
        st.info("Definisci un portafoglio e i parametri nella sidebar, poi premi 'Esegui Backtest'.")