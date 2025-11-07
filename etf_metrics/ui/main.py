# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from etf_metrics.ui.single_etf import render_single_etf_ui
from etf_metrics.ui.momentum import render_momentum_ui
from etf_metrics.ui.portfolio_tracker import render_portfolio_tracker_ui
from etf_metrics.ui.portfolio_backtester import render_portfolio_backtester_ui
from .comparison import render_etf_comparison_ui
from etf_metrics.ui.pac_screener import render_pac_screener_ui


def init_session_state():
    """Inizializza lo stato della sessione per l'applicazione."""
    defaults = {
        "data_out": None,
        "portfolio_results": None,
        "momentum_results": None,
        "pac_screener_results": pd.DataFrame(),
        "lab_momentum_raw_data": None,
        "core_satellite_results": None,
        "tactical_signals": [],
        "simulated_trades": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def run_app():
    """Avvia l'applicazione Streamlit."""
    init_session_state()
    st.set_page_config(page_title="ETF & Portfolio Analysis", page_icon="📊", layout="wide")

    st.sidebar.title("Strumenti di Analisi 📈")
    app_mode = st.sidebar.selectbox(
        "Scegli modalità",
        [
            "🔎 Analisi ETF",
            "⚖️ Confronta ETF",
            "💼 Portfolio Tracker",
            "⏳ Portfolio Backtester",
            "🎯 Screener PAC",
            "⚡ Verifica Momentum",
            "💹 Trading"
        ],
    )

    if app_mode == "🔎 Analisi ETF":
        render_single_etf_ui()
    elif app_mode == "⚖️ Confronta ETF":
        render_etf_comparison_ui()
    elif app_mode == "💼 Portfolio Tracker":
        render_portfolio_tracker_ui()
    elif app_mode == "⏳ Portfolio Backtester":
        render_portfolio_backtester_ui()
    elif app_mode == "🎯 Screener PAC":
        render_pac_screener_ui()
    elif app_mode == "⚡ Verifica Momentum":
        render_momentum_ui()

if __name__ == '__main__':
    run_app()