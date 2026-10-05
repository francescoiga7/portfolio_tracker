# -*- coding: utf-8 -*-
import streamlit as st

from etf_metrics.ui.portfolio_tracker import render_portfolio_tracker_ui
from etf_metrics.ui.portfolio_backtester import render_portfolio_backtester_ui
from etf_metrics.ui.pac_screener import render_pac_screener_ui
from etf_metrics.ui.trading import render_trading_ui
from etf_metrics.ui.data_hub import render_data_hub_ui
from etf_metrics.ui.algo_live import render_algo_live_ui


def init_session_state():
    """Inizializza lo stato della sessione per l'applicazione."""
    # Le chiavi di sessione vive sono gestite direttamente dai moduli UI.
    pass


def run_app():
    """Avvia l'applicazione Streamlit."""
    init_session_state()
    st.set_page_config(page_title="ETF & Portfolio Analysis", page_icon="📊", layout="wide")

    st.sidebar.title("Strumenti di Analisi 📈")
    app_mode = st.sidebar.selectbox(
        "Scegli modalità",
        [
            "💼 Portfolio Tracker",
            "⏳ Portfolio Backtester",
            "🎯 Screener PAC",
            "💹 Trading",
            "📥 Gestione Dati",
            "💲 Portafoglio Live",
        ],
    )

    if app_mode == "💼 Portfolio Tracker":
        render_portfolio_tracker_ui()
    elif app_mode == "⏳ Portfolio Backtester":
        render_portfolio_backtester_ui()
    elif app_mode == "🎯 Screener PAC":
        render_pac_screener_ui()
    elif app_mode == "💹 Trading":
        render_trading_ui()
    elif app_mode == "📥 Gestione Dati":
        render_data_hub_ui()
    elif app_mode == "💲 Portafoglio Live":
        render_algo_live_ui()


if __name__ == '__main__':
    run_app()
