# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

# Import dei renderer per ciascun app_mode
from .ui_single_etf import render_single_etf_ui
from .ui_momentum_app import render_momentum_ui
from .ui_portfolio_tracker_app import render_portfolio_tracker_ui
from .ui_portfolio_backtester_app import render_portfolio_backtester_ui
from .ui_etf_comparison_app import render_etf_comparison_ui
from .ui_pac_screener_app import render_pac_screener_ui


def init_session_state():
    defaults = {
        "data_out": None,
        "portfolio_results": None,
        "momentum_results": None,
        # *** CORREZIONE: Inizializza come DataFrame vuoto ***
        "pac_screener_results": pd.DataFrame(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_app():
    init_session_state()
    st.set_page_config(page_title="ETF & Portfolio Analysis", page_icon="📊", layout="wide")

    # --- SIDEBAR ---
    st.sidebar.title("Strumenti di Analisi 📈")
    app_mode = st.sidebar.selectbox(
        "Scegli modalità",
        ["Portfolio Tracker", "Portfolio Backtester", "Analisi Singolo ETF", "Confronta ETF", "Verifica Momentum", "Screener PAC"],
    )

    if app_mode == "Portfolio Tracker":
        render_portfolio_tracker_ui()
    elif app_mode == "Portfolio Backtester":
        render_portfolio_backtester_ui()
    elif app_mode == "Analisi Singolo ETF":
        render_single_etf_ui()
    elif app_mode == "Confronta ETF":
        render_etf_comparison_ui()
    elif app_mode == "Verifica Momentum":
        render_momentum_ui()
    elif app_mode == "Screener PAC":
        render_pac_screener_ui()


# --- ENTRY POINT ---
if __name__ == '__main__':
    run_app()