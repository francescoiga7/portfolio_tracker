# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

# Import dei renderer
from .ui_single_etf import render_single_etf_ui
from .ui_momentum_app import render_momentum_ui
from .ui_portfolio_tracker_app import render_portfolio_tracker_ui
from .ui_portfolio_backtester_app import render_portfolio_backtester_ui
from .ui_etf_comparison_app import render_etf_comparison_ui
from .ui_pac_screener_app import render_pac_screener_ui
import subprocess
subprocess.run(["uv", "pip", "install", "-r", "requirements.txt", "--upgrade"])

def init_session_state():
    defaults = {
        "data_out": None,
        "portfolio_results": None,
        "momentum_results": None,
        "pac_screener_results": pd.DataFrame(),
        "lab_momentum_raw_data": None,
        "core_satellite_results": None, # NUOVO STATO
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def run_app():
    init_session_state()
    st.set_page_config(page_title="ETF & Portfolio Analysis", page_icon="📊", layout="wide")

    st.sidebar.title("Strumenti di Analisi 📈")
    app_mode = st.sidebar.selectbox(
        "Scegli modalità",
        [
            "Analisi Singolo ETF",
            "Confronta ETF",
            "Portfolio Tracker",
            "Portfolio Backtester",
            "Screener Tattico PAC",
            "Verifica Momentum"
        ],
    )

    if app_mode == "Portfolio Tracker":
        render_portfolio_tracker_ui()
    elif app_mode == "Portfolio Backtester":
        render_portfolio_backtester_ui()
    elif app_mode == "Analisi Singolo ETF":
        render_single_etf_ui()
    elif app_mode == "Confronta ETF":
        render_etf_comparison_ui()
    elif app_mode == "Screener Tattico PAC":
        render_pac_screener_ui()
    elif app_mode == "Verifica Momentum":
        render_momentum_ui()

if __name__ == '__main__':
    run_app()