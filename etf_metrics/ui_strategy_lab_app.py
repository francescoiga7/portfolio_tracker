# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import date

# Importa le logiche di calcolo esistenti
from .momentum import process_momentum_rankings, fetch_momentum_data
from .pac_screener import screen_for_tactical_etfs, get_market_regime


def render_strategy_lab_ui():
    """Renderizza la UI per il backtesting point-in-time delle strategie."""
    st.title("🔬 Laboratorio Strategie (Time Travel)")
    st.caption("Verifica le performance passate delle strategie spostandoti a una data specifica.")

    st.sidebar.header("⚙️ Impostazioni Time Travel")

    as_of_date = st.sidebar.date_input(
        "Posizionati alla data del:",
        date(2025, 1, 1),
        min_value=date(2020, 1, 1),
        max_value=date.today()
    )

    strategy_to_test = st.sidebar.selectbox(
        "Scegli la strategia da testare:",
        ["Screener Tattico", "Verifica Momentum"]
    )

    st.info(f"Stai simulando l'analisi come se fossi al **{as_of_date.strftime('%d/%m/%Y')}**.")

    if strategy_to_test == "Screener Tattico":
        render_tactical_screener_lab(as_of_date)
    else:
        render_momentum_lab(as_of_date)


def render_tactical_screener_lab(as_of_date: date):
    """UI specifica per testare lo Screener Tattico."""
    st.header(f"Test: Screener Tattico al {as_of_date.strftime('%d/%m/%Y')}")

    discovery_limit = st.sidebar.slider("Universo ETF", 200, 1000, 500, 50, key="lab_discovery")
    min_avg_value = st.sidebar.number_input("Volume minimo scambiato (€)", 0, 1000000, 50000, 50000, key="lab_volume")

    market_info = get_market_regime(as_of_date=as_of_date)
    vix = market_info.get("vix")
    regime = market_info.get("regime")

    if regime == "Favorevole al Rischio":
        st.sidebar.success(f"Regime di Mercato: Favorevole (VIX: {vix:.2f})")
        if st.sidebar.button("▶️ Avvia Analisi nel Passato"):
            with st.spinner("Esecuzione dello screener nel passato..."):
                results = screen_for_tactical_etfs(
                    discovery_limit=discovery_limit,
                    min_avg_value=min_avg_value,
                    as_of_date=as_of_date
                )
                st.session_state.lab_screener_results = results
    else:
        vix_str = f"{vix:.2f}" if vix is not None else "N/D"
        st.sidebar.error(f"Regime di Mercato: {regime} (VIX: {vix_str})")
        st.sidebar.warning(
            "In questa data il mercato era in una fase di alta volatilità. Lo screener non avrebbe operato.")

    if 'lab_screener_results' in st.session_state:
        results = st.session_state.lab_screener_results
        if results is not None and not results.empty:
            st.subheader("Risultati dello Screener")
            st.dataframe(results)
        else:
            st.warning("Nessun ETF ha soddisfatto i criteri alla data selezionata.")


def render_momentum_lab(as_of_date: date):
    """UI specifica per testare la strategia Momentum."""
    st.header(f"Test: Verifica Momentum al {as_of_date.strftime('%d/%m/%Y')}")

    momentum_isins = st.text_area(
        "ISIN da analizzare (uno per riga)",
        "IE00B4L5Y983\nIE00B5BMR087\nIE00BK5BQT80",
        height=120,
        key="lab_momentum_isins"
    )

    if st.button("Carica Dati Storici e Analizza"):
        isins_list = [isin.strip().upper() for isin in momentum_isins.split('\n') if isin.strip()]
        if isins_list:
            raw_data = fetch_momentum_data(isins_list)
            st.session_state.lab_momentum_raw_data = raw_data
        else:
            st.warning("Inserisci almeno un ISIN.")

    if 'lab_momentum_raw_data' in st.session_state and st.session_state.lab_momentum_raw_data:
        lookback = st.slider("Periodo di lookback (mesi)", 1, 24, 6, 1, key="lab_lookback")
        rf_ann = st.number_input("Risk-free annuo (%)", -5.0, 10.0, 3.5, 0.25, key="lab_rf")

        results = process_momentum_rankings(
            st.session_state.lab_momentum_raw_data,
            lookback,
            rf_ann,
            end_date_override=pd.to_datetime(as_of_date)
        )

        if results:
            df = pd.DataFrame(results).rename(columns={
                "isin": "ISIN", "ticker": "Ticker",
                "sharpe_ratio": "Sharpe Ratio", "signal": "Segnale Operativo"
            })
            st.dataframe(df[["ISIN", "Ticker", "Sharpe Ratio", "Segnale Operativo"]], width="stretch")
        else:
            st.warning(
                f"Nessun ETF con dati sufficienti per il lookback di {lookback} mesi alla data del {as_of_date.strftime('%d/%m/%Y')}.")