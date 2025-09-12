# -*- coding: utf-8 -*-
import streamlit as st

from .etf_search_engine import search_etfs_universal, format_results_for_display


def render_etf_search_ui():
    """UI Streamlit per ricerca universale ETF UCITS (default periodo 1Y)."""
    st.title("🔎 ETF Search Engine (Universale • UCITS)")

    st.sidebar.header("⚙️ Parametri di ricerca")
    period = st.sidebar.selectbox(
        "Periodo storico",
        options=["1m", "3m", "6m", "1y", "3y", "5y", "10y", "ytd", "max"],
        index=3  # "1y"
    )
    risk_free_pct = st.sidebar.number_input(
        "Tasso risk-free annuo (%)",
        value=3.95, min_value=-5.0, max_value=20.0, step=0.10, format="%.2f"
    )
    quotes_per_query = st.sidebar.slider("Risultati per query seed", min_value=20, max_value=200, value=100, step=10)
    limit_universe = st.sidebar.slider("Limite universo (discovery)", min_value=100, max_value=2000, value=600, step=50)
    max_results = st.sidebar.slider("ETF da valutare (post discovery)", min_value=10, max_value=500, value=150, step=10)

    sort_by = st.sidebar.selectbox("Ordina per", options=["sharpe ratio", "rendimento cagr", "max drawdown", "ticker"], index=0)
    ascending = st.sidebar.checkbox("Ordine crescente", value=False)

    if st.sidebar.button("Esegui ricerca universale"):
        st.subheader("Risultati ricerca ETF (UCITS)")
        with st.spinner("Scopro l'universo ETF UCITS e calcolo le metriche..."):
            df = search_etfs_universal(
                period=period,
                risk_free_rate_pct=risk_free_pct,  # in %
                quotes_per_query=quotes_per_query,
                limit_universe=limit_universe,
                max_results=max_results,
                sort_by=sort_by,
                ascending=ascending,
            )
        if df.empty:
            st.warning("Nessun risultato utile o dati insufficienti per calcolare le metriche.")
        else:
            st.dataframe(format_results_for_display(df), use_container_width=True)
