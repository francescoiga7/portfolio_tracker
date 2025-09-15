# -*- coding: utf-8 -*-
from typing import Tuple
import pandas as pd
import streamlit as st

from .etf_search_engine import find_top_performers


def format_results_for_display(
        df: pd.DataFrame,
        pct_cols: Tuple[str, ...] = ("cagr_pct", "mdd_pct"),
        round_cols: Tuple[str, ...] = ("sharpe_ratio",),
        pct_decimals: int = 2,
        num_decimals: int = 2,
) -> pd.DataFrame:
    """Formatta il DataFrame per una visualizzazione pulita nella UI."""
    if df is None or df.empty:
        return df
    df_fmt = df.copy()
    # Rinomina le colonne per la visualizzazione
    df_fmt = df_fmt.rename(columns={
        "isin": "ISIN", "ticker": "Ticker", "name": "Nome",
        "cagr_pct": "CAGR %", "mdd_pct": "Max DD %", "sharpe_ratio": "Sharpe Ratio"
    })

    for col in pct_cols:
        col_display = col.replace("_pct", " %").replace("_", " ").title()
        if col_display in df_fmt.columns:
            df_fmt[col_display] = df_fmt[col_display].apply(
                lambda x: f"{x:.{pct_decimals}f}%" if pd.notna(x) else "n.d."
            )

    for col in round_cols:
        col_display = col.replace("_", " ").title()
        if col_display in df_fmt.columns:
            df_fmt[col_display] = df_fmt[col_display].apply(
                lambda x: f"{x:.{num_decimals}f}" if pd.notna(x) else "n.d."
            )

    return df_fmt


@st.cache_data(show_spinner=True, ttl=60 * 60)  # Cache per 1 ora
def _cached_find_top_performers(
        period: str,
        risk_free_pct: float,
        quotes_per_query: int,
        discovery_limit: int,
):
    """Wrapper con cache per la funzione di ricerca principale."""
    return find_top_performers(
        period=period,
        risk_free_rate_pct=risk_free_pct,
        quotes_per_query=quotes_per_query,
        discovery_limit=discovery_limit,
    )


def render_etf_search_ui():
    """Renderizza la UI di Streamlit per il nuovo ETF Search Engine."""
    st.title("🔎 ETF Search Engine (UCITS • Mercato Italiano)")
    st.caption("Scopri i migliori ETF ed ETC per Sharpe Ratio, con priorità Borsa Italiana.")

    st.sidebar.header("⚙️ Parametri di Ricerca")

    # Selettore del periodo aggiornato
    period = st.sidebar.selectbox(
        "Periodo storico di analisi",
        options=["1d", "1m", "3m", "6m", "1y", "3y", "5y"],
        index=4  # Default su "1y"
    )

    risk_free_pct = st.sidebar.number_input(
        "Tasso risk-free annuo (%)",
        value=3.50, min_value=-5.0, max_value=20.0, step=0.10, format="%.2f"
    )

    # Parametri per la fase di scoperta
    st.sidebar.subheader("Impostazioni di Scoperta")
    quotes_per_query = st.sidebar.slider(
        "Risultati per query di ricerca",
        min_value=50, max_value=200, value=150, step=10,
        help="Numero di risultati che Yahoo Finance restituisce per ogni query di ricerca. Un valore più alto aumenta l'universo di ETF scoperti."
    )
    discovery_limit = st.sidebar.slider(
        "Limite universo ETF da scoprire",
        min_value=200, max_value=2500, value=1500, step=100,  # Limite default aumentato
        help="Numero massimo di ETF totali da scoprire prima di passare alla fase di analisi. Limita il tempo di esecuzione."
    )

    if st.sidebar.button("▶️ Esegui Ricerca"):
        st.subheader(f"🏆 Classifica ETF/ETC per Sharpe Ratio (Periodo: {period})")

        with st.spinner("Ricerca in corso... Questa operazione potrebbe richiedere fino a un minuto."):
            df = _cached_find_top_performers(
                period=period,
                risk_free_pct=risk_free_pct,
                quotes_per_query=quotes_per_query,
                discovery_limit=discovery_limit,
            )

        if df.empty:
            st.warning(
                "Nessun risultato trovato. Prova ad aumentare i limiti di scoperta o a controllare la connessione.")
        else:
            st.dataframe(format_results_for_display(df), use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Scarica risultati in CSV",
                data=csv,
                file_name=f'classifica_etf_{period}.csv',
                mime='text/csv',
            )
    else:
        st.info("Imposta i parametri nella barra laterale e premi 'Esegui Ricerca' per iniziare.")