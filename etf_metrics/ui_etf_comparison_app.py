# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re

# Importazioni dal progetto
from .yahoo_client import resolve_isin_one, get_series
from .utils import to_percent_index

# Plotly per grafici interattivi
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Mappa dei periodi per i controlli della UI
PERIOD_MAP_LABEL_TO_YF = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd",
    "1A": "1y", "3A": "3y", "5A": "5y", "Max": "max"
}

def render_etf_comparison_ui():
    """Renderizza la UI per la comparazione di più ETF."""
    st.title("🔍 Confronto Multi-ETF")
    st.caption("Visualizza e confronta le performance di più ETF o azioni su diversi orizzonti temporali.")

    # --- Controlli nella Sidebar ---
    st.sidebar.header("⚙️ Impostazioni Confronto")
    compare_input = st.sidebar.text_area(
        "ISIN o Ticker da confrontare",
        "IE00B4L5Y983, SPY, QQQ",
        height=150,
        help="Inserisci ISIN o Ticker separati da virgola o uno per riga."
    )

    # --- Logica di Risoluzione e Fetching ---
    inputs = [item.strip().upper() for item in re.split(r'[,\n]', compare_input) if item.strip()]
    tickers_to_compare = []

    with st.spinner("Risoluzione ISIN/Ticker in corso..."):
        for item in inputs:
            # Controlla se è un ISIN, altrimenti lo tratta come un ticker
            if re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', item):
                ticker = resolve_isin_one(item)
                if ticker:
                    tickers_to_compare.append(ticker)
                else:
                    st.sidebar.warning(f"ISIN {item} non trovato, verrà saltato.")
            else:
                tickers_to_compare.append(item)

    tickers = list(set(tickers_to_compare))  # Rimuove duplicati

    if not tickers:
        st.info("Aggiungi ISIN o Ticker validi nella sidebar per avviare il confronto.")
        return

    # --- Controlli e Grafico nella Pagina Principale ---
    st.subheader("Grafico di Performance Normalizzato")
    selected_period_label = st.radio(
        "Seleziona periodo di analisi",
        list(PERIOD_MAP_LABEL_TO_YF.keys()),
        index=4,  # Default su "1A"
        horizontal=True,
        key="comparison_period_radio"
    )
    period_yf = PERIOD_MAP_LABEL_TO_YF[selected_period_label]

    # Scarica i dati per tutti i ticker selezionati
    data = {ticker: get_series(ticker, period_yf) for ticker in tickers}

    # Logica per la visualizzazione del grafico
    if HAS_PLOTLY:
        fig = go.Figure()
        for ticker, series in data.items():
            if series is not None and not series.empty:
                pct_series = to_percent_index(series)
                fig.add_trace(go.Scatter(x=pct_series.index, y=pct_series.values, mode='lines', name=ticker))
        fig.update_layout(
            title=f"Andamento Confrontato degli Asset ({selected_period_label})",
            yaxis_title="Performance (%)",
            xaxis_title="Data",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="comparison_chart")
    else:
        # Fallback a grafico statico se Plotly non è disponibile
        chart_data = pd.DataFrame({
            ticker: to_percent_index(series)
            for ticker, series in data.items()
            if series is not None and not series.empty
        })
        if not chart_data.empty:
            st.line_chart(chart_data)
        else:
            st.warning("Nessun dato valido da visualizzare per il periodo selezionato.")