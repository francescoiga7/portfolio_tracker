# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re

from etf_metrics.clients.yahoo_client import resolve_isin_one, get_series
from etf_metrics.shared.utils import to_percent_index

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

PERIOD_MAP_LABEL_TO_YF = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd",
    "1A": "1y", "3A": "3y", "5A": "5y", "Max": "max"
}


def render_etf_comparison_ui():
    """Renderizza la UI per la comparazione di più ETF."""
    st.title("⚖️ Confronto Multi-ETF")
    st.caption("Visualizza e confronta le performance di più ETF o azioni su diversi orizzonti temporali.")

    with st.expander("📖 Metodologia e Funzionamento"):
        st.markdown("""
        Questa sezione ti permette di confrontare le performance di più ETF o azioni. Puoi:
        - **Inserire un elenco di ISIN o Ticker** da confrontare.
        - **Visualizzare un grafico interattivo** con le performance normalizzate degli asset.
        - **Scegliere il periodo di analisi** per il confronto.
        """)

    st.sidebar.header("⚙️ Impostazioni Confronto")
    compare_input = st.sidebar.text_area(
        "ISIN o Ticker da confrontare",
        "IE00B4L5Y983, SPY, QQQ",
        height=150,
        help="Inserisci ISIN o Ticker separati da virgola o uno per riga."
    )

    inputs = [item.strip().upper() for item in re.split(r'[,\n]', compare_input) if item.strip()]
    tickers_to_compare = []

    with st.spinner("Risoluzione ISIN/Ticker in corso..."):
        for item in inputs:
            # Controllo semplificato per ISIN (2 lettere, 9 alfanumerici, 1 cifra)
            if re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', item):
                ticker = resolve_isin_one(item)
                if ticker:
                    tickers_to_compare.append(ticker)
                else:
                    st.sidebar.warning(f"ISIN {item} non trovato, verrà saltato.")
            else:
                # Tratta il resto come Ticker (es. SPY, QQQ)
                tickers_to_compare.append(item)

    # Rimuovi i duplicati
    tickers = list(set(tickers_to_compare))

    if not tickers:
        st.info("Aggiungi ISIN o Ticker validi nella sidebar per avviare il confronto.")
        return

    st.subheader("Grafico di Performance Normalizzato")
    selected_period_label = st.radio(
        "Seleziona periodo di analisi",
        list(PERIOD_MAP_LABEL_TO_YF.keys()),
        index=4,
        horizontal=True,
        key="comparison_period_radio"
    )
    period_yf = PERIOD_MAP_LABEL_TO_YF[selected_period_label]

    # Recupera i dati
    data = {ticker: get_series(ticker, period_yf) for ticker in tickers}

    if HAS_PLOTLY:
        fig = go.Figure()

        # Aggiungi tracce al grafico
        for ticker, series in data.items():
            if series is not None and not series.empty:
                pct_series = to_percent_index(series)
                fig.add_trace(go.Scatter(x=pct_series.index, y=pct_series.values, mode='lines', name=ticker))

        # Aggiorna il layout del grafico
        fig.update_layout(
            title=f"Andamento Confrontato degli Asset ({selected_period_label})",
            yaxis_title="Performance (%)",
            xaxis_title="Data",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True, key="comparison_chart")
    else:
        # Codice di fallback per st.line_chart
        chart_data = pd.DataFrame({
            ticker: to_percent_index(series)
            for ticker, series in data.items()
            if series is not None and not series.empty
        })
        if not chart_data.empty:
            st.line_chart(chart_data)
        else:
            st.warning("Nessun dato valido da visualizzare per il periodo selezionato.")