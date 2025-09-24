# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from .config import PERIODS_ALL
from .pipeline import compute_etf_over_periods
from .utils import to_percent_index
from .yahoo_client import get_series
from .metrics import get_trading_signal

PERIOD_MAP_LABEL_TO_YF = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd",
    "1A": "1y", "3A": "3y", "5A": "5y", "Max": "max"
}


def render_single_etf_ui():
    st.sidebar.header("Impostazioni Analisi")
    isin = st.sidebar.text_input("ISIN", value="IE00BK5BQT80").strip().upper()
    bench_override = st.sidebar.text_input("Override Benchmark (Ticker Yahoo)", "").strip() or None
    rf_ann = st.sidebar.number_input("Risk-free annuo (%)", -5.0, 10.0, 3.95, 0.25)

    if st.sidebar.button("Analizza ETF"):
        if not isin:
            st.warning("Inserisci un ISIN.")
        else:
            with st.spinner("Calcolo in corso..."):
                try:
                    info_out, rows, frames = _cached_compute(isin, tuple(PERIODS_ALL), bench_override, rf_ann)
                    st.session_state.data_out = (info_out, rows, frames)
                except Exception as e:
                    st.error(f"Errore durante l'analisi: {e}")
                    st.session_state.data_out = None

    st.title("Analisi Singolo ETF")
    if 'data_out' in st.session_state and st.session_state.data_out:
        display_single_etf_results()
    else:
        st.info("Inserisci un ISIN e premi 'Analizza ETF'.")


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _cached_compute(isin_: str, periods_: Tuple[str, ...], bench_override_: Optional[str], rf_ann_: float):
    return compute_etf_over_periods(isin_, list(periods_), bench_override_, rf_ann_)


def display_single_etf_results():
    info_out, rows, aligned_frames = st.session_state.data_out
    etf_ticker = info_out.get("yahoo_symbol")

    # --- Trading Signal ---
    if etf_ticker:
        full_series = get_series(etf_ticker, "2y")
        signal_info = get_trading_signal(full_series) if full_series is not None and not full_series.empty else {}
        signal, reason = signal_info.get("signal", "N/A"), signal_info.get("reason", "")
        if "Compra" in signal:
            st.success(f"**Segnale Operativo: {signal}** - *{reason}*")
        elif "Vendi" in signal:
            st.error(f"**Segnale Operativo: {signal}** - *{reason}*")
        else:
            st.info(f"**Segnale Operativo: {signal}** - *{reason}*")
    st.markdown("---")

    # --- Info Generali ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ISIN", info_out.get("isin", "n.d."))
        st.metric("Ticker Yahoo", info_out.get("yahoo_symbol", "n.d."))
    with col2:
        st.metric("TER", info_out.get("ter_pct", "n.d."))
        st.metric("Dimensione Fondo", info_out.get("fund_size", "n.d."))
    with col3:
        st.metric("Metodo di Replica", info_out.get("replication_method", "n.d."))
        st.metric("Distribuzione", info_out.get("distribution", "n.d."))
    b_sym, b_name = info_out.get("benchmark_symbol"), info_out.get("benchmark_name")
    st.write(f"**Benchmark:** `{b_sym}` - *{b_name}*")

    # --- UI Metriche con nuove aggiunte ---
    selected_period_label = st.radio("Seleziona periodo:", list(PERIOD_MAP_LABEL_TO_YF.keys()), index=4,
                                     horizontal=True)
    selected_period = PERIOD_MAP_LABEL_TO_YF[selected_period_label]
    df = pd.DataFrame(rows).fillna(pd.NA)
    filtered_df = df[df['period'] == selected_period]

    if not filtered_df.empty:
        st.subheader(f"📊 Metriche di Performance e Rischio - {selected_period_label}")
        row = filtered_df.iloc[0]

        # Performance
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rendimento Totale", f"{row.get('total_return_pct', 0):.2f}%")
        c2.metric("CAGR", f"{row.get('cagr_pct', 0):.2f}%")
        c3.metric("Volatilità Ann.", f"{row.get('vol_ann_pct', 0):.2f}%")
        c4.metric("Max Drawdown", f"{row.get('mdd_pct', 0):.2f}%")

        # Risk-Adjusted e Nuove Metriche
        st.markdown("---")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Sharpe Ratio", f"{row.get('sharpe_ratio', 0):.2f}")
        c6.metric("Sortino Ratio", f"{row.get('sortino_ratio', 0):.2f}",
                  help="Simile allo Sharpe, ma considera solo la volatilità negativa. Più alto è, meglio è.")
        c7.metric("Omega Ratio", f"{row.get('omega_ratio', 0):.2f}",
                  help="Rapporto tra probabilità di guadagni e perdite. Valori > 1 indicano un profilo favorevole.")
        c8.metric("VaR 95% (1 giorno)", f"{row.get('var_95_1d_pct', 0):.2f}%",
                  help="Massima perdita attesa in 1 giorno con il 95% di confidenza.")

    # --- Grafico (invariato) ---
    st.subheader(f"📈 Grafico Performance - {selected_period_label}")
    frame = aligned_frames.get(selected_period)
    if frame is not None and not frame.empty and len(frame) > 1:
        pct_frame = frame.apply(to_percent_index)
        st.line_chart(pct_frame)
    else:
        st.warning(f"Dati insufficienti per il grafico nel periodo '{selected_period_label}'.")

    # --- Tabella Completa (aggiornata con nuove colonne) ---
    with st.expander("📋 Tabella Completa Tutti i Periodi"):
        view_cols = [
            "period", "total_return_pct", "cagr_pct", "vol_ann_pct", "mdd_pct",
            "sharpe_ratio", "sortino_ratio", "omega_ratio", "var_95_1d_pct"
        ]
        df_view = df[view_cols].rename(columns={
            "period": "Periodo", "total_return_pct": "Totale %", "cagr_pct": "CAGR %",
            "vol_ann_pct": "Vol Ann %", "mdd_pct": "Max DD %", "sharpe_ratio": "Sharpe",
            "sortino_ratio": "Sortino", "omega_ratio": "Omega", "var_95_1d_pct": "VaR 95% (1d) %"
        })
        formatters = {c: "{:.2f}%" for c in df_view.columns if "%" in c}
        formatters.update({c: "{:.2f}" for c in ["Sharpe", "Sortino", "Omega"]})
        st.dataframe(df_view.style.format(formatters, na_rep="n.d."))