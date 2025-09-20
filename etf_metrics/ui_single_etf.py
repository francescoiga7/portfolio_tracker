# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    st.warning("⚠️ Plotly non disponibile - i grafici saranno disabilitati")

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

    periods = PERIODS_ALL
    bench_override = st.sidebar.text_input(
        "Override Benchmark (Ticker Yahoo)", "",
        help="Inserisci un ticker Yahoo per sovrascrivere il benchmark.",
    ).strip() or None
    rf_ann = st.sidebar.number_input("Risk-free annuo (%)", -5.0, 10.0, 3.95, 0.25)

    if st.sidebar.button("Analizza ETF"):
        if not isin:
            st.warning("Inserisci un ISIN.")
        else:
            with st.spinner("Calcolo in corso…"):
                try:
                    info_out, rows, frames = _cached_compute(isin, tuple(periods), bench_override, rf_ann)
                    st.session_state.data_out = (info_out, rows, frames)
                    st.session_state.etf_params = {'periods': periods}
                except Exception as e:
                    st.error(f"Errore durante l'analisi: {e}")
                    st.session_state.data_out = None

    st.title("Analisi Singolo ETF")
    if 'data_out' in st.session_state and st.session_state.data_out:
        display_single_etf_results()
    else:
        st.info("Inserisci un ISIN e premi 'Analizza ETF'.")


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _cached_compute(
        isin_: str,
        periods_: Tuple[str, ...],
        bench_override_: Optional[str],
        rf_ann_: float
):
    return compute_etf_over_periods(isin_, list(periods_), bench_override_, rf_ann_)


def display_single_etf_results():
    info_out, rows, aligned_frames = st.session_state.data_out
    etf_ticker = info_out.get("yahoo_symbol")

    signal_info = {"signal": "N/A", "reason": "Ticker non disponibile"}
    if etf_ticker:
        full_series = get_series(etf_ticker, "2y")
        if full_series is not None and not full_series.empty:
            signal_info = get_trading_signal(full_series)
        else:
            signal_info = {"signal": "Dati Insufficienti", "reason": "Impossibile scaricare lo storico dei prezzi."}

    st.subheader("Informazioni ETF")
    signal, reason = signal_info.get("signal", "Errore"), signal_info.get('reason', '')
    if "Compra" in signal:
        st.success(f"**Segnale Operativo: {signal}**")
        st.caption(f"_{reason}_")
    elif "Vendi" in signal:
        st.error(f"**Segnale Operativo: {signal}**")
        st.caption(f"_{reason}_")
    else:
        st.info(f"**Segnale Operativo: {signal}**")
        st.caption(f"_{reason}_")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ISIN", info_out.get("isin", "n.d."))
        st.metric("Ticker Yahoo", info_out.get("yahoo_symbol", "n.d."))
        st.metric("Categoria", info_out.get("category", "n.d."))
    with col2:
        st.metric("TER", info_out.get("ter_pct", "n.d."))
        st.metric("Tracking Difference", info_out.get("td_external", "n.d."))
        st.metric("Dimensione Fondo", info_out.get("fund_size", "n.d."))
    with col3:
        st.metric("Metodo di Replica", info_out.get("replication_method", "n.d."))
        st.metric("Distribuzione", info_out.get("distribution", "n.d."))

    b_sym, b_name = info_out.get("benchmark_symbol"), info_out.get("benchmark_name")
    st.write(f"**Benchmark:** `{b_sym}` - *{b_name}*")

    selected_period_label = st.radio(
        "Seleziona periodo:", list(PERIOD_MAP_LABEL_TO_YF.keys()), index=4,
        horizontal=True, key="etf_analysis_period_radio"
    )
    selected_period = PERIOD_MAP_LABEL_TO_YF[selected_period_label]
    df = pd.DataFrame(rows).fillna(pd.NA)
    filtered_df = df[df['period'] == selected_period] if not df.empty else pd.DataFrame()

    if not filtered_df.empty:
        st.subheader(f"📊 Metriche Performance - {selected_period_label}")
        row_data = filtered_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rendimento Totale", f"{row_data.get('total_return_pct'):.2f}%" if pd.notna(row_data.get('total_return_pct')) else "n.d.")
        c2.metric("CAGR", f"{row_data.get('cagr_pct'):.2f}%" if pd.notna(row_data.get('cagr_pct')) else "n.d.")
        c3.metric("Volatilità Ann.", f"{row_data.get('vol_ann_pct'):.2f}%" if pd.notna(row_data.get('vol_ann_pct')) else "n.d.")
        c4.metric("Max Drawdown", f"{row_data.get('mdd_pct'):.2f}%" if pd.notna(row_data.get('mdd_pct')) else "n.d.")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Sharpe Ratio", f"{row_data.get('sharpe_ratio'):.2f}" if pd.notna(row_data.get('sharpe_ratio')) else "n.d.")
        c6.metric("Calmar Ratio", f"{row_data.get('calmar_ratio'):.2f}" if pd.notna(row_data.get('calmar_ratio')) else "n.d.")
        c7.metric("Tracking Diff.", f"{row_data.get('tracking_diff_pct'):.2f}%" if pd.notna(row_data.get('tracking_diff_pct')) else "n.d.")
        ter = row_data.get('ter_pct')
        c8.metric("TER", ter if isinstance(ter, str) else (f"{ter:.2f}%" if pd.notna(ter) else "n.d."))

    st.subheader(f"📈 Grafico Performance - {selected_period_label}")
    frame = aligned_frames.get(selected_period)
    if frame is not None and not frame.empty and len(frame) > 1:
        if HAS_PLOTLY:
            fig = go.Figure()
            for col in frame.columns:
                series = frame[col].dropna()
                if len(series) < 2: continue
                pct = to_percent_index(series)
                fig.add_trace(go.Scatter(x=pct.index, y=pct.values, mode='lines', name=col))
            fig.update_layout(title="Performance vs Benchmark", yaxis_title="Performance (%)", height=500)
            st.plotly_chart(fig,  width="stretch")
        else:
            chart_data = frame.apply(to_percent_index)
            if not chart_data.empty: st.line_chart(chart_data)
    else:
        st.warning(f"Dati insufficienti per il grafico nel periodo '{selected_period_label}'.")

    with st.expander("📋 Tabella Completa Tutti i Periodi"):
        view_cols = ["period", "total_return_pct", "cagr_pct", "vol_ann_pct", "mdd_pct", "sharpe_ratio", "calmar_ratio"]
        if all(c in df.columns for c in view_cols):
            df_view = df[view_cols].rename(columns={
                "period": "Periodo", "total_return_pct": "Totale %", "cagr_pct": "CAGR %",
                "vol_ann_pct": "Vol Ann %", "mdd_pct": "Max DD %", "sharpe_ratio": "Sharpe", "calmar_ratio": "Calmar"
            })
            formatters = {c: "{:.2f}%" for c in ["Totale %", "CAGR %", "Vol Ann %", "Max DD %"]}
            formatters.update({c: "{:.2f}" for c in ["Sharpe", "Calmar"]})
            st.dataframe(df_view.style.format(formatters, na_rep="n.d."),  width="stretch")