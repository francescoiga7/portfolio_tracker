# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import pandas as pd
import streamlit as st

# Import condizionali per plotly
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    st.warning("⚠️ Plotly non disponibile - i grafici saranno disabilitati")

from .config import PERIODS_ALL
from .pipeline import compute_etf_over_periods
from .utils import to_percent_index
from .yahoo_client import get_series, resolve_isin_one
from .metrics import get_trading_signal

# Mappa periodi per grafici
PERIOD_MAP_LABEL_TO_YF = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd",
    "1A": "1y", "3A": "3y", "5A": "5y", "Max": "max"
}


def render_single_etf_ui():
    st.sidebar.header("Impostazioni Analisi")
    isin = st.sidebar.text_input("ISIN", value="IE00BK5BQT80").strip().upper()

    periods = PERIODS_ALL

    bench_override = st.sidebar.text_input(
        "Override Benchmark (Ticker Yahoo)",
        "",
        help="Inserisci un ticker Yahoo per sovrascrivere il benchmark.",
    ).strip() or None

    rf_ann = st.sidebar.number_input("Risk-free annuo (%)", -5.0, 10.0, 3.95, 0.25)

    if st.sidebar.button("Analizza ETF"):
        if not isin:
            st.warning("Inserisci un ISIN.")
        else:
            with st.spinner("Calcolo in corso…"):
                try:
                    info_out, rows, frames = _cached_compute(
                        isin, tuple(periods), bench_override, rf_ann
                    )
                    st.session_state.data_out = (info_out, rows, frames)
                    st.session_state.etf_params = {'periods': periods}
                except Exception as e:
                    st.error(f"Errore durante l'analisi: {e}")
                    st.session_state.data_out = None

    st.title("Analisi Singolo ETF")
    if st.session_state.data_out:
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
    signal = signal_info.get("signal", "Errore")
    reason = signal_info.get('reason', '')
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
    b_sym = info_out.get("benchmark_symbol")
    b_name = info_out.get("benchmark_name")
    st.write(f"**Benchmark:** `{b_sym}` - *{b_name}*")

    alternative_etfs = info_out.get("alternative_etfs", [])
    if alternative_etfs:
        st.subheader("Alternative con Tracking Difference Migliore")
        alt_df = pd.DataFrame(alternative_etfs)
        st.dataframe(alt_df.style.format({"td": "{:.2f}%"}), use_container_width=True)

    # --- Sezione Grafico e Metriche ---
    selected_period_label = st.radio(
        "Seleziona periodo:",
        list(PERIOD_MAP_LABEL_TO_YF.keys()),
        index=4,
        horizontal=True,
        key="etf_analysis_period_radio"
    )
    selected_period = PERIOD_MAP_LABEL_TO_YF[selected_period_label]

    df = pd.DataFrame(rows).fillna(pd.NA)
    filtered_df = df[df['period'] == selected_period] if not df.empty else pd.DataFrame()

    if not filtered_df.empty:
        st.subheader(f"📊 Metriche Performance - {selected_period_label}")
        row_data = filtered_df.iloc[0]
        # ... (codice per mostrare le metriche come prima)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_ret = row_data.get('total_return_pct')
            st.metric("Rendimento Totale", f"{total_ret:.2f}%" if pd.notna(total_ret) else "n.d.")
        with col2:
            cagr = row_data.get('cagr_pct')
            st.metric("CAGR", f"{cagr:.2f}%" if pd.notna(cagr) else "n.d.")
        with col3:
            vol = row_data.get('vol_ann_pct')
            st.metric("Volatilità Ann.", f"{vol:.2f}%" if pd.notna(vol) else "n.d.")
        with col4:
            mdd = row_data.get('mdd_pct')
            st.metric("Max Drawdown", f"{mdd:.2f}%" if pd.notna(mdd) else "n.d.")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            sharpe = row_data.get('sharpe_ratio')
            st.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "n.d.")
        with col6:
            calmar = row_data.get('calmar_ratio')
            st.metric("Calmar Ratio", f"{calmar:.2f}" if pd.notna(calmar) else "n.d.")
        with col7:
            td = row_data.get('tracking_diff_pct')
            st.metric("Tracking Diff.", f"{td:.2f}%" if pd.notna(td) else "n.d.")
        with col8:
            ter = row_data.get('ter_pct')
            st.metric("TER", ter if isinstance(ter, str) else (f"{ter:.2f}%" if pd.notna(ter) else "n.d."))

    # --- LOGICA DEL GRAFICO (INVARIATA) ---
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
            fig.update_layout(title=f"Performance vs Benchmark", yaxis_title="Performance (%)", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart_data = frame.apply(to_percent_index)
            if not chart_data.empty:
                st.line_chart(chart_data)
    else:
        st.warning(f"Dati insufficienti per visualizzare il grafico per il periodo '{selected_period_label}'.")

    with st.expander("📋 Tabella Completa Tutti i Periodi"):
        view_cols = ["period", "total_return_pct", "cagr_pct", "vol_ann_pct", "mdd_pct", "sharpe_ratio", "calmar_ratio"]
        if all(col in df.columns for col in view_cols):
            df_view = df[view_cols].rename(columns={
                "period": "Periodo", "total_return_pct": "Totale %",
                "cagr_pct": "CAGR %", "vol_ann_pct": "Vol Ann %",
                "mdd_pct": "Max DD %", "sharpe_ratio": "Sharpe", "calmar_ratio": "Calmar"
            })
            formatters = {
                "Totale %": "{:.2f}%", "CAGR %": "{:.2f}%",
                "Vol Ann %": "{:.2f}%", "Max DD %": "{:.2f}%",
                "Sharpe": "{:.2f}", "Calmar": "{:.2f}"
            }
            st.dataframe(
                df_view.style.format(formatters, na_rep="n.d."),
                use_container_width=True
            )