# -*- coding: utf-8 -*-
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from etf_metrics.shared.config import PERIODS_ALL
from etf_metrics.core.single_asset_analysis import compute_single_asset_analysis
from etf_metrics.shared.utils import to_percent_index

PERIOD_MAP_LABEL_TO_YF = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd",
    "1A": "1y", "3A": "3y", "5A": "5y", "Max": "max"
}


def render_single_etf_ui():
    st.title("🔎 Analisi ETF & Azioni")
    st.caption("Analisi tecnica e fondamentale centralizzata.")

    st.sidebar.header("⚙️ Impostazioni")
    isin = st.sidebar.text_input("ISIN o Ticker", value="IE00BK5BQT80").strip().upper()
    bench_override = st.sidebar.text_input("Override Benchmark (Ticker Yahoo)", "").strip() or None
    rf_ann = st.sidebar.number_input("Risk-free annuo (%)", -5.0, 10.0, 3.95, 0.25)

    if st.sidebar.button("Analizza"):
        if not isin:
            st.warning("Inserisci un ISIN.")
        else:
            with st.spinner("Analisi 10y e calcolo metriche..."):
                try:
                    info, rows, frames = _cached_compute(isin, tuple(PERIODS_ALL), bench_override, rf_ann)
                    st.session_state.data_out = (info, rows, frames)
                except Exception as e:
                    st.error(f"Errore: {e}")

    if 'data_out' in st.session_state and st.session_state.data_out:
        display_single_etf_results()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _cached_compute(isin_, periods_, bench_, rf_):
    return compute_single_asset_analysis(isin_, list(periods_), bench_, rf_)


def display_single_etf_results():
    info, rows, frames = st.session_state.data_out

    ts = info.get('trading_signal', {})
    if ts:
        sig = ts.get('signal', 'N/A')
        conf = ts.get('confidence', 0)
        reason = ts.get('reason', '')

        color = "blue"
        if "COMPRA" in sig:
            color = "green"
        elif "VENDI" in sig:
            color = "red"
        elif "MANTIENI" in sig:
            color = "orange"

        st.markdown(f"### Segnale AI: :{color}[{sig}] (Confidenza: {conf}/100)")
        st.caption(f"Motivazione: {reason}")

        cols = st.columns(4)
        inds = ts.get('indicators', {})
        cols[0].metric("Prezzo", f"{ts.get('price', 0):.2f}")
        cols[1].metric("Stop Loss", f"{ts.get('stop_loss', 0):.2f}")
        cols[2].metric("RSI (14)", f"{inds.get('RSI', 0):.1f}")
        cols[3].metric("Trend (ADX)", f"{inds.get('ADX', 0):.1f}")
        st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Nome", info.get('name'))
    c2.metric("Categoria", info.get('category'))
    c3.metric("TER", info.get('ter_pct'))

    st.markdown("---")

    st.subheader("Performance e Rischio")
    selected_period_label = st.radio(
        "Orizzonte:",
        list(PERIOD_MAP_LABEL_TO_YF.keys()),
        index=4,
        horizontal=True,
        label_visibility="collapsed"
    )
    selected_period_yf = PERIOD_MAP_LABEL_TO_YF[selected_period_label]

    current_metrics = next((r for r in rows if r['period'] == selected_period_yf), None)

    if current_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rendimento", f"{current_metrics.get('total_return_pct', 0):.2f}%")
        m2.metric("CAGR", f"{current_metrics.get('cagr_pct', 0):.2f}%")
        m3.metric("Volatilità", f"{current_metrics.get('vol_ann_pct', 0):.2f}%")
        m4.metric("Max DD", f"{current_metrics.get('mdd_pct', 0):.2f}%")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Sharpe", f"{current_metrics.get('sharpe_ratio', 0):.2f}")
        r2.metric("Sortino", f"{current_metrics.get('sortino_ratio', 0):.2f}")
        r3.metric("Omega", f"{current_metrics.get('omega_ratio', 0):.2f}")
        r4.metric("VaR 95%", f"{current_metrics.get('var_95_1d_pct', 0):.2f}%")
    else:
        st.warning(f"Dati metrici non disponibili per il periodo.")

    st.markdown("###")
    if selected_period_yf in frames:
        fr = frames[selected_period_yf]
        if fr is not None and not fr.empty:

            st.caption("📈 Performance Relativa (%)")
            st.line_chart(fr.apply(to_percent_index))

            st.caption("📉 Drawdown (Rischio/Dolore)")
            wealth_index = fr / fr.iloc[0]
            previous_peaks = wealth_index.cummax()
            drawdown = (wealth_index - previous_peaks) / previous_peaks

            if len(drawdown.columns) == 2:
                chart_colors = ["#ff4b4b", "#a0a0a0"]
            else:
                chart_colors = ["#ff4b4b"]

            st.area_chart(drawdown, color=chart_colors)

        else:
            st.info(f"Dati grafici insufficienti.")