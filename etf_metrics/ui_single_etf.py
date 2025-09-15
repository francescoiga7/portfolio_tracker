# -*- coding: utf-8 -*-
from typing import Optional, Tuple, List, Dict
import re
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

# Mappa periodi per grafici e confronto (condivisa nel modulo)
PERIOD_MAP_LABEL_TO_YF = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd",
    "1A": "1y", "3A": "3y", "5A": "5y", "Max": "max"
}


def render_single_etf_ui():
    st.sidebar.header("Impostazioni Analisi")
    isin = st.sidebar.text_input("ISIN", value="IE00BK5BQT80").strip().upper()

    # Periodi sempre tutti – coerente con l'impostazione originale
    periods = PERIODS_ALL  # Usa tutti i periodi disponibili

    bench_override = st.sidebar.text_input(
        "Override Benchmark (Ticker Yahoo)",
        "",
        help="Inserisci un ticker Yahoo per sovrascrivere il benchmark. "
             "Se l'ISIN non è mappato, verrà aggiunto automaticamente alla configurazione.",
    ).strip() or None

    rf_ann = st.sidebar.number_input("Risk-free annuo (%)", -5.0, 10.0, 1.0, 0.25)

    compare_input = st.sidebar.text_area("Confronta con (ISIN o Ticker)", "IE00B4L5Y983, SPY")

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
                    st.session_state.etf_params = {'compare_input': compare_input, 'periods': periods}
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
    params = st.session_state.etf_params
    periods = params['periods']

    # Header con informazioni estese
    st.subheader("Informazioni ETF")
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

    # Mostra alternative con tracking difference migliore (se presenti)
    alternative_etfs = info_out.get("alternative_etfs", [])
    if alternative_etfs:
        st.subheader("Alternative con Tracking Difference Migliore")
        alt_df = pd.DataFrame(alternative_etfs)
        st.dataframe(alt_df.style.format({"td": "{:.2f}%"}), use_container_width=True)

    # Tabs: Grafico e Confronto
    tabs = st.tabs(["📈 Grafico", "🔍 Confronto"])

    # --- Tab Grafico ---
    with tabs[0]:
        selected_period_label = st.radio(
            "Seleziona periodo:",
            list(PERIOD_MAP_LABEL_TO_YF.keys()),
            index=4,  # "1A"
            horizontal=True,
            key="etf_analysis_period_radio"
        )
        selected_period = PERIOD_MAP_LABEL_TO_YF[selected_period_label]

        df = pd.DataFrame(rows).fillna(pd.NA)
        filtered_df = df[df['period'] == selected_period] if not df.empty else pd.DataFrame()

        # Metriche del periodo selezionato
        if not filtered_df.empty:
            st.subheader(f"📊 Metriche Performance - {selected_period_label}")
            row_data = filtered_df.iloc[0]

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

        # Grafico
        frame = aligned_frames.get(selected_period)
        if frame is not None and not frame.empty and len(frame) > 1:
            if HAS_PLOTLY:
                fig = go.Figure()
                for col in frame.columns:
                    try:
                        series = frame[col].dropna()
                        if len(series) < 2:
                            st.warning(f"Serie {col} ha dati insufficienti ({len(series)} punti)")
                            continue
                        pct = to_percent_index(series)
                        if pct.empty:
                            st.warning(f"Serie percentuale vuota per {col}")
                            continue
                        fig.add_trace(
                            go.Scatter(
                                x=pct.index, y=pct.values, mode='lines', name=col,
                                line=dict(width=2), connectgaps=True
                            )
                        )
                        # Media mobile 50 giorni solo per l'ETF (prima colonna)
                        if len(series) >= 50 and col == frame.columns[0]:
                            ma_50 = series.rolling(window=50).mean()
                            ma_50_pct = to_percent_index(ma_50.dropna())
                            if not ma_50_pct.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=ma_50_pct.index, y=ma_50_pct.values, mode='lines',
                                        name=f'{col} MA50',
                                        line=dict(width=1.5, dash='dash', color='rgba(128,128,128,0.7)'),
                                        connectgaps=True
                                    )
                                )
                    except Exception as e:
                        st.error(f"Errore nella creazione del grafico per {col}: {e}")
                        continue
                if fig.data:
                    fig.update_layout(
                        title=f"Performance vs Benchmark - {selected_period_label}",
                        yaxis_title="Performance (%)",
                        xaxis_title="Data",
                        hovermode='x unified',
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"etf_chart_{selected_period}")
                else:
                    st.error("Nessuna traccia valida da visualizzare nel grafico")
            else:
                try:
                    chart_data = frame.apply(to_percent_index)
                    if not chart_data.empty:
                        st.line_chart(chart_data)
                    else:
                        st.warning("Dati del grafico vuoti dopo conversione percentuale")
                except Exception as e:
                    st.error(f"Errore nella creazione del grafico fallback: {e}")
        else:
            if frame is None:
                st.warning(f"Nessun dato di allineamento disponibile per il periodo {selected_period_label}")
            elif frame.empty:
                st.warning(f"Frame vuoto per il periodo {selected_period_label}")
            elif len(frame) <= 1:
                st.warning(f"Dati insufficienti per il grafico (solo {len(frame)} punto/i)")

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

    # --- Tab Confronto ---
    with tabs[1]:
        display_comparison_tab(params['compare_input'])


def display_comparison_tab(compare_input: str):
    st.subheader("Confronto ETF su diversi periodi")
    # Risoluzione ISIN o Ticker
    inputs = [item.strip().upper() for item in re.split(r'[,\\n]', compare_input) if item.strip()]
    tickers_to_compare: List[str] = []
    with st.spinner("Risoluzione ISIN in corso..."):
        for item in inputs:
            if re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', item):  # ISIN
                ticker = resolve_isin_one(item)
                if ticker:
                    tickers_to_compare.append(ticker)
                else:
                    st.warning(f"ISIN {item} non trovato, verrà saltato.")
            else:  # Ticker
                tickers_to_compare.append(item)
    tickers = list(set(tickers_to_compare))  # dedup
    if not tickers:
        st.info("Aggiungi ISIN o Ticker validi nella sidebar per confrontarli.")
        return
    selected_period_label = st.radio(
        "Seleziona periodo",
        list(PERIOD_MAP_LABEL_TO_YF.keys()),
        index=4, horizontal=True, key="comparison_period_radio"
    )
    period_yf = PERIOD_MAP_LABEL_TO_YF[selected_period_label]

    data = {ticker: get_series(ticker, period_yf) for ticker in tickers}

    if HAS_PLOTLY:
        fig = go.Figure()
        for ticker, series in data.items():
            if series is not None and not series.empty:
                pct = to_percent_index(series)
                fig.add_trace(go.Scatter(x=pct.index, y=pct.values, mode='lines', name=ticker))
        fig.update_layout(title=f"Andamento ETF ({selected_period_label})", yaxis_title="% dal primo punto")
        st.plotly_chart(fig, use_container_width=True, key="comparison_chart")
    else:
        chart_data = pd.DataFrame({
            ticker: to_percent_index(series)
            for ticker, series in data.items()
            if series is not None and not series.empty
        })
        if not chart_data.empty:
            st.line_chart(chart_data)