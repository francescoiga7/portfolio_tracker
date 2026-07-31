# -*- coding: utf-8 -*-
"""Componenti di rendering Streamlit condivisi tra le varie pagine UI."""
from typing import Dict, List

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from etf_metrics.core.metrics import compute_metrics_from_series, compute_sharpe_ratio


def get_portfolio_store():
    """Restituisce il PortfolioStore conservato in session_state (caricato da JSON al primo uso)."""
    from etf_metrics.core.portfolio_tracker import PortfolioStore
    if "portfolio_store" not in st.session_state:
        st.session_state["portfolio_store"] = PortfolioStore.load_from_json()
    return st.session_state["portfolio_store"]


def style_pnl_columns(val):
    """Colore verde per valori positivi, rosso per negativi."""
    if isinstance(val, (int, float)):
        color = '#28a745' if val > 0 else '#dc3545' if val < 0 else '#333333'
        return f'color: {color}; font-weight: 600;'
    return ''


def style_trend_signal(val: str) -> str:
    """Applica uno stile colorato alla colonna dei segnali di trend."""
    val_lower = val.lower()
    if "mantieni" in val_lower:
        return 'background-color: #28a745; color: white; font-weight: bold;'
    elif "monitora" in val_lower:
        return 'background-color: #ffc107; color: black; font-weight: bold;'
    elif val_lower.startswith("vendi"):
        return 'background-color: #dc3545; color: white; font-weight: bold;'
    return ''


def format_dataframe(
    df: pd.DataFrame,
    column_config: Dict,
    pnl_cols: List[str] = [],
    bar_cols: List[str] = [],
    trend_cols: List[str] = [],
):
    """Applica formattazione completa a un DataFrame per la visualizzazione."""
    df_display = df.rename(columns=column_config)

    styler = df_display.style

    renamed_pnl_cols = [column_config.get(col) for col in pnl_cols if column_config.get(col) in df_display.columns]
    renamed_bar_cols = [column_config.get(col) for col in bar_cols if column_config.get(col) in df_display.columns]
    renamed_trend_cols = [column_config.get(col) for col in trend_cols if column_config.get(col) in df_display.columns]

    format_dict = {
        name: '€{:,.2f}' for col, name in column_config.items() if '€' in name
    }
    format_dict.update({
        name: '{:+.2f}%' for col, name in column_config.items() if '%' in name
    })
    format_dict.update({
        name: '{:.6f}' for col, name in column_config.items() if 'Quantità' in name
    })
    styler = styler.format(format_dict, na_rep="n.d.")

    if renamed_pnl_cols:
        styler = styler.apply(lambda x: x.map(style_pnl_columns), subset=renamed_pnl_cols)

    if renamed_trend_cols:
        for col_name in renamed_trend_cols:
            styler = styler.map(style_trend_signal, subset=[col_name])

    if renamed_bar_cols:
        for col_name in renamed_bar_cols:
            styler = styler.background_gradient(cmap='viridis_r', subset=[col_name])

    styler = styler.set_properties(**{'text-align': 'right'}).set_properties(
        subset=[column_config.get('isin', 'isin')], **{'text-align': 'left'}
    )

    return styler


def render_allocation_pie(labels: List[str], values: List[float], title: str, key="alloc_pie"):
    """Renderizza un grafico a torta dell'allocazione."""
    if not labels or not values or sum(values) <= 0:
        st.info("Aggiungi almeno una posizione per vedere l'allocazione.")
        return
    if HAS_PLOTLY:
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo="label+percent", pull=[0.05] * len(labels))]
        )
        fig.update_layout(
            title_text=title,
            margin=dict(t=50, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        )
        st.plotly_chart(fig, width="stretch", key=key)


def display_backtest_results(all_series: Dict[str, pd.Series], rf_ann: float, key_prefix="backtest"):
    """Visualizza l'andamento e le metriche di un set di serie di portafoglio."""
    if not all_series:
        st.error("Nessun dato da visualizzare.")
        return
    period_map = {"1M": 1, "3M": 3, "6M": 6, "YTD": "ytd", "1A": 12, "3A": 36, "5A": 60, "Max": None}
    selected_period_label = st.radio(
        "Seleziona periodo di analisi",
        list(period_map.keys()),
        index=len(period_map) - 1,
        horizontal=True,
        key=f"{key_prefix}_period_radio",
    )
    lookback_months = period_map[selected_period_label]
    end_date = pd.to_datetime('today').normalize()
    if lookback_months == "ytd":
        start_date = pd.to_datetime(f"{end_date.year}-01-01")
    elif lookback_months is not None:
        start_date = end_date - pd.DateOffset(months=lookback_months)
    else:
        start_date = min(s.index.min() for s in all_series.values() if s is not None and not s.empty)

    metrics_list, series_to_plot = [], {}
    for name, series in all_series.items():
        if series is None or series.empty:
            continue
        filtered = series[series.index >= start_date]
        if filtered.shape[0] < 2:
            continue
        series_to_plot[name] = filtered
        metrics = compute_metrics_from_series(filtered)
        metrics["sharpe"] = compute_sharpe_ratio(filtered, rf_ann)
        metrics["name"] = name
        metrics_list.append(metrics)

    st.subheader(f"Andamento Portafogli ({selected_period_label})")
    st.line_chart(series_to_plot)

    st.subheader(f"Metriche di Performance ({selected_period_label})")
    if metrics_list:
        df = pd.DataFrame(metrics_list).set_index("name")[["cagr", "vol_ann", "mdd", "sharpe"]]
        df.columns = ["CAGR %", "Volatilità Ann. %", "Max Drawdown %", "Sharpe Ratio"]
        st.dataframe(df.style.format("{:.2f}", na_rep="n.d."), width="stretch")
