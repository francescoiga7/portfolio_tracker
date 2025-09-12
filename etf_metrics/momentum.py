# -*- coding: utf-8 -*-
from typing import List, Dict
import pandas as pd
import streamlit as st

from .yahoo_client import resolve_isin_one, get_series
from .metrics import compute_metrics_from_series, compute_sharpe_ratio


@st.cache_data(show_spinner=True, ttl=60 * 30)
def calculate_momentum_rankings(isins: List[str], lookback_months: int, rf_ann: float) -> List[Dict]:
    """
    Calcola un set completo di metriche per una lista di asset su un periodo di lookback
    e li ordina per Sharpe Ratio.
    """
    results: List[Dict] = []

    end_date = pd.to_datetime("today").normalize()
    start_date = end_date - pd.DateOffset(months=lookback_months)

    # Periodo dinamico per ottenere abbastanza storico da yfinance (con piccolo buffer)
    period_months = max(lookback_months + 3, 36)  # garantisce almeno 36 mesi
    period = f"{period_months}mo"

    for isin in isins:
        isin = (isin or "").strip().upper()
        if not isin:
            continue
        ticker = resolve_isin_one(isin)
        if not ticker:
            st.warning(f"ISIN {isin} non trovato, verrà saltato.")
            continue

        series = get_series(ticker, period=period)
        if series is None or series.empty:
            st.warning(f"Nessun dato storico per {ticker} ({isin}).")
            continue

        series_filtered = series.loc[start_date:end_date]
        if series_filtered.shape[0] < 21:  # richiede almeno ~1 mese di dati
            st.warning(
                f"Dati insufficienti per {ticker} ({isin}) nel periodo di {lookback_months} mesi."
            )
            continue

        # Calcola le metriche (gestendo eventuale None)
        metrics = compute_metrics_from_series(series_filtered) if series_filtered is not None else {}
        sharpe = compute_sharpe_ratio(series_filtered, rf_ann) if series_filtered is not None else None

        results.append(
            {
                "isin": isin,
                "ticker": ticker,
                "cagr_pct": metrics.get("cagr"),
                "mdd_pct": metrics.get("mdd"),
                "sharpe_ratio": sharpe,
                "start_date": series_filtered.index[0].strftime("%Y-%m-%d"),
                "end_date": series_filtered.index[-1].strftime("%Y-%m-%d"),
            }
        )

    # Ordina per Sharpe Ratio (discendente), i None finiscono in fondo
    ranked_results = sorted(
        results,
        key=lambda x: x["sharpe_ratio"] if x["sharpe_ratio"] is not None else float("-inf"),
        reverse=True,
    )
    return ranked_results