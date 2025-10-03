# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st

from etf_metrics.clients.yahoo_client import resolve_isin_one, get_series
from .metrics import compute_metrics_from_series, compute_sharpe_ratio, get_trading_signal


@st.cache_data(show_spinner="Caricamento dati storici...", ttl=60 * 30)
def fetch_momentum_data(isins: List[str]) -> List[Dict]:
    """
    Carica i dati storici completi e calcola il segnale di trading una sola volta.
    """
    fetched_data = []
    period = "5y"  # Carica uno storico sufficientemente lungo per il backtesting

    for isin in isins:
        isin = (isin or "").strip().upper()
        if not isin: continue

        ticker = resolve_isin_one(isin)
        if not ticker:
            st.warning(f"ISIN {isin} non trovato, verrà saltato.")
            continue

        series = get_series(ticker, period=period)
        if series is None or series.empty:
            st.warning(f"Nessun dato storico per {ticker} ({isin}).")
            continue

        fetched_data.append({
            "isin": isin,
            "ticker": ticker,
            "series": series,
        })
    return fetched_data


def process_momentum_rankings(
        fetched_data: List[Dict],
        lookback_months: int,
        rf_ann: float,
        end_date_override: Optional[pd.Timestamp] = None
) -> List[Dict]:
    """
    Elabora i dati per un specifico periodo, con possibilità di specificare una data di fine.
    """
    results = []
    end_date = end_date_override or pd.to_datetime("today").normalize()
    start_date = end_date - pd.DateOffset(months=lookback_months)

    for data in fetched_data:
        # Filtra la serie storica fino alla data specificata
        series_as_of = data["series"][data["series"].index <= end_date]
        if series_as_of.empty: continue

        # Calcola il segnale usando i dati fino a "end_date"
        signal_info = get_trading_signal(series_as_of)

        # Filtra per il periodo di lookback per le metriche di performance
        series_filtered = series_as_of.loc[start_date:end_date]
        if series_filtered.shape[0] < 21: continue

        metrics = compute_metrics_from_series(series_filtered) or {}
        sharpe = compute_sharpe_ratio(series_filtered, rf_ann)

        results.append({
            "isin": data["isin"],
            "ticker": data["ticker"],
            "cagr_pct": metrics.get("cagr"),
            "mdd_pct": metrics.get("mdd"),
            "sharpe_ratio": sharpe,
            "signal": signal_info.get("signal", "N/D"),
        })

    return sorted(
        results,
        key=lambda x: x["sharpe_ratio"] if x["sharpe_ratio"] is not None else float("-inf"),
        reverse=True,
    )