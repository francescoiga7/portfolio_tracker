# -*- coding: utf-8 -*-
from typing import List, Dict
import pandas as pd
import streamlit as st

from .yahoo_client import resolve_isin_one, get_series
from .metrics import compute_metrics_from_series, compute_sharpe_ratio, get_trading_signal


@st.cache_data(show_spinner=True, ttl=60 * 30)
def calculate_momentum_rankings(isins: List[str], lookback_months: int, rf_ann: float) -> List[Dict]:
    """
    Calcola un set completo di metriche per una lista di asset su un periodo di lookback
    e li ordina per Sharpe Ratio, aggiungendo un segnale di trading.
    """
    results: List[Dict] = []

    end_date = pd.to_datetime("today").normalize()
    start_date = end_date - pd.DateOffset(months=lookback_months)

    # Periodo dinamico per ottenere abbastanza storico (almeno 3 anni per la SMA200)
    period_months = max(lookback_months + 3, 36)
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

        # Filtra la serie per il periodo di lookback per le metriche di performance
        series_filtered = series.loc[start_date:end_date]
        if series_filtered.shape[0] < 21:  # richiede almeno ~1 mese di dati
            st.warning(
                f"Dati insufficienti per {ticker} ({isin}) nel periodo di {lookback_months} mesi."
            )
            continue

        # Calcola le metriche di performance sulla serie filtrata
        metrics = compute_metrics_from_series(series_filtered) or {}
        sharpe = compute_sharpe_ratio(series_filtered, rf_ann)

        # Calcola il segnale di trading sulla serie COMPLETA per avere dati sufficienti per la SMA200
        signal_info = get_trading_signal(series)

        results.append(
            {
                "isin": isin,
                "ticker": ticker,
                "cagr_pct": metrics.get("cagr"),
                "mdd_pct": metrics.get("mdd"),
                "sharpe_ratio": sharpe,
                "signal": signal_info.get("signal", "N/D"), # Aggiungi il segnale
            }
        )

    # Ordina per Sharpe Ratio (discendente), i None finiscono in fondo
    ranked_results = sorted(
        results,
        key=lambda x: x["sharpe_ratio"] if x["sharpe_ratio"] is not None else float("-inf"),
        reverse=True,
    )
    return ranked_results