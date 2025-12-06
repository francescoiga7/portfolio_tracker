# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st
import yfinance as yf

from etf_metrics.clients.yahoo_client import resolve_isin_one
from etf_metrics.core.metrics import compute_metrics_from_series, compute_sharpe_ratio
from etf_metrics.core.trading import analyze_ticker


@st.cache_data(show_spinner="Risoluzione ISIN e Download Multiplo...", ttl=60 * 30)
def fetch_momentum_data(isins: List[str]) -> List[Dict]:
    """
    1. Risolve ISIN -> Ticker.
    2. Esegue BULK DOWNLOAD da Yahoo (molto più veloce).
    3. Restituisce lista di dizionari pronti per l'analisi.
    """
    resolved_map = {}
    tickers_to_download = []

    for item in isins:
        item = item.strip().upper()
        if not item: continue

        if len(item) < 9 and not item[0].isdigit():
            ticker = item
        else:
            ticker = resolve_isin_one(item)

        if ticker:
            resolved_map[item] = ticker
            tickers_to_download.append(ticker)

    if not tickers_to_download:
        return []

    try:
        # group_by='ticker' restituisce un DataFrame con colonne (Ticker, OHLC)
        # threads=True attiva il download parallelo di yfinance
        bulk_df = yf.download(tickers_to_download, period="5y", group_by='ticker', auto_adjust=False, threads=True)
    except Exception as e:
        st.error(f"Errore download massivo: {e}")
        return []

    fetched_data = []

    is_single = len(tickers_to_download) == 1

    for isin_orig, ticker in resolved_map.items():
        try:
            if is_single:
                df = bulk_df.copy()
            else:
                df = bulk_df[ticker].copy()

            df = df.dropna(how='all')
            if df.empty or len(df) < 50: continue

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            fetched_data.append({
                "isin": isin_orig,
                "ticker": ticker,
                "series": df,
            })
        except KeyError:
            continue

    return fetched_data


def process_momentum_rankings(
        fetched_data: List[Dict],
        lookback_months: int,
        rf_ann: float,
        end_date_override: Optional[pd.Timestamp] = None
) -> List[Dict]:
    results = []
    end_date = end_date_override or pd.to_datetime("today").normalize()
    start_date = end_date - pd.DateOffset(months=lookback_months)

    for data in fetched_data:
        df = data["series"]
        ticker = data["ticker"]

        df_as_of = df[df.index <= end_date].copy()
        if df_as_of.empty: continue

        trading_res = analyze_ticker(ticker, df_as_of)
        signal_str = trading_res.get("signal", "N/D")

        df_filtered = df_as_of.loc[start_date:end_date]
        if len(df_filtered) < 21: continue

        close_series = df_filtered['Close']
        metrics = compute_metrics_from_series(close_series) or {}
        sharpe = compute_sharpe_ratio(close_series, rf_ann)

        results.append({
            "isin": data["isin"],
            "ticker": ticker,
            "cagr_pct": metrics.get("cagr"),
            "mdd_pct": metrics.get("mdd"),
            "sharpe_ratio": sharpe,
            "signal": signal_str,
        })

    return sorted(
        results,
        key=lambda x: x["sharpe_ratio"] if x["sharpe_ratio"] is not None else float("-inf"),
        reverse=True,
    )