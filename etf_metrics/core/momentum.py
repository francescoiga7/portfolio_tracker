# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st
import yfinance as yf

from etf_metrics.clients.yahoo_client import resolve_isin_one
from etf_metrics.core.metrics import compute_metrics_from_series, compute_sharpe_ratio
from etf_metrics.core.trading import analyze_ticker
from etf_metrics.core.data_manager import MarketDataManager  # Integrazione DB


@st.cache_data(show_spinner="Gestione Dati Momentum (DB)...", ttl=60 * 30)
def fetch_momentum_data(isins: List[str]) -> List[Dict]:
    """
    1. Risolve ISIN -> Ticker.
    2. Check DB locale (Cache).
    3. Download solo differenziale.
    4. Caricamento veloce.
    """
    resolved_map = {}  # ISIN -> Ticker
    tickers_list = []

    # 1. Risoluzione ISIN
    for item in isins:
        item = item.strip().upper()
        if not item: continue

        if len(item) < 9 and not item[0].isdigit():
            ticker = item
        else:
            ticker = resolve_isin_one(item)

        if ticker:
            resolved_map[item] = ticker
            tickers_list.append(ticker)

    if not tickers_list:
        return []

    # 2. Integrazione Database Manager
    db_manager = MarketDataManager()

    # Identifica mancanti
    missing = db_manager.get_tickers_needing_update(tickers_list)

    # Download e Salvataggio mancanti
    if missing:
        try:
            bulk_df = yf.download(missing, period="1y", group_by='ticker', auto_adjust=False, threads=True)

            new_data = {}
            is_single = len(missing) == 1

            for t in missing:
                try:
                    if is_single:
                        df = bulk_df.copy()
                    else:
                        if t not in bulk_df.columns.levels[0]: continue
                        df = bulk_df[t].copy()

                    df = df.dropna(how='all')
                    if not df.empty:
                        if df.index.tz is not None: df.index = df.index.tz_localize(None)
                        new_data[t] = df
                except:
                    continue

            if new_data:
                db_manager.save_bulk_data(new_data)
        except Exception as e:
            st.error(f"Errore download: {e}")

    # 3. Caricamento Finale da DB
    loaded_data = db_manager.load_data(tickers_list)

    # 4. Ricostruzione formato output (Lista di dizionari con ISIN originale)
    fetched_data = []

    # Mappa inversa Ticker -> Lista di ISIN (un ticker potrebbe corrispondere a più ISIN in input se l'utente sbaglia, ma gestiamo il caso base)
    # Più semplice: iteriamo sugli ISIN richiesti
    for isin_orig, ticker in resolved_map.items():
        if ticker in loaded_data:
            df = loaded_data[ticker]
            if len(df) > 20:
                fetched_data.append({
                    "isin": isin_orig,
                    "ticker": ticker,
                    "series": df
                })

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

        # Filtro data corrente
        df_as_of = df[df.index <= end_date].copy()
        if df_as_of.empty: continue

        # Analisi Tecnica (Trading module)
        trading_res = analyze_ticker(ticker, df_as_of)
        signal_str = trading_res.get("signal", "N/D")

        # Calcolo Performance
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