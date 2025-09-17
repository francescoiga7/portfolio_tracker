# -*- coding: utf-8 -*-
import logging
import time
from typing import Dict, List, Iterable, Optional
from functools import lru_cache
from collections import defaultdict
import pandas as pd
import streamlit as st

from .yahoo_client import yahoo_search, get_series, get_info
from .utils import pick_preferred_symbol
from .config import DEFAULT_SEED_QUERIES, PREFERRED_SUFFIXES

logger = logging.getLogger(__name__)


# --- Funzioni Helper per la Scoperta (con de-duplicazione migliorata) ---
@lru_cache(maxsize=4096)
def _get_isin_for_ticker(ticker: str) -> Optional[str]:
    if not ticker: return None
    try:
        info = get_info(ticker=ticker) or {}
        for k, v in info.items():
            if "isin" in k.lower() and isinstance(v, str) and len(v) == 12:
                return v.strip().upper()
    except Exception:
        pass
    return None


def _is_valid_etf_quote(q: Dict) -> bool:
    return (q.get("quoteType") or "").upper() in ["ETF", "ETP", "ETN"]


def _discover_universe(queries: Iterable[str], quotes_per_query: int, limit: int) -> List[str]:
    tickers_seen, ticker_list = set(), []
    for query in queries:
        try:
            quotes = yahoo_search(query, quotes_count=quotes_per_query)
            for q in quotes:
                ticker = q.get("symbol")
                if ticker and ticker not in tickers_seen and _is_valid_etf_quote(q):
                    tickers_seen.add(ticker)
                    ticker_list.append(ticker)
            if len(ticker_list) >= limit: break
        except Exception as e:
            logger.warning(f"Errore ricerca per '{query}': {e}")
    return ticker_list


def _get_unique_preferred_tickers(tickers: List[str]) -> List[str]:
    """Logica di de-duplicazione migliorata."""
    isin_to_candidates = defaultdict(list)
    tickers_without_isin = []
    for ticker in tickers:
        isin = _get_isin_for_ticker(ticker)
        if isin:
            isin_to_candidates[isin].append(ticker)
        else:
            tickers_without_isin.append(ticker)

    isin_to_preferred_ticker = {
        isin: pick_preferred_symbol(candidates, PREFERRED_SUFFIXES)
        for isin, candidates in isin_to_candidates.items()
    }

    final_list = [v for v in isin_to_preferred_ticker.values() if v]
    processed_bases = {t.split('.')[0] for t in final_list}

    for ticker in tickers_without_isin:
        base = ticker.split('.')[0]
        if base not in processed_bases:
            final_list.append(ticker)
            processed_bases.add(base)

    return final_list


# --- Logica dello Screener "Breakout Potential" (con pausa di sicurezza) ---
@st.cache_data(show_spinner=False, ttl=60 * 60)
def _get_benchmark_series(period: str) -> Optional[pd.Series]:
    return get_series("IWDA.AS", period=period)


def _calculate_metrics_for_breakout(ticker: str, benchmark_series: pd.Series) -> Dict:
    # ... (logica interna invariata) ...
    try:
        series = get_series(ticker, period="2y")
        if series is None or len(series) < 252: return {}
        end_date = series.index.max()
        start_date_12m = end_date - pd.DateOffset(months=12)
        start_date_52w = end_date - pd.DateOffset(weeks=52)
        sma20 = series.rolling(window=20).mean()
        std20 = series.rolling(window=20).std()
        bollinger_width = ((sma20 + std20 * 2) - (sma20 - std20 * 2)) / sma20
        current_bw = bollinger_width.iloc[-1]
        min_bw_12m = bollinger_width.loc[start_date_12m:].min()
        volatility_compression = current_bw / min_bw_12m
        combined = pd.concat([series.rename('asset'), benchmark_series.rename('benchmark')], axis=1, join='inner')
        if len(combined) < 100: return {}
        rs_ratio = combined['asset'] / combined['benchmark']
        rs_sma = rs_ratio.rolling(window=50).mean()
        relative_strength = rs_ratio.iloc[-1] / rs_sma.iloc[-1]
        high_52w = series.loc[start_date_52w:].max()
        current_price = series.iloc[-1]
        proximity_to_high = current_price / high_52w
        info = get_info(ticker=ticker) or {}
        return {"ticker": ticker, "name": info.get("longName", ticker),
                "volatility_compression": volatility_compression, "relative_strength": relative_strength,
                "proximity_to_high": proximity_to_high, "price": current_price, "high_52w": high_52w}
    except Exception:
        return {}


def screen_for_breakout_etfs(discovery_limit: int = 1000) -> pd.DataFrame:
    raw_tickers = _discover_universe(DEFAULT_SEED_QUERIES, quotes_per_query=200, limit=discovery_limit)
    unique_tickers = _get_unique_preferred_tickers(raw_tickers)

    benchmark_series = _get_benchmark_series("2y")
    if benchmark_series is None:
        st.error("Impossibile caricare i dati del benchmark. Riprova più tardi.")
        return pd.DataFrame()

    all_metrics = []
    progress_bar = st.progress(0, text=f"Analisi di {len(unique_tickers)} ETF unici in corso...")
    for i, ticker in enumerate(unique_tickers):
        metrics = _calculate_metrics_for_breakout(ticker, benchmark_series)
        if metrics: all_metrics.append(metrics)
        progress_bar.progress((i + 1) / len(unique_tickers), text=f"Analisi: {ticker}")
        time.sleep(0.05)  # PAUSA DI SICUREZZA
    progress_bar.empty()

    if not all_metrics: return pd.DataFrame()
    df = pd.DataFrame(all_metrics).dropna()
    if df.empty: return pd.DataFrame()

    df['squeeze_score'] = df['volatility_compression'].rank(pct=True, ascending=True) * 100
    df['rs_score'] = df['relative_strength'].rank(pct=True) * 100
    df['proximity_score'] = df['proximity_to_high'].rank(pct=True) * 100

    df['final_score'] = (df['squeeze_score'] * 0.50 + df['rs_score'] * 0.30 + df['proximity_score'] * 0.20)

    df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    view_cols = ["ticker", "name", "final_score", "squeeze_score", "rs_score", "proximity_score", "price", "high_52w"]
    return df.reindex(columns=view_cols)