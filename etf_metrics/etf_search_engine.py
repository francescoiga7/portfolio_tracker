# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Iterable, Optional
from functools import lru_cache
from collections import defaultdict
import pandas as pd
import streamlit as st

from .yahoo_client import yahoo_search, get_series, get_info
from .metrics import compute_sharpe_ratio
from .etf_info import fallback_ter_from_yahoo_info
from .utils import pick_preferred_symbol
from .config import DEFAULT_SEED_QUERIES

logger = logging.getLogger(__name__)


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
    tickers_seen = set()
    ticker_list = []
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
    isin_to_tickers = defaultdict(list)
    tickers_without_isin = []
    for ticker in tickers:
        isin = _get_isin_for_ticker(ticker)
        if isin:
            isin_to_tickers[isin].append(ticker)
        else:
            tickers_without_isin.append(ticker)
    preferred_tickers = [pick_preferred_symbol(c) for c in isin_to_tickers.values() if c]
    preferred_tickers.extend(tickers_without_isin)
    return list(dict.fromkeys(preferred_tickers))


def _calculate_metrics_for_screener(ticker: str) -> Dict:
    try:
        series = get_series(ticker, period="18mo")
        if series is None or len(series) < 252: return {}
        series_12m = series.last('12M')
        if len(series_12m) < 200: return {}
        info = get_info(ticker=ticker) or {}

        sharpe = compute_sharpe_ratio(series_12m, rf_annual_pct=3.0)
        volatility = series_12m.pct_change().std() * (252 ** 0.5) * 100
        ter = fallback_ter_from_yahoo_info(info)
        total_assets = info.get("totalAssets")

        return {
            "ticker": ticker, "name": info.get("longName", ticker),
            "sharpe_ratio": sharpe, "volatility": volatility,
            "ter": ter, "total_assets": total_assets
        }
    except Exception:
        return {}


def screen_best_etf_for_pac(discovery_limit: int = 1000) -> pd.DataFrame:
    raw_tickers = _discover_universe(DEFAULT_SEED_QUERIES, quotes_per_query=200, limit=discovery_limit)
    unique_tickers = _get_unique_preferred_tickers(raw_tickers)

    all_metrics = []
    progress_bar = st.progress(0, text="Analisi ETF in corso...")
    for i, ticker in enumerate(unique_tickers):
        metrics = _calculate_metrics_for_screener(ticker)
        if metrics: all_metrics.append(metrics)
        progress_bar.progress((i + 1) / len(unique_tickers))
    progress_bar.empty()

    if not all_metrics: return pd.DataFrame()
    df = pd.DataFrame(all_metrics).dropna(subset=['sharpe_ratio', 'volatility', 'ter', 'total_assets'])
    if df.empty: return pd.DataFrame()

    df['momentum_score'] = df['sharpe_ratio'].rank(pct=True) * 100
    df['cost_score'] = df['ter'].rank(pct=True, ascending=False) * 100
    df['volatility_score'] = df['volatility'].rank(pct=True, ascending=False) * 100
    df['size_score'] = df['total_assets'].rank(pct=True) * 100
    df['final_score'] = (df['momentum_score'] * 0.40 + df['cost_score'] * 0.30 + df['volatility_score'] * 0.20 + df[
        'size_score'] * 0.10)

    df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    view_cols = ["ticker", "name", "final_score", "momentum_score", "cost_score", "volatility_score", "size_score",
                 "sharpe_ratio", "ter", "volatility"]
    return df.reindex(columns=view_cols)