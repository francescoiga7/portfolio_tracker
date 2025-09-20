# -*- coding: utf-8 -*-
import logging
import time
from typing import Dict, List, Iterable, Optional
from functools import lru_cache
from collections import defaultdict
import pandas as pd
import streamlit as st
from datetime import date

from .yahoo_client import yahoo_search, get_series, get_info, resolve_isin_one
from .utils import pick_preferred_symbol
from .config import DEFAULT_SEED_QUERIES, PREFERRED_SUFFIXES
from .metrics import compute_metrics_from_series

logger = logging.getLogger(__name__)


# --- Funzioni Helper (invariate) ---
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
    isin_to_candidates, tickers_without_isin = defaultdict(list), []
    for ticker in tickers:
        isin = _get_isin_for_ticker(ticker)
        if isin:
            isin_to_candidates[isin].append(ticker)
        else:
            tickers_without_isin.append(ticker)
    isin_to_preferred_ticker = {isin: pick_preferred_symbol(c, PREFERRED_SUFFIXES) for isin, c in
                                isin_to_candidates.items()}
    final_list = [v for v in isin_to_preferred_ticker.values() if v]
    processed_bases = {t.split('.')[0] for t in final_list}
    for ticker in tickers_without_isin:
        base = ticker.split('.')[0]
        if base not in processed_bases:
            final_list.append(ticker)
            processed_bases.add(base)
    return list(dict.fromkeys(final_list))


# --- Logica dello Screener Tattico Avanzato (con supporto Point-in-Time) ---
@st.cache_data(show_spinner=False, ttl=60 * 15)
def get_market_regime(as_of_date: Optional[date] = None) -> Dict:
    """Controlla il VIX per determinare il regime di mercato a una data specifica."""
    try:
        # Carica uno storico sufficiente per avere dati a qualsiasi data recente
        vix_series = get_series("^VIX", period="5y")
        if vix_series is None or vix_series.empty:
            return {"vix": None, "regime": "Sconosciuto"}

        end_date = pd.to_datetime(as_of_date) if as_of_date else vix_series.index.max()
        series_as_of = vix_series[vix_series.index <= end_date]
        if series_as_of.empty:
            return {"vix": None, "regime": "Dati VIX non disponibili per la data"}

        current_vix = series_as_of.iloc[-1]
        regime = "Avverso al Rischio" if current_vix > 20 else "Favorevole al Rischio"
        return {"vix": current_vix, "regime": regime}
    except Exception:
        return {"vix": None, "regime": "Sconosciuto"}


def _calculate_tactical_metrics(ticker: str, as_of_date: date) -> Dict:
    """Calcola le metriche per il modello Tattico a una data specifica."""
    try:
        series_full = get_series(ticker, period="5y")  # Carica uno storico lungo
        if series_full is None: return {}

        end_date = pd.to_datetime(as_of_date)
        series = series_full[series_full.index <= end_date]  # Taglia i dati alla data scelta
        if series.empty or len(series) < 252: return {}

        info = get_info(ticker=ticker) or {}
        avg_volume_3m = info.get("averageDailyVolume3Month")  # Usa una metrica più stabile se disponibile
        avg_value_eur = (avg_volume_3m * series.iloc[-1]) if avg_volume_3m is not None else 0

        series_6m = series[series.index >= (end_date - pd.DateOffset(months=6))]
        if len(series_6m) < 60: return {}
        metrics_6m = compute_metrics_from_series(series_6m)
        cagr, mdd = metrics_6m.get("cagr"), metrics_6m.get("mdd")
        calmar_ratio = -cagr / mdd if cagr is not None and mdd is not None and mdd != 0 else None

        sma20 = series.rolling(window=20).mean()
        std20 = series.rolling(window=20).std()
        bollinger_width = ((sma20 + std20 * 2) - (sma20 - std20 * 2)) / sma20
        min_bw_6m = bollinger_width[bollinger_width.index >= (end_date - pd.DateOffset(months=6))].min()
        volatility_compression = bollinger_width.iloc[-1] / min_bw_6m if min_bw_6m > 0 else None

        high_52w = series[series.index >= (end_date - pd.DateOffset(weeks=52))].max()
        proximity_to_high = series.iloc[-1] / high_52w if high_52w > 0 else None

        return {
            "ticker": ticker, "isin": _get_isin_for_ticker(ticker), "name": info.get("longName", ticker),
            "avg_value_eur": avg_value_eur, "calmar_ratio": calmar_ratio,
            "volatility_compression": volatility_compression, "proximity_to_high": proximity_to_high,
        }
    except Exception:
        return {}


def screen_for_tactical_etfs(
        discovery_limit: int,
        min_avg_value: float,
        specific_isins: Optional[List[str]] = None,
        as_of_date: Optional[date] = None,
) -> pd.DataFrame:
    """Esegue lo screening tattico, focalizzato sugli ETP europei."""

    if specific_isins:
        unique_tickers = [resolve_isin_one(isin) for isin in specific_isins]
        unique_tickers = list(filter(None, unique_tickers))
    else:
        raw_tickers = _discover_universe(DEFAULT_SEED_QUERIES, quotes_per_query=200, limit=discovery_limit)

        # --- FILTRO PER MERCATI EUROPEI ---
        european_suffixes = [
            ".MI", ".DE", ".AS", ".L", ".PA", ".SW", ".BR", ".LS", ".IR",
            ".MC", ".HE", ".CO", ".ST", ".OL", ".VI"
        ]

        # Filtra per mantenere solo i ticker con suffisso europeo
        european_tickers = [
            ticker for ticker in raw_tickers
            if any(ticker.endswith(suffix) for suffix in european_suffixes)
        ]

        raw_tickers = european_tickers  # Sovrascrive la lista con quella filtrata

        unique_tickers = _get_unique_preferred_tickers(raw_tickers)

    analysis_date = as_of_date or date.today()

    all_metrics = []
    progress_bar = st.progress(0, text=f"Analisi di {len(unique_tickers)} ETF europei in corso...")
    for i, ticker in enumerate(unique_tickers):
        metrics = _calculate_tactical_metrics(ticker, analysis_date)
        if metrics: all_metrics.append(metrics)
        progress_bar.progress((i + 1) / len(unique_tickers), text=f"Analisi: {ticker}")
        time.sleep(0.05)
    progress_bar.empty()

    if not all_metrics: return pd.DataFrame()
    df = pd.DataFrame(all_metrics)

    essential_cols = ['calmar_ratio', 'volatility_compression', 'proximity_to_high']
    df = df.dropna(subset=essential_cols)
    if df.empty: return pd.DataFrame()

    df['avg_value_eur'] = df['avg_value_eur'].fillna(0)
    df = df[df['avg_value_eur'] >= min_avg_value]
    if df.empty: return pd.DataFrame()

    df['quality_score'] = df['calmar_ratio'].rank(pct=True) * 100
    df['squeeze_score'] = df['volatility_compression'].rank(pct=True, ascending=True) * 100
    df['proximity_score'] = df['proximity_to_high'].rank(pct=True) * 100
    df['final_score'] = (df['quality_score'] * 0.50 + df['squeeze_score'] * 0.30 + df['proximity_score'] * 0.20)

    df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    view_cols = ["ticker", "isin", "name", "final_score", "quality_score", "squeeze_score", "proximity_score",
                 "calmar_ratio", "avg_value_eur"]
    return df.reindex(columns=view_cols)