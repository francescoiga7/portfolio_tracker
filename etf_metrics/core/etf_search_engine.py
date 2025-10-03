# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Iterable, Optional
from functools import lru_cache
from collections import defaultdict
import re

from etf_metrics.clients.yahoo_client import yahoo_search, get_info
from etf_metrics.shared.utils import pick_preferred_symbol
from etf_metrics.shared.config import PREFERRED_SUFFIXES

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4096)
def _get_info_for_ticker(ticker: str) -> Dict:
    """Cached function to get info for a ticker."""
    if not ticker:
        return {}
    try:
        return get_info(ticker=ticker) or {}
    except Exception:
        return {}


def _normalize_name(name: str) -> str:
    """Normalizes an ETF name for better matching."""
    if not name:
        return ""
    # Lowercase, remove common suffixes and special characters
    name = name.lower()
    name = re.sub(r'\bucits etf\b', '', name)
    name = re.sub(r'\b(usd|eur|gbp)\b', '', name)
    name = re.sub(r'\b(acc|dist|accumulating|distributing)\b', '', name)
    name = re.sub(r'\bclass\s[a-z]\b', '', name) # remove 'class a', etc.
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    return " ".join(name.split())  # Normalize whitespace

@lru_cache(maxsize=4096)
def _get_isin_for_ticker(ticker: str) -> Optional[str]:
    """Wrapper to get ISIN from the new cached info function."""
    info = _get_info_for_ticker(ticker)
    v = info.get("isin")
    if isinstance(v, str) and len(v) == 12:
        return v.strip().upper()
    # Fallback for other keys if needed
    for k, val in info.items():
        if "isin" in k.lower() and isinstance(val, str) and len(val) == 12:
            return val.strip().upper()
    return None


def get_unique_preferred_tickers(tickers: List[str]) -> List[str]:
    """
    Deduplicates a list of tickers based on ISIN and then by normalized ETF name,
    picking the preferred ticker for each unique ETF.
    """
    if not tickers:
        return []

    isin_to_candidates = defaultdict(list)
    tickers_without_isin = []
    for ticker in tickers:
        isin = _get_isin_for_ticker(ticker)
        if isin:
            isin_to_candidates[isin].append(ticker)
        else:
            tickers_without_isin.append(ticker)

    preferred_tickers_from_isin = {
        pick_preferred_symbol(candidates, PREFERRED_SUFFIXES)
        for candidates in isin_to_candidates.values()
    }
    all_preferred_tickers = list(dict.fromkeys(list(preferred_tickers_from_isin) + tickers_without_isin))

    name_to_candidates = defaultdict(list)
    tickers_without_name = []

    for ticker in all_preferred_tickers:
        if not ticker: continue
        info = _get_info_for_ticker(ticker)
        long_name = info.get("longName") or info.get("shortName")
        if long_name:
            normalized_name = _normalize_name(long_name)
            if len(normalized_name) > 10:
                name_to_candidates[normalized_name].append(ticker)
            else:
                tickers_without_name.append(ticker)
        else:
            tickers_without_name.append(ticker)

    final_tickers = []
    for name, candidates in name_to_candidates.items():
        if candidates:
            preferred = pick_preferred_symbol(candidates, PREFERRED_SUFFIXES)
            if preferred:
                final_tickers.append(preferred)

    final_tickers.extend(tickers_without_name)
    return list(dict.fromkeys(final_tickers))


def discover_universe(queries: Iterable[str], quotes_per_query: int, instrument_types: List[str] = ["ETF", "ETP", "ETN"]) -> List[str]:
    """
    Discovers a universe of instruments based on a list of queries and instrument types.
    """
    tickers_seen, ticker_list = set(), []
    for query in queries:
        try:
            quotes = yahoo_search(query, quotes_count=quotes_per_query)
            for q in quotes:
                ticker = q.get("symbol")
                quote_type = (q.get("quoteType") or "").upper()
                if ticker and ticker not in tickers_seen and quote_type in instrument_types:
                    logger.info(f"Ticker trovato: {ticker} (Tipo: {quote_type})")
                    tickers_seen.add(ticker)
                    ticker_list.append(ticker)
        except Exception as e:
            logger.warning(f"Errore ricerca per '{query}': {e}")
    return ticker_list