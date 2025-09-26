# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Iterable, Optional
from functools import lru_cache
from collections import defaultdict

from .yahoo_client import yahoo_search, get_info
from .utils import pick_preferred_symbol
from .config import PREFERRED_SUFFIXES

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


def discover_universe(queries: Iterable[str], quotes_per_query: int) -> List[str]:
    tickers_seen, ticker_list = set(), []
    for query in queries:
        try:
            quotes = yahoo_search(query, quotes_count=quotes_per_query)
            for q in quotes:
                ticker = q.get("symbol")
                if ticker and ticker not in tickers_seen and _is_valid_etf_quote(q):
                    tickers_seen.add(ticker)
                    ticker_list.append(ticker)
        except Exception as e:
            logger.warning(f"Errore ricerca per '{query}': {e}")
    return ticker_list


def get_unique_preferred_tickers(tickers: List[str]) -> List[str]:
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