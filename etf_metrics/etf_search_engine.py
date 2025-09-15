# -*- coding: utf-8 -*-
"""
ETF Search Engine Definitivo - Robusto e Tollerante

Logica Operativa:
1.  **Scoperta Massimale**: Usa query ampie per scoprire il maggior numero possibile
    di ETF/ETC/ETN sui mercati europei.
2.  **Deduplicazione Flessibile**: Raggruppa per ISIN solo quando disponibile,
    privilegiando sempre Borsa Italiana (.MI). Gli strumenti senza ISIN vengono
    comunque mantenuti.
3.  **Calcolo "Best-Effort"**: Tenta di calcolare tutte le metriche per ogni strumento.
    Se un calcolo fallisce (es. Sharpe Ratio), il dato viene lasciato vuoto ma
    lo strumento NON viene scartato.
4.  **Classifica Informativa**: Ordina i risultati per Sharpe Ratio decrescente,
    posizionando in fondo gli strumenti con dati incompleti. Questo garantisce
    che l'utente veda sempre l'intero universo scoperto.
"""
import logging
from typing import Dict, List, Optional, Tuple, Iterable
from functools import lru_cache
from collections import defaultdict
import pandas as pd

from .yahoo_client import yahoo_search, get_series, get_info
from .metrics import compute_metrics_from_series, compute_sharpe_ratio
from .utils import pick_preferred_symbol

logger = logging.getLogger(__name__)

# --- Configurazioni per la Ricerca ---

DEFAULT_SEED_QUERIES: Tuple[str, ...] = (
    # Azionari ad ampia copertura
    "MSCI World UCITS ETF", "FTSE All-World UCITS ETF", "Global Equity UCITS ETF",
    "S&P 500 UCITS ETF", "NASDAQ 100 UCITS ETF", "STOXX Europe 600 UCITS ETF",
    "MSCI Emerging Markets UCITS ETF", "Japan UCITS ETF",
    # Tematici e Settoriali
    "Technology Sector UCITS ETF", "Healthcare Sector UCITS ETF", "Financial Sector UCITS ETF",
    "Clean Energy UCITS ETF", "AI & Robotics UCITS ETF", "Cybersecurity UCITS ETF",
    # Fattoriali (Smart Beta)
    "Value Factor UCITS ETF", "Growth Factor UCITS ETF", "Momentum Factor UCITS ETF", "Quality Factor UCITS ETF",
    # Obbligazionari
    "Global Aggregate Bond UCITS ETF EUR", "Government Bond UCITS ETF EUR", "Corporate Bond UCITS ETF EUR",
    "High Yield Bond UCITS ETF EUR", "Inflation-Linked Bond UCITS ETF",
    # Materie Prime e Crypto
    "Gold ETC", "Silver ETC", "Broad Commodities ETC", "Bitcoin ETP", "Ethereum ETP"
)

PERIOD_MAP = {
    "1d": (1 / 30.4, "1mo"), "1m": (1, "3mo"), "3m": (3, "6mo"),
    "6m": (6, "1y"), "1y": (12, "3y"), "3y": (36, "5y"), "5y": (60, "10y"),
}


# --- Funzioni Helper ---

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


# --- Pipeline di Ricerca ---

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

    preferred_tickers = []
    for isin, candidates in isin_to_tickers.items():
        preferred = pick_preferred_symbol(candidates)
        if preferred: preferred_tickers.append(preferred)

    # Aggiunge i ticker per cui non è stato trovato un ISIN, trattandoli come unici
    preferred_tickers.extend(tickers_without_isin)
    return list(dict.fromkeys(preferred_tickers))  # Rimuove eventuali duplicati finali


def _calculate_metrics_for_ticker(ticker: str, lookback_months: float, fetch_period: str, rf_ann_pct: float) -> Dict:
    # Inizializza il dizionario dei risultati per garantire che venga sempre restituito qualcosa
    info = get_info(ticker=ticker) or {}
    result = {
        "isin": _get_isin_for_ticker(ticker),
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "cagr_pct": None,
        "mdd_pct": None,
        "sharpe_ratio": None,
    }

    series = get_series(ticker, period=fetch_period)
    if series is None or len(series) < 2:
        return result

    end_date = series.index.max()
    start_date = end_date - pd.DateOffset(days=int(lookback_months * 30.4))
    series_filtered = series.loc[start_date:end_date]

    if len(series_filtered) < 10:
        return result

    metrics = compute_metrics_from_series(series_filtered) or {}
    sharpe = compute_sharpe_ratio(series_filtered, rf_ann_pct)

    result.update({
        "cagr_pct": metrics.get("cagr"),
        "mdd_pct": metrics.get("mdd"),
        "sharpe_ratio": sharpe,
    })
    return result


# --- API Pubblica ---

def find_top_performers(period: str = "1y", risk_free_rate_pct: float = 3.0, quotes_per_query: int = 150,
                        discovery_limit: int = 1500) -> pd.DataFrame:
    if period not in PERIOD_MAP:
        raise ValueError(f"Periodo '{period}' non supportato. Validi: {list(PERIOD_MAP.keys())}")

    lookback_months, fetch_period = PERIOD_MAP[period]

    raw_tickers = _discover_universe(DEFAULT_SEED_QUERIES, quotes_per_query, discovery_limit)
    unique_tickers = _get_unique_preferred_tickers(raw_tickers)

    results = []
    for ticker in unique_tickers:
        try:
            metrics = _calculate_metrics_for_ticker(ticker, lookback_months, fetch_period, risk_free_rate_pct)
            if metrics: results.append(metrics)
        except Exception as e:
            logger.error(f"Errore calcolo metriche per {ticker}: {e}")

    if not results: return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(by="sharpe_ratio", ascending=False, na_position='last').reset_index(drop=True)

    return df