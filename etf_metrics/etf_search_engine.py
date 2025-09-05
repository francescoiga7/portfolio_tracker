# -*- coding: utf-8 -*-
"""
ETF Search Engine (Universale • UCITS) — metrics come in momentum.py
--------------------------------------------------------------------
- Discovery universale (senza query utente) di ETF **UCITS** tramite Yahoo Finance.
- Calcolo metriche nello **stesso stile di momentum.py**:
  * scarica serie ampia (period="3y")
  * filtra per lookback in mesi
  * compute_metrics_from_series(series_filtrata) -> dict con 'total','cagr','vol_ann','mdd' (in %)
  * compute_sharpe_ratio(series_filtrata, rf_annual_pct)

Output DataFrame:
['ticker', 'isin', 'nome etf', 'rendimento cagr', 'max drawdown', 'sharpe ratio']

Progettato per Streamlit e richiamato da UI.py. Nessun codice CLI.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Iterable
from functools import lru_cache

import pandas as pd
import streamlit as st

from .yahoo_client import yahoo_search, get_series, get_info
from .metrics import compute_metrics_from_series, compute_sharpe_ratio

# MANUAL_ISIN_MAP è opzionale (per risolvere ISIN se vuoi forzare alcuni mapping)
try:
    from .config import MANUAL_ISIN_MAP
except Exception:
    MANUAL_ISIN_MAP = {}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
#                       Periodi: mapping -> mesi
# ---------------------------------------------------------------------

_PERIOD_CHOICES = ("1m", "3m", "6m", "1y", "3y", "5y", "10y", "ytd", "max")

def _normalize_period_key(period: Optional[str]) -> str:
    """Valida il periodo richiesto (default: '1y')."""
    if not period or not isinstance(period, str):
        return "1y"
    p = period.strip().lower()
    return p if p in _PERIOD_CHOICES else "1y"

def _period_to_lookback_months(period_key: str) -> int:
    """Converte il periodo selezionato in mesi per l’estrazione e slicing (stile momentum.py)."""
    from datetime import date
    today = pd.to_datetime("today").date()

    mapping = {
        "1m": 1,
        "3m": 3,
        "6m": 6,
        "1y": 12,
        "3y": 36,
        "5y": 60,
        "10y": 120,
    }
    if period_key in mapping:
        return mapping[period_key]

    if period_key == "ytd":
        jan1 = date(today.year, 1, 1)
        days = (today - jan1).days
        months = max(1, int(round(days / 30.4)))
        return months

    if period_key == "max":
        # momentum.py scarica 3y internamente; segnaliamo che oltre 36 mesi la copertura effettiva rimane ~3y
        logger.warning("Periodo 'max' richiesto ma il fetch resta '3y' (stile momentum.py): la finestra effettiva è ~3 anni.")
        return 120

    # fallback
    return 12

# ---------------------------------------------------------------------
#                        Helper ETF / ISIN / Nome
# ---------------------------------------------------------------------

def _is_etf_quote(q: Dict) -> bool:
    qt = (q.get("quoteType") or q.get("type") or q.get("typeDisp") or "").upper()
    nm = (q.get("longname") or q.get("shortname") or q.get("name") or "").upper()
    return "ETF" in qt or "ETF" in nm
def _is_ucits_quote(q: Dict) -> bool:
    nm = (q.get("longname") or q.get("shortname") or q.get("name") or "")
    return "UCITS" in nm.upper()

def _extract_name(info: Optional[Dict], fallback: str) -> str:
    candidates = []
    if isinstance(info, dict):
        candidates += [info.get("longName"), info.get("shortName")]
    candidates.append(fallback)
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return fallback

def _invert_manual_isin_map() -> Dict[str, str]:
    inverted = {}
    try:
        for isin, syms in (MANUAL_ISIN_MAP or {}).items():
            if isinstance(syms, (list, tuple, set)):
                for s in syms:
                    if s:
                        inverted[str(s).upper()] = isin
            elif syms:
                inverted[str(syms).upper()] = isin
    except Exception:
        pass
    return inverted

@lru_cache(maxsize=2048)
def _get_isin_for_ticker(ticker: str) -> Optional[str]:
    """Ricava ISIN da inversione MANUAL_ISIN_MAP o da get_info() (chiavi contenenti 'isin')."""
    if not ticker:
        return None
    inv = _invert_manual_isin_map()
    if ticker.upper() in inv:
        return inv[ticker.upper()]
    try:
        info = get_info(ticker=ticker) or {}
        for k, v in (info or {}).items():
            if isinstance(k, str) and "isin" in k.lower() and isinstance(v, str) and len(v) >= 10:
                return v.strip()
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------
#                     Discovery universo ETF (UCITS only)
# ---------------------------------------------------------------------

DEFAULT_SEED_QUERIES: Tuple[str, ...] = (
    # Generiche + UCITS
    "ETF UCITS", "UCITS", "INDEX ETF UCITS", "ACCUMULATING UCITS", "DISTRIBUTING UCITS",
    # Azionario
    "MSCI UCITS", "S&P UCITS", "NASDAQ UCITS", "FTSE UCITS", "STOXX UCITS",
    "SMALL CAP UCITS", "VALUE UCITS", "GROWTH UCITS", "ESG UCITS", "SRI UCITS",
    # Geografici
    "WORLD UCITS", "EUROPE UCITS", "EMERGING UCITS", "ASIA UCITS", "JAPAN UCITS", "CHINA UCITS", "USA UCITS",
    # Obbligazionario
    "BOND UCITS", "TREASURY UCITS", "CORPORATE UCITS", "HIGH YIELD UCITS", "INFLATION UCITS", "TIPS UCITS",
    # Altri
    "COMMODITY UCITS", "GOLD UCITS", "REIT UCITS", "CLEAN ENERGY UCITS",
)

@st.cache_data(show_spinner=False, ttl=60 * 30)
def _discover_ucits_tickers(
    seed_queries: Optional[Iterable[str]] = None,
    quotes_per_query: int = 100,
    limit_universe: int = 600,
) -> List[str]:
    """Scopre un insieme ampio di ticker ETF **UCITS** (deduplicati)."""
    seed = list(seed_queries) if seed_queries is not None else list(DEFAULT_SEED_QUERIES)
    tickers: List[str] = []
    seen: set = set()

    for q in seed:
        try:
            quotes = yahoo_search(q, quotes_count=quotes_per_query) or []
        except Exception as e:
            logger.warning(f"Errore ricerca seed '{q}': {e}")
            quotes = []

        for quote in quotes:
            if not _is_etf_quote(quote):
                continue
            if not _is_ucits_quote(quote):
                continue

            t = quote.get("symbol")
            if not t or not isinstance(t, str) or t in seen:
                continue

            seen.add(t)
            tickers.append(t)

            if len(tickers) >= limit_universe:
                return tickers

    return tickers

# ---------------------------------------------------------------------
#                       Calcolo metriche (stile momentum.py)
# ---------------------------------------------------------------------

def _compute_row_for_ticker_like_momentum(
    ticker: str,
    lookback_months: int,
    rf_ann_pct: float,
) -> Optional[Dict[str, Optional[float]]]:
    """
    Replica la logica di momentum.py per un singolo ticker:
    - scarica serie ampia (period='3y')
    - filtra per la finestra [end - lookback_months : end]
    - calcola metrics con compute_metrics_from_series + compute_sharpe_ratio
    """
    end_date = pd.to_datetime("today").normalize()
    start_date = end_date - pd.DateOffset(months=lookback_months)

    # Scarica uno storico sufficiente per i calcoli (stile momentum.py)
    series = get_series(ticker, period="3y")
    if series is None or series.empty:
        return None

    series_filtered = series.loc[start_date:end_date]
    # momentum.py richiede almeno 21 osservazioni (≈ un mese)
    if series_filtered.shape[0] < 21:
        return None

    # Calcola le metriche
    try:
        metrics = compute_metrics_from_series(series_filtered) or {}
    except Exception:
        metrics = {}

    try:
        sharpe = compute_sharpe_ratio(series_filtered, rf_ann_pct)
    except Exception:
        sharpe = None

    # Nome + UCITS enforcement addizionale
    try:
        info = get_info(ticker=ticker) or {}
    except Exception:
        info = {}
    name = _extract_name(info, fallback=ticker)
    nm_upper = (info.get("longName") or info.get("shortName") or name or "").upper()
    if "UCITS" not in nm_upper:
        # Discovery è già UCITS; se qui non appare, filtriamo lo stesso per essere coerenti
        return None

    # ISIN “se possibile” (non necessario per le metriche)
    isin = _get_isin_for_ticker(ticker)

    return {
        "ticker": ticker,
        "isin": isin,
        "nome etf": name,
        "rendimento cagr": metrics.get("cagr"),  # % (non frazione)
        "max drawdown": metrics.get("mdd"),      # % (non frazione)
        "sharpe ratio": sharpe,                  # adimensionale
    }

# ---------------------------------------------------------------------
#                           Public API (UI.py)
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=True, ttl=60 * 30)
def search_etfs_universal(
    period: str = "1y",               # ✅ default 1 anno
    risk_free_rate_pct: float = 0.0,  # ✅ RF in % annua (es. 4.0 = 4.0%)
    quotes_per_query: int = 100,
    limit_universe: int = 600,
    max_results: int = 150,
    sort_by: str = "sharpe ratio",    # 'sharpe ratio' | 'rendimento cagr' | 'max drawdown' | 'ticker'
    ascending: bool = False,
    seed_queries: Optional[Iterable[str]] = None,  # opzionale override
) -> pd.DataFrame:
    """
    Ricerca universale **solo UCITS**:
    - Discovery via seed queries (UCITS-only) → TICKER
    - Calcolo metriche per ogni ticker con logica **identica** a momentum.py
      (serie '3y' → slicing per lookback mesi → compute_metrics_from_series / compute_sharpe_ratio)
    - Output tabellare ordinato.

    Ritorna DataFrame con colonne:
    ['ticker', 'isin', 'nome etf', 'rendimento cagr', 'max drawdown', 'sharpe ratio']
    Dove 'rendimento cagr' e 'max drawdown' sono espresse in **percentuale** (non frazioni decimali).
    """
    pkey = _normalize_period_key(period)
    lookback_months = _period_to_lookback_months(pkey)

    # 1) Discovery UCITS (ticker)
    universe = _discover_ucits_tickers(
        seed_queries=seed_queries,
        quotes_per_query=quotes_per_query,
        limit_universe=limit_universe,
    )
    if not universe:
        return pd.DataFrame(columns=["ticker", "isin", "nome etf", "rendimento cagr", "max drawdown", "sharpe ratio"])

    # 2) Taglia il numero da analizzare per performance
    tickers = universe[:max_results]

    # 3) Calcolo metriche per ogni ticker (stile momentum.py)
    rows: List[Dict] = []
    for t in tickers:
        try:
            row = _compute_row_for_ticker_like_momentum(
                ticker=t,
                lookback_months=lookback_months,
                rf_ann_pct=risk_free_rate_pct,
            )
            if row is not None:
                rows.append(row)
        except Exception as e:
            logger.warning(f"Errore elaborando {t}: {e}")

    if not rows:
        return pd.DataFrame(columns=["ticker", "isin", "nome etf", "rendimento cagr", "max drawdown", "sharpe ratio"])

    df = pd.DataFrame(rows)

    # 4) Ordinamento
    valid_sort_cols = {"ticker", "isin", "nome etf", "rendimento cagr", "max drawdown", "sharpe ratio"}
    sort_key = sort_by if sort_by in valid_sort_cols else "sharpe ratio"
    df = df.sort_values(by=[sort_key], ascending=ascending, na_position="last").reset_index(drop=True)

    return df


def format_results_for_display(
    df: pd.DataFrame,
    pct_cols: Tuple[str, ...] = ("rendimento cagr", "max drawdown"),
    round_cols: Tuple[str, ...] = ("sharpe ratio",),
    pct_decimals: int = 2,
    num_decimals: int = 2,
) -> pd.DataFrame:
    """
    Formattazione per UI (percentuali già in scala % → NON moltiplicare per 100).
    Mantieni df originale per export numerico.
    """
    if df is None or df.empty:
        return df

    df_fmt = df.copy()

    for col in pct_cols:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.{pct_decimals}f}%" if pd.notna(x) else "")

    for col in round_cols:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.{num_decimals}f}" if pd.notna(x) else "")

    return df_fmt
