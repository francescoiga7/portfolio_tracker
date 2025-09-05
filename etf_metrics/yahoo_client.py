# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional
import requests
import pandas as pd
import yfinance as yf
from .config import REQUEST_HEADERS, MANUAL_ISIN_MAP
from .utils import pick_preferred_symbol

logger = logging.getLogger(__name__)

def yahoo_search(query: str, quotes_count: int = 40) -> List[Dict]:
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": quotes_count, "newsCount": 0, "listsCount": 0}
    try:
        resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=10)
        if resp.status_code == 429:
            resp = requests.get(url.replace("query2", "query1"), params=params, headers=REQUEST_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("quotes", []) or []
    except Exception:
        return []


def get_info_fallback(isin: str) -> Dict:
    """Recupera informazioni di fallback per ISIN sconosciuti"""
    try:
        # Prova a cercare l'ISIN su Yahoo
        quotes = yahoo_search(isin, quotes_count=5)

        if quotes:
            # Prendi il primo risultato
            quote = quotes[0]
            ticker = quote.get("symbol")

            if ticker:
                yf_info = yf.Ticker(ticker).info
                return {
                    "longName": quote.get("longname", quote.get("shortname", isin)),
                    "shortName": quote.get("shortname", isin),
                    "fundFamily": yf_info.get("fundFamily", "Sconosciuto"),
                    "category": yf_info.get("category", "Sconosciuto")
                }

    except Exception as e:
        print(f"Errore nel fallback per {isin}: {e}")

    return {
        "longName": isin,
        "shortName": isin,
        "fundFamily": "Sconosciuto",
        "category": "Sconosciuto"
    }


def resolve_isin_one(isin: str) -> Optional[str]:
    if isin in MANUAL_ISIN_MAP:
        return pick_preferred_symbol(MANUAL_ISIN_MAP[isin])
    quotes = yahoo_search(isin, quotes_count=60)
    symbols = [q.get("symbol") for q in quotes if q.get("symbol")]
    return pick_preferred_symbol(symbols)


def get_series(ticker: str, period: str) -> Optional[pd.Series]:
    """
    Recupera serie storica da Yahoo Finance con gestione errori robusta.

    Args:
        ticker: Simbolo del ticker
        period: Periodo (es. '1y', '5y', 'max')

    Returns:
        Serie storica normalizzata o None se errore
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(period=period, auto_adjust=False)

        if df.empty:
            logger.warning(f"Nessun dato storico per {ticker} nel periodo {period}")
            return None

        # Scegli la colonna migliore per i prezzi
        col = "Adj Close" if "Adj Close" in df.columns and not df["Adj Close"].isna().all() else "Close"
        s = df[col].dropna().copy()

        if s.empty:
            logger.warning(f"Serie vuota dopo pulizia per {ticker}")
            return None

        # Normalizza timezone per evitare conflitti
        if hasattr(s.index, 'tz') and s.index.tz is not None:
            s.index = s.index.tz_localize(None)

        # Ordina per data e assegna nome
        s = s.sort_index()
        s.name = ticker

        # Validazione aggiuntiva
        if len(s) < 2:
            logger.warning(f"Dati insufficienti per {ticker}: {len(s)} osservazioni")
            return None

        # Controlla valori realistici
        if (s <= 0).any():
            logger.warning(f"Valori non validi rilevati per {ticker}")
            # Rimuovi valori <= 0 invece di scartare tutta la serie
            s = s[s > 0]
            if s.empty:
                return None

        # Controlla movimenti estremi (possibili split non aggiustati)
        returns = s.pct_change().dropna()
        extreme_moves = returns[abs(returns) > 0.5]  # Movimenti > 50%
        if len(extreme_moves) > 0:
            logger.info(f"Movimenti estremi rilevati per {ticker}: {len(extreme_moves)} giorni")

        return s

    except Exception as e:
        logger.error(f"Errore nel recupero dati per {ticker}: {e}")
        return None


def get_info(isin: Optional[str] = None, ticker: Optional[str] = None) -> Dict:
    """
    Recupera le informazioni di base (come fundFamily, longName, etc.)
    di un ETF da Yahoo Finance, usando ISIN o ticker.
    """
    if not isin and not ticker:
        return {}

    # Se hai l'ISIN, prova a risolverlo prima
    if isin:
        t = resolve_isin_one(isin)
        if t:
            ticker = t

    if ticker:
        try:
            return yf.Ticker(ticker).info
        except Exception:
            return {}
    return {}