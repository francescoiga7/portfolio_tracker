# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from .config import REQUEST_HEADERS, MANUAL_ISIN_MAP
from .utils import pick_preferred_symbol
from .base_client import BaseFinancialClient

logger = logging.getLogger(__name__)

UNKNOWN = "Sconosciuto"
BASE_INFO = {"fundFamily": UNKNOWN, "category": UNKNOWN}


class YahooClient(BaseFinancialClient):
    """Client Yahoo Finance che riusa la sessione con retry della BaseFinancialClient."""

    def __init__(self):
        super().__init__("YahooFinance", base_timeout=12)
        self._search_hosts = ("query2", "query1")  # fallback sequence

    def validate_input(self, input_data: str) -> bool:
        """Accetta qualsiasi stringa non vuota (ticker, query o ISIN)."""
        return isinstance(input_data, str) and input_data.strip() != ""

    # ---- SEARCH ----
    def _search_once(self, host: str, query: str, quotes_count: int) -> List[Dict]:
        url = f"https://{host}.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": quotes_count, "newsCount": 0, "listsCount": 0}
        resp = self.safe_request(url, headers=REQUEST_HEADERS, params=params, timeout=10)
        if not resp:
            return []
        try:
            data = resp.json() or {}
        except Exception:
            return []
        return data.get("quotes", []) or []

    def search(self, query: str, quotes_count: int = 40) -> List[Dict]:
        if not self.validate_input(query):
            return []
        for host in self._search_hosts:
            quotes = self._search_once(host, query, quotes_count)
            if quotes:
                return quotes
        return []

    # ---- ISIN → Ticker ----
    def resolve_isin_one(self, isin: str) -> Optional[str]:
        if isin in MANUAL_ISIN_MAP:
            return pick_preferred_symbol(MANUAL_ISIN_MAP[isin])
        quotes = self.search(isin, quotes_count=60)
        symbols = [q.get("symbol") for q in quotes if q.get("symbol")]
        return pick_preferred_symbol(symbols)

    # ---- Serie storiche ----
    def get_series(self, ticker: str, period: str) -> Optional[pd.Series]:
        """
        Recupera la serie storica da Yahoo Finance usando yfinance.
        Restituisce una pd.Series con indice datetime e nome = ticker.
        """
        if not self.validate_input(ticker):
            return None
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
            if df.empty:
                logger.warning(f"Nessun dato storico per {ticker} nel periodo {period}")
                return None
            col = "Adj Close" if ("Adj Close" in df.columns and not df["Adj Close"].isna().all()) else "Close"
            s = df[col].dropna().copy()
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s = s.sort_index()
            s.name = ticker
            if len(s) < 2:
                logger.warning(f"Dati insufficienti per {ticker}: {len(s)} osservazioni")
                return None
            if (s <= 0).any():
                s = s[s > 0]
            if s.empty:
                return None
            # Log informativo su movimenti estremi (es. split non aggiustati)
            rets = s.pct_change().dropna()
            if (rets.abs() > 0.5).any():
                logger.info(f"Movimenti estremi rilevati per {ticker}")
            return s
        except Exception as e:
            logger.error(f"Errore nel recupero dati per {ticker}: {e}")
            return None

    # ---- Info ----
    def get_info_fallback(self, isin: str) -> Dict:
        """
        Informazioni minime quando l'ISIN è poco coperto:
        prova a risolvere un ticker e poi usa yfinance.info.
        """
        try:
            quotes = self.search(isin, quotes_count=5)
            if not quotes:
                return {**BASE_INFO, "longName": isin, "shortName": isin}
            ticker = quotes[0].get("symbol")
            if not ticker:
                return {**BASE_INFO, "longName": isin, "shortName": isin}
            info = yf.Ticker(ticker).info
            return {
                "longName": quotes[0].get("longname", quotes[0].get("shortname", isin)),
                "shortName": quotes[0].get("shortname", isin),
                "fundFamily": info.get("fundFamily", UNKNOWN),
                "category": info.get("category", UNKNOWN),
            }
        except Exception as e:
            logger.debug(f"Fallback info fallito per {isin}: {e}")
            return {**BASE_INFO, "longName": isin, "shortName": isin}

    def get_info(self, isin: Optional[str] = None, ticker: Optional[str] = None) -> Dict:
        if not isin and not ticker:
            return {}
        if isin and not ticker:
            t = self.resolve_isin_one(isin)
            if t:
                ticker = t
        if not ticker:
            return {}
        try:
            return yf.Ticker(ticker).info
        except Exception:
            return {}


# ---- Istanza singleton + wrapper per retro-compatibilità ----
_yahoo_client = YahooClient()


def yahoo_search(query: str, quotes_count: int = 40) -> List[Dict]:
    return _yahoo_client.search(query, quotes_count)


def resolve_isin_one(isin: str) -> Optional[str]:
    return _yahoo_client.resolve_isin_one(isin)


def get_series(ticker: str, period: str) -> Optional[pd.Series]:
    return _yahoo_client.get_series(ticker, period)


def get_info(isin: Optional[str] = None, ticker: Optional[str] = None) -> Dict:
    return _yahoo_client.get_info(isin=isin, ticker=ticker)


def get_info_fallback(isin: str) -> Dict:
    return _yahoo_client.get_info_fallback(isin)