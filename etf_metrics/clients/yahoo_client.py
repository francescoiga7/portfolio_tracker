# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from etf_metrics.shared.config import REQUEST_HEADERS, MANUAL_ISIN_MAP
from etf_metrics.shared.utils import pick_preferred_symbol
from .base_client import BaseFinancialClient

logger = logging.getLogger(__name__)

UNKNOWN = "Sconosciuto"
BASE_INFO = {"fundFamily": UNKNOWN, "category": UNKNOWN}


class YahooClient(BaseFinancialClient):
    """Client Yahoo Finance che riusa la sessione con retry della BaseFinancialClient."""

    def __init__(self):
        super().__init__("YahooFinance", base_timeout=12)
        self._search_hosts = ("query2", "query1")

    def validate_input(self, input_data: str) -> bool:
        """Accetta qualsiasi stringa non vuota (ticker, query o ISIN)."""
        return isinstance(input_data, str) and input_data.strip() != ""

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

    def search(self, query: str, quotes_count: int = 100) -> List[Dict]:
        if not self.validate_input(query):
            return []
        for host in self._search_hosts:
            quotes = self._search_once(host, query, quotes_count)
            if quotes:
                return quotes
        return []

    def resolve_isin_one(self, isin: str) -> Optional[str]:
        if isin in MANUAL_ISIN_MAP:
            return pick_preferred_symbol(MANUAL_ISIN_MAP[isin])
        quotes = self.search(isin, quotes_count=60)
        symbols = [q.get("symbol") for q in quotes if q.get("symbol")]
        return pick_preferred_symbol(symbols)

    def resolve_ticker_to_isin(self, ticker: str) -> Optional[str]:
        """Tenta di trovare l'ISIN per un dato ticker."""
        try:
            info = self.get_info(ticker=ticker)
            if info and isinstance(info.get('isin'), str):
                return info['isin']
        except Exception:
            pass
        return None

    def get_series(self, ticker: str, period: str, as_dataframe: bool = False) -> Optional[
        Union[pd.Series, pd.DataFrame]]:
        """
        Recupera la serie storica da Yahoo Finance.
        Restituisce una pd.Series (default) o un pd.DataFrame se as_dataframe=True.
        """
        if not self.validate_input(ticker):
            return None
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
            if df.empty:
                logger.warning(f"Nessun dato storico per {ticker} nel periodo {period}")
                return None

            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()

            if as_dataframe:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    return None
                return df[required_cols]

            col = "Adj Close" if ("Adj Close" in df.columns and not df["Adj Close"].isna().all()) else "Close"
            s = df[col].dropna().copy()
            s.name = ticker
            if len(s) < 2:
                return None
            if (s <= 0).any():
                s = s[s > 0]
            if s.empty:
                return None
            return s
        except Exception as e:
            logger.error(f"Errore nel recupero dati per {ticker}: {e}")
            return None

    def get_info(self, isin: Optional[str] = None, ticker: Optional[str] = None) -> Dict:
        if not isin and not ticker:
            return {}

        resolved_ticker = ticker
        if isin and not resolved_ticker:
            resolved_ticker = self.resolve_isin_one(isin)

        if not resolved_ticker:
            return {}

        try:
            info = yf.Ticker(resolved_ticker).info
            if not isin and 'isin' in info and isinstance(info['isin'], str):
                pass
            elif isin:
                info['isin'] = isin
            return info
        except Exception:
            return {}


_YC: Optional[YahooClient] = None


def _get_yahoo_client() -> YahooClient:
    global _YC
    if _YC is None:
        _YC = YahooClient()
    return _YC


def yahoo_search(query: str, quotes_count: int = 100) -> List[Dict]:
    return _get_yahoo_client().search(query, quotes_count)


def resolve_isin_one(isin: str) -> Optional[str]:
    return _get_yahoo_client().resolve_isin_one(isin)


def resolve_ticker_to_isin(ticker: str) -> Optional[str]:
    return _get_yahoo_client().resolve_ticker_to_isin(ticker)


def get_series(ticker: str, period: str, as_dataframe: bool = False) -> Optional[Union[pd.Series, pd.DataFrame]]:
    return _get_yahoo_client().get_series(ticker, period, as_dataframe=as_dataframe)


def get_info(isin: Optional[str] = None, ticker: Optional[str] = None) -> Dict:
    return _get_yahoo_client().get_info(isin=isin, ticker=ticker)