# -*- coding: utf-8 -*-
import logging
import re
from typing import Optional

from .config import REQUEST_HEADERS
from .base_client import BaseFinancialClient, DataValidator

logger = logging.getLogger(__name__)


class TrackingDifferencesClient(BaseFinancialClient):
    """
    Client per trackingdifferences.com che riusa la BaseFinancialClient
    per sessione, timeout e retry.
    """

    def __init__(self):
        super().__init__("TrackingDifferences", base_timeout=10)
        self.base_url = "https://www.trackingdifferences.com"
        # Compilo i pattern una sola volta (semplici e robusti)
        self._patterns = [
            re.compile(r"Tracking\s*Difference[^\<]{0,200}?(-?\d+(?:[.,]\d+)?)\s*%", re.IGNORECASE),
            re.compile(r"\bTD\b[^\<]{0,120}?(-?\d+(?:[.,]\d+)?)\s*%", re.IGNORECASE),
            re.compile(r"difference[^\<]{0,120}?(-?\d+(?:[.,]\d+)?)\s*%", re.IGNORECASE),
        ]

    def validate_input(self, isin: str) -> bool:
        return DataValidator.validate_isin(isin)

    def _parse_value(self, html: str) -> Optional[float]:
        """
        Estrae il valore percentuale dai pattern.
        Restituisce punti percentuali (es. 0.50 per 0,50%).
        """
        if not html:
            return None
        for pat in self._patterns:
            m = pat.search(html)
            if m:
                raw = m.group(1).strip().replace(",", ".")
                try:
                    val = float(raw)
                    # Sanity check: TD ragionevoli in (-50%, +50%)
                    if -50.0 <= val <= 50.0:
                        return val
                    logger.debug(f"Valore TD fuori range plausibile: {val}")
                except ValueError:
                    continue
        return None

    def fetch_tracking_difference(self, isin: str) -> Optional[float]:
        """
        Recupera la tracking difference per un ISIN da trackingdifferences.com.
        Ritorna il valore in punti percentuali (es. -0.42 per -0,42%).
        """
        if not self.validate_input(isin):
            logger.error(f"ISIN non valido: {isin!r}")
            return None
        url = f"{self.base_url}/ETF/ISIN/{isin}"
        resp = self.safe_request(url, headers=REQUEST_HEADERS, timeout=10)
        if not resp:
            return None
        value = self._parse_value(resp.text)
        if value is None:
            logger.info(f"Nessuna tracking difference trovata per {isin}")
        return value


# ---- Istanza singleton + wrapper per retro‑compatibilità ----
_td_client = TrackingDifferencesClient()


def fetch_tracking_difference(isin: str) -> Optional[float]:
    """
    Wrapper retro-compatibile: stessa firma della versione precedente.
    """
    return _td_client.fetch_tracking_difference(isin)