# -*- coding: utf-8 -*-
import re
import requests
from .config import REQUEST_HEADERS
from typing import Optional

def fetch_tracking_difference(isin: str) -> Optional[float]:
    """
    Recupera la tracking difference da trackingdifferences.com
    """
    try:
        url = f"https://www.trackingdifferences.com/ETF/ISIN/{isin}"
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        r.raise_for_status()
        html = r.text

        # Cerca il valore di tracking difference
        patterns = [
            r"Tracking Difference[^<]{0,100}?(-?\d+(?:[.,]\d+)?)\s*%",
            r"TD[^<]{0,50}?(-?\d+(?:[.,]\d+)?)\s*%",
            r"difference[^<]{0,50}?(-?\d+(?:[.,]\d+)?)\s*%"
        ]

        for pattern in patterns:
            match = re.search(pattern, html, flags=re.IGNORECASE)
            if match:
                return float(match.group(1).replace(",", "."))

    except Exception as e:
        print(f"Errore nel recupero tracking difference per {isin}: {e}")

    return None