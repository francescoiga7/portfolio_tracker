# -*- coding: utf-8 -*-
import re
import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from .config import REQUEST_HEADERS
from .base_client import BaseFinancialClient, DataValidator
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class JustETFClient(BaseFinancialClient):
    """Client robusto per JustETF con gestione errori avanzata e cache integrata."""

    def __init__(self):
        super().__init__("JustETF", base_timeout=15)
        self.base_url = "https://www.justetf.com"
        self.supported_langs = ["it", "en", "de", "fr", "es"]
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)

    def validate_input(self, isin: str) -> bool:
        """Valida l'ISIN fornito con controlli avanzati."""
        if not DataValidator.validate_isin(isin):
            return False

        # Controlla se l'ISIN è nella blacklist (es. fondi non supportati)
        blacklist_prefixes = ["LU", "CH"]  # Aggiungi prefissi problematici se necessario
        if any(isin.startswith(prefix) for prefix in blacklist_prefixes):
            logger.warning(f"ISIN {isin} potrebbe non essere supportato completamente")

        return True

    def _get_from_cache(self, isin: str) -> Optional[str]:
        """Recupera dati dalla cache se validi."""
        if isin in self.cache:
            cached_data, timestamp = self.cache[isin]
            if datetime.now() - timestamp < self.cache_ttl:
                logger.debug(f"Dati JustETF recuperati dalla cache per {isin}")
                return cached_data
            else:
                del self.cache[isin]
        return None

    def _cache_data(self, isin: str, data: str) -> None:
        """Salva dati nella cache."""
        self.cache[isin] = (data, datetime.now())

    def fetch_etf_page(self, isin: str) -> Optional[str]:
        """Recupera la pagina JustETF per un ISIN con retry intelligente e cache."""
        if not self.validate_input(isin):
            logger.error(f"ISIN non valido: {isin}")
            return None

        # Controlla cache prima
        cached_data = self._get_from_cache(isin)
        if cached_data:
            return cached_data

        def fetch_operation():
            for lang in self.supported_langs:
                url = f"{self.base_url}/{lang}/etf-profile.html?isin={isin}"

                response = self.safe_request(url, REQUEST_HEADERS)
                if response and response.text.strip():
                    # Verifica che la pagina contenga dati reali
                    if self._validate_page_content(response.text):
                        logger.info(f"Dati JustETF recuperati per {isin} in {lang}")
                        self._cache_data(isin, response.text)
                        return response.text
                    else:
                        logger.warning(f"Pagina vuota o non valida per {isin} in {lang}")

            raise Exception(f"Nessuna pagina valida trovata per ISIN {isin}")

        try:
            return self.retry_operation(fetch_operation, max_retries=2)
        except Exception:
            logger.warning(f"Impossibile recuperare dati JustETF per {isin}")
            return None

    def _validate_page_content(self, html: str) -> bool:
        """Valida che la pagina HTML contenga dati ETF reali."""
        if not html or len(html) < 1000:
            return False

        # Controlla presenza di elementi chiave
        required_elements = ["vallabel", "val", "etf", "fund"]
        return any(element in html.lower() for element in required_elements)

    def clear_cache(self) -> None:
        """Pulisce la cache."""
        self.cache.clear()
        logger.info("Cache JustETF pulita")

    def get_cache_info(self) -> Dict[str, int]:
        """Restituisce informazioni sulla cache."""
        valid_entries = sum(1 for _, (_, timestamp) in self.cache.items()
                          if datetime.now() - timestamp < self.cache_ttl)
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries
        }


# Istanza globale del client
_justetf_client = JustETFClient()

def fetch_justetf_page(isin: str, timeout: int = 12) -> Optional[str]:
    """
    Fetches the JustETF page for a given ISIN in multiple languages.
    Backward compatibility wrapper for the new client.
    """
    return _justetf_client.fetch_etf_page(isin)

def _strip_tags(html: str) -> str:
    """
    Strips HTML tags and normalizes whitespace.
    """
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- NEW: robust label-value extraction with BeautifulSoup ---
def _find_value_by_labels_soup(soup: BeautifulSoup, labels: List[str]) -> Optional[str]:
    """
    Cerca un valore su JustETF. Per il benchmark, cerca prima un link per ottenere un nome pulito.
    """
    label_norm = [l.strip().lower() for l in labels]
    is_benchmark_search = any(l in ["indice", "benchmark"] for l in label_norm)

    for lab_div in soup.select("div.vallabel"):
        text = lab_div.get_text(" ", strip=True).lower()
        if text in label_norm:
            val_div = lab_div.find_next_sibling("div", class_="val")
            if val_div:
                # Se stiamo cercando il benchmark, il nome pulito è quasi sempre in un link
                if is_benchmark_search:
                    link = val_div.find("a")
                    if link and link.get_text(strip=True):
                        return link.get_text(strip=True)

                # Fallback per tutti i campi: prendi il testo completo
                full_text = val_div.get_text(" ", strip=True)

                # Se è il benchmark, tronca la descrizione
                if is_benchmark_search and '. ' in full_text:
                    return full_text.split('. ', 1)[0]

                return full_text
    return None

def parse_ter_from_html(html: str) -> Optional[float]:
    """
    Parses the Total Expense Ratio (TER) from HTML content.
    """
    patterns = [
        r"(?:TER|Total\s*expense\s*ratio|Ongoing\s*charges|Costi\s*correnti)[^%0-9]{0,60}([0-9]+(?:[.,][0-9]+)?)\s*%",
        r"Kostenquote[^%0-9]{0,60}([0-9]+(?:[.,][0-9]+)?)\s*%",
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", ".")
            try:
                f = float(val)
                if 0.0 < f < 3.0:
                    return f
            except Exception:
                continue
    text = _strip_tags(html).lower()
    m2 = re.search(r"(ter|total expense ratio|ongoing charges|costi correnti)[^%0-9]{0,60}([0-9]+(?:[.,][0-9]+)?)\s*%",
                   text, flags=re.IGNORECASE)
    if m2:
        try:
            f = float(m2.group(2).replace(",", "."))
            if 0.0 < f < 3.0:
                return f
        except Exception:
            pass
    return None

def parse_benchmark_name_from_html(html: str) -> Optional[str]:
    """
    Parses the benchmark name from HTML content using the robust soup helper.
    """
    name = None
    try:
        soup = BeautifulSoup(html, "html.parser")
        name = _find_value_by_labels_soup(soup, ["indice", "benchmark", "index", "reference index"])
    except Exception:
        return None

    if name:
        clean = re.sub(r"\s*(?:Net\s*Total\s*Return|Total\s*Return|Price\s*Return|NR|TR|EUR)\s*$", "", name, flags=re.IGNORECASE).strip()
        if clean.endswith('.'):
            clean = clean[:-1]
        return clean

    return None

def parse_etf_details_from_html(html: str) -> dict:
    """
    Parses various ETF details from HTML content using BeautifulSoup first, then fallback regex.
    """
    details: Dict[str, str] = {}

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Primary extraction via soup
        details_map = {
            "benchmark_name": ["Indice", "Benchmark", "Index", "Reference index"],
            "category": ["Focus di investimento", "Investment focus", "Category"],
            "fund_size": ["Dimensione del fondo", "Fund size"],
            "replication_method": ["Replicazione", "Metodo di replica", "Replication", "Method of replication"],
            "distribution": ["Distribuzione", "Distribution policy", "Distribution"],
            "provider": ["Fornitore", "Provider", "Issuer"],
        }

        for key, labels in details_map.items():
            val = _find_value_by_labels_soup(soup, labels)
            if val:
                details[key] = val.strip()

    except Exception:
        pass

    # Fallback regex on stripped text (con etichette italiane incluse)
    text_fallback = _strip_tags(html)
    patterns = {
        "benchmark_name": r"(?:Indice|Reference index|Benchmark|Index)\s*[:\-]?\s*(.+?)(?:\s*(?:Focus di investimento|Fund size|Dimensione del fondo|$))",
        "category": r"(?:Focus di investimento|Investment focus|Category)\s*[:\-]?\s*(.+?)(?:\s*(?:Fund size|Dimensione del fondo|Fornitore|Provider|$))",
        "fund_size": r"(?:Dimensione del fondo|Fund size)\s*[:\-]?\s*(.+?)(?:\s*(?:Total expense ratio|TER|Costi correnti|Indicatore sintetico|$))",
        "replication_method": r"(?:Replicazione|Metodo di replica|Replication|Method of replication)\s*[:\-]?\s*(.+?)(?:\s*(?:Distribuzione|Distribution|$))",
        "distribution": r"(?:Distribuzione|Distribution policy|Distribution)\s*[:\-]?\s*(.+?)(?:\s*(?:Fund size|Dimensione del fondo|$))",
        "provider": r"(?:Fornitore|Provider|Issuer)\s*[:\-]?\s*(.+?)(?:\s*(?:TER|Total expense ratio|$))"
    }

    for key, pattern in patterns.items():
        if key not in details:
            m_text = re.search(pattern, text_fallback, re.IGNORECASE | re.DOTALL)
            if m_text:
                details[key] = m_text.group(1).strip()

    # IMPORTANT: keep original strings; no aggressive normalization on replication/distribution/fund_size.
    if 'fund_size' in details and details['fund_size']:
        details['fund_size'] = details['fund_size'].replace("\xa0", " ").strip()

    return details