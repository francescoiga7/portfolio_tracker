# -*- coding: utf-8 -*-
"""
Classe base per tutti i client esterni con gestione errori robusta e retry logic.
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class BaseFinancialClient(ABC):
    """Classe base per client finanziari con gestione errori robusta."""

    def __init__(self, name: str, base_timeout: int = 12):
        self.name = name
        self.base_timeout = base_timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Crea una sessione HTTP con retry automatico."""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def safe_request(self, url: str, headers: Dict[str, str], 
                    timeout: Optional[int] = None, **kwargs) -> Optional[requests.Response]:
        """Esegue una richiesta HTTP sicura con gestione errori."""
        timeout = timeout or self.base_timeout

        try:
            response = self.session.get(url, headers=headers, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            logger.warning(f"{self.name}: Timeout per {url}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"{self.name}: Errore di connessione per {url}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"{self.name}: HTTP error {e.response.status_code} per {url}")
        except Exception as e:
            logger.error(f"{self.name}: Errore imprevisto per {url}: {e}")

        return None

    def retry_operation(self, operation: Callable, max_retries: int = 3, 
                       delay: float = 1.0, backoff_factor: float = 2.0) -> Any:
        """Retry intelligente per operazioni critiche."""
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"{self.name}: Operazione fallita dopo {max_retries} tentativi: {e}")
                    raise

                sleep_time = delay * (backoff_factor ** attempt)
                logger.warning(f"{self.name}: Tentativo {attempt + 1} fallito, retry tra {sleep_time:.1f}s")
                time.sleep(sleep_time)

    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Valida i dati di input specifici del client."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()


class DataValidator:
    """Utility per validazione robusta dei dati finanziari."""

    @staticmethod
    def validate_isin(isin: str) -> bool:
        """Valida il formato ISIN."""
        import re
        return bool(re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', isin.strip().upper()))

    @staticmethod
    def validate_percentage(value: float, min_val: float = -100, max_val: float = 1000) -> bool:
        """Valida un valore percentuale."""
        return min_val <= value <= max_val

    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Valida un ticker symbol."""
        import re
        return bool(re.match(r'^[A-Z0-9]+(\.[A-Z]{1,3})?$', ticker.strip().upper()))

    @staticmethod
    def clean_currency_value(value: str) -> Optional[float]:
        """Pulisce e converte valori monetari."""
        import re
        if not value:
            return None

        # Rimuovi spazi e simboli di valuta
        cleaned = re.sub(r'[^\d.,\-]', '', value.replace('\xa0', ' '))

        # Gestisci formati europei (virgola come decimale)
        if ',' in cleaned and '.' in cleaned:
            # Se ci sono sia virgole che punti, la virgola è il separatore decimale
            if cleaned.rfind(',') > cleaned.rfind('.'):
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Solo virgole - potrebbe essere separatore migliaia o decimale
            if len(cleaned.split(',')[-1]) <= 2:  # Decimali
                cleaned = cleaned.replace(',', '.')
            else:  # Separatore migliaia
                cleaned = cleaned.replace(',', '')

        try:
            return float(cleaned)
        except ValueError:
            return None