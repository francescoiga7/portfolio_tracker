# -*- coding: utf-8 -*-
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from etf_metrics.shared.config import (
    REQUEST_HEADERS,
    MANUAL_ISIN_MAP,
    BULK_DOWNLOAD_BATCH_SIZE,
    BULK_DOWNLOAD_THREADS,
    BULK_INFO_WORKERS,
    BULK_BATCH_PAUSE,
    YAHOO_RATE_LIMIT_COOLDOWN,
)
from etf_metrics.shared.utils import pick_preferred_symbol
from .base_client import BaseFinancialClient

logger = logging.getLogger(__name__)

UNKNOWN = "Sconosciuto"
BASE_INFO = {"fundFamily": UNKNOWN, "category": UNKNOWN}

# TTL della cache negativa in memoria (secondi): dopo N secondi viene rilettta dal DB
_FAILED_CACHE_TTL = 60.0
# TTL dell'health-check di rete (secondi)
_NETWORK_CHECK_TTL = 120.0


# ------------------------------------------------------------------
# Gestione rate limit Yahoo (HTTP 429 / "Too Many Requests")
#
# yf.download NON lancia un'eccezione in caso di 429: si limita a loggare
# "N Failed downloads: [...]: YFRateLimitError(...)" sul logger "yfinance".
# Un handler dedicato intercetta quei messaggi e attiva un cooldown globale:
# finché è attivo, tutte le chiamate di rete vengono sospese (niente martellate)
# e nessun ticker finisce in blacklist per colpa del rate limit.
# ------------------------------------------------------------------
_rate_lock = threading.Lock()
_RATE_LIMITED_UNTIL = 0.0  # time.monotonic() fino al quale evitare chiamate

_RATE_LIMIT_PATTERNS = (
    "yratelimiterror",
    "too many requests",
    "rate limit",
    "rate-limit",
    "rate limited",
    "http 429",
    "response 429",
    "status 429",
)


def _looks_like_rate_limit(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _RATE_LIMIT_PATTERNS)


def is_rate_limited() -> bool:
    """True se Yahoo ha risposto 429 di recente e il cooldown è ancora attivo."""
    return rate_limit_wait_seconds() > 0


def rate_limit_wait_seconds() -> float:
    """Secondi (approssimati) rimanenti di cooldown per il rate limit Yahoo."""
    return max(0.0, _RATE_LIMITED_UNTIL - time.monotonic())


def _mark_rate_limited(reason: str = "") -> None:
    """Attiva/estende il cooldown: tutte le chiamate di rete vengono sospese."""
    global _RATE_LIMITED_UNTIL
    with _rate_lock:
        _RATE_LIMITED_UNTIL = max(_RATE_LIMITED_UNTIL, time.monotonic() + YAHOO_RATE_LIMIT_COOLDOWN)
    logger.warning(
        "Yahoo ha risposto 429 (Too Many Requests)%s: download sospesi per %d minuti. "
        "I ticker non verranno messi in blacklist per questo motivo.",
        f" ({reason})" if reason else "",
        max(1, round(YAHOO_RATE_LIMIT_COOLDOWN / 60)),
    )


def _reset_rate_limit() -> None:
    """Solo per test: azzera il cooldown del rate limit."""
    global _RATE_LIMITED_UNTIL
    with _rate_lock:
        _RATE_LIMITED_UNTIL = 0.0


class _RateLimitDetector(logging.Handler):
    """Intercetta i log di yfinance per rilevare i 429 anche quando
    yf.download li registra invece di lanciarli (comportamento standard)."""

    def emit(self, record):
        try:
            if _looks_like_rate_limit(record.getMessage()):
                _mark_rate_limited()
        except Exception:
            pass


def _install_rate_limit_detector() -> None:
    yf_logger = logging.getLogger("yfinance")
    if not any(isinstance(h, _RateLimitDetector) for h in yf_logger.handlers):
        yf_logger.addHandler(_RateLimitDetector())
    # assicura che i WARNING di yfinance (es. "Crumb fetch rate-limited") passino
    if yf_logger.getEffectiveLevel() > logging.WARNING:
        yf_logger.setLevel(logging.WARNING)


_install_rate_limit_detector()


def _get_data_manager():
    """Import lazy per evitare dipendenze circolari clients <-> core."""
    try:
        from etf_metrics.core.data_manager import MarketDataManager
        return MarketDataManager()
    except Exception as e:
        logger.debug(f"MarketDataManager non disponibile: {e}")
        return None


class YahooClient(BaseFinancialClient):
    """Client Yahoo Finance che riusa la sessione con retry della BaseFinancialClient.

    Gestisce inoltre:
    - cache negativa persistente dei ticker "no data found" (non vengono riscaricati)
    - download massivo in batch paralleli (get_series_bulk)
    - recupero parallelo di nomi/info (get_names_bulk / get_infos_bulk)
    """

    def __init__(self):
        super().__init__("YahooFinance", base_timeout=12)
        self._search_hosts = ("query1", "query2")
        # Cache negativa dei ticker senza dati (in memoria + persistita su DB)
        self._failed_cache: Optional[set] = None
        self._failed_cache_ts: float = 0.0
        # Health check di rete (per distinguere "ticker inesistente" da "Yahoo offline")
        self._net_ok_ts: float = 0.0
        self._net_ok_val: bool = False

    def validate_input(self, input_data: str) -> bool:
        """Accetta qualsiasi stringa non vuota (ticker, query o ISIN)."""
        return isinstance(input_data, str) and input_data.strip() != ""

    # ------------------------------------------------------------------
    # Cache negativa dei ticker "no data found"
    # ------------------------------------------------------------------

    def _load_failed_cache(self, force: bool = False) -> set:
        """Legge la blacklist dei ticker falliti dal DB (con TTL in memoria)."""
        now = time.time()
        if not force and self._failed_cache is not None and (now - self._failed_cache_ts) < _FAILED_CACHE_TTL:
            return self._failed_cache
        dm = _get_data_manager()
        if dm is not None:
            try:
                self._failed_cache = dm.get_failed_tickers()
            except Exception:
                if self._failed_cache is None:
                    self._failed_cache = set()
        else:
            self._failed_cache = self._failed_cache or set()
        self._failed_cache_ts = now
        return self._failed_cache

    def _record_failed_tickers(self, tickers: List[str], reason: str = "no data found") -> None:
        """Aggiunge i ticker alla blacklist persistente (DB) e a quella in memoria.

        Gli indici (^VIX, ^GSPC, ...) non vengono mai blacklistati: sono riferimenti
        di mercato e un loro fallimento è quasi sempre un problema temporaneo.
        """
        tickers = [t for t in (tickers or []) if t and not t.startswith("^")]
        if not tickers:
            return
        dm = _get_data_manager()
        if dm is not None:
            try:
                dm.record_failed_tickers(tickers, reason=reason)
            except Exception as e:
                logger.warning(f"Impossibile persistire i ticker falliti: {e}")
        if self._failed_cache is not None:
            self._failed_cache.update(tickers)
        logger.info(f"Ticker senza dati ({len(tickers)}): aggiunti alla cache negativa. "
                    f"Esempi: {tickers[:5]}")

    def is_failed_ticker(self, ticker: str) -> bool:
        """True se il ticker è noto come 'senza dati' (verrà saltato senza chiamare Yahoo)."""
        return ticker in self._load_failed_cache()

    def clear_failed_tickers(self, tickers: List[str] = None) -> int:
        """Svuota la blacklist dei ticker falliti (tutti o solo quelli indicati)."""
        dm = _get_data_manager()
        removed = 0
        if dm is not None:
            try:
                removed = dm.clear_failed_tickers(tickers)
            except Exception as e:
                logger.warning(f"Impossibile pulire la cache negativa: {e}")
        self._load_failed_cache(force=True)
        return removed

    def get_failed_tickers(self) -> set:
        """Copia dell'insieme dei ticker attualmente in blacklist."""
        return set(self._load_failed_cache())

    def _network_ok(self) -> bool:
        """Verifica se Yahoo risponde (con cache a breve termine).

        Serve a distinguere un ticker davvero senza dati da un problema di rete:
        solo se la rete funziona un risultato vuoto viene considerato definitivo
        e il ticker finisce in blacklist.
        """
        # Rate limit attivo = rete "non affidabile": mai blacklistare in questo stato
        if is_rate_limited():
            return False
        now = time.time()
        if now - self._net_ok_ts < _NETWORK_CHECK_TTL:
            return self._net_ok_val
        try:
            df = yf.Ticker("AAPL").history(period="5d")
            self._net_ok_val = df is not None and not df.empty
        except Exception:
            self._net_ok_val = False
        self._net_ok_ts = now
        if not self._net_ok_val:
            logger.warning("Health-check Yahoo fallito: rete non raggiungibile, "
                           "i risultati vuoti NON verranno messi in blacklist.")
        return self._net_ok_val

    def _search_once(self, host: str, query: str, quotes_count: int) -> List[Dict]:
        url = f"https://{host}.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": quotes_count, "newsCount": 0, "listsCount": 0}

        try:
            resp = self.safe_request(url, headers=REQUEST_HEADERS, params=params, timeout=10)
            if not resp:
                return []
            try:
                data = resp.json() or {}
            except Exception:
                return []
            return data.get("quotes", []) or []
        except Exception as e:
            logger.warning(f"Yahoo Search failed on {host}: {e}")
            return []

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
        """Tenta di risolvere un ticker nel suo ISIN usando get_info."""
        info = self.get_info(ticker=ticker)
        return info.get('isin')

    def get_series(self, ticker: str, period: str, as_dataframe: bool = False) -> Optional[
        Union[pd.Series, pd.DataFrame]]:
        """
        Recupera la serie storica da Yahoo Finance.
        Restituisce una pd.Series (default) o un pd.DataFrame se as_dataframe=True.

        I ticker in cache negativa ("no data found") vengono saltati immediatamente,
        senza alcuna chiamata di rete. Un ticker senza dati viene registrato in blacklist
        SOLO se l'health-check conferma che Yahoo è raggiungibile.
        """
        if not self.validate_input(ticker):
            return None

        if self.is_failed_ticker(ticker):
            logger.debug(f"{ticker}: in cache negativa (no data), download saltato.")
            return None

        # Rate limit attivo: sospende subito, senza consumare richieste
        if is_rate_limited():
            logger.warning(f"{ticker}: download saltato, rate limit Yahoo attivo "
                           f"(ancora {rate_limit_wait_seconds():.0f}s).")
            return None

        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
            if df.empty:
                logger.warning(f"Nessun dato storico per {ticker} nel periodo {period}")
                # Blacklist solo se la rete funziona: altrimenti è un problema di connessione
                if self._network_ok():
                    self._record_failed_tickers([ticker], reason=f"no data for period={period}")
                return None

            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()

            if as_dataframe:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                return df[required_cols]

            col = "Adj Close" if ("Adj Close" in df.columns and not df["Adj Close"].isna().all()) else "Close"
            s = df[col].dropna().copy()
            s.name = ticker
            if len(s) < 2:
                if self._network_ok():
                    self._record_failed_tickers([ticker], reason="series too short")
                return None
            if (s <= 0).any():
                s = s[s > 0]
            if s.empty:
                return None
            return s
        except Exception as e:
            # Errore di rete/HTTP: NON registrare in blacklist (non è colpa del ticker)
            if _looks_like_rate_limit(f"{type(e).__name__}: {e}"):
                _mark_rate_limited(reason=f"get_series {ticker}")
            logger.error(f"Errore nel recupero dati per {ticker}: {e}")
            return None

    def get_info(self, isin: Optional[str] = None, ticker: Optional[str] = None) -> Dict:
        if not isin and not ticker:
            return {}

        # Rate limit attivo: la richiesta fallirebbe comunque, evita di aggravarlo
        if is_rate_limited():
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

    # ------------------------------------------------------------------
    # Download massivo (universi grandi, es. tutto Xetra)
    # ------------------------------------------------------------------

    def get_series_bulk(
        self,
        tickers: List[str],
        period: str = "5y",
        start=None,
        batch_size: int = None,
        threads: Union[bool, int] = None,
        progress_callback=None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download massivo dei prezzi con yf.download in batch paralleli.

        Vantaggi rispetto al loop di get_series (o a un'unica chiamata gigante):
        - batch isolati: un fallimento non blocca l'universo intero
        - i ticker in cache negativa vengono saltati senza chiamate di rete
        - finiscono in blacklist SOLO i ticker completamente senza dati, e solo
          se il batch ha avuto successo almeno parziale (rete funzionante);
          i ticker con poche righe (es. ETF appena quotati) NON vengono
          blacklistati: verranno ritentati e cresceranno nei run successivi
        - progress_callback(batch_done, total_batches, downloaded_count) per la UI
        - start="YYYY-MM-DD" (opzionale): download DELTA da quella data (inclusa)
          invece dell'intero period — usato dagli aggiornamenti incrementali

        Ritorna {ticker: DataFrame[Open, High, Low, Close, Volume]}.
        """
        batch_size = int(batch_size or BULK_DOWNLOAD_BATCH_SIZE)
        if threads is None:
            threads = BULK_DOWNLOAD_THREADS

        # Deduplica preservando l'ordine, salta i ticker in blacklist
        unique = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
        failed_cache = self._load_failed_cache()
        to_download = [t for t in unique if t not in failed_cache]
        skipped = len(unique) - len(to_download)
        if skipped:
            logger.info(f"Cache negativa: {skipped} ticker saltati (nessun dato in passato).")

        results: Dict[str, pd.DataFrame] = {}
        if not to_download:
            return results

        # Rate limit attivo: niente martellate, si riprova dopo il cooldown.
        # I dati già nel DB restano comunque utilizzabili dal chiamante.
        if is_rate_limited():
            logger.warning("Download bulk sospeso: rate limit Yahoo attivo "
                           f"(ancora {rate_limit_wait_seconds():.0f}s). Riprova più tardi.")
            return results

        batches = [to_download[i:i + batch_size] for i in range(0, len(to_download), batch_size)]
        mode_desc = f"delta da {start}" if start is not None else f"periodo {period}"
        logger.info(f"Download bulk: {len(to_download)} ticker in {len(batches)} batch "
                    f"da {batch_size} ({mode_desc}, thread={threads}).")

        # Parametri di download: start (delta inclusivo) OPPURE period (storico completo)
        dl_kwargs = dict(group_by='ticker', auto_adjust=False, actions=False,
                         progress=False, threads=threads)
        if start is not None:
            dl_kwargs['start'] = str(start)
        else:
            dl_kwargs['period'] = period

        for bi, batch in enumerate(batches):
            # Un 429 in un batch precedente interrompe tutto: inutile peggiorare la situazione.
            # I risultati già ottenuti vengono comunque restituiti (e salvati dal chiamante).
            if is_rate_limited():
                logger.warning(f"Rate limit Yahoo rilevato: interrompo il download bulk al batch "
                               f"{bi + 1}/{len(batches)}. Restituiti {len(results)} ticker; "
                               f"riprova tra ~{rate_limit_wait_seconds() / 60:.0f} minuti.")
                break

            # Pausa tra i batch: riduce la probabilità di far scattare il rate limit
            if bi > 0 and BULK_BATCH_PAUSE > 0:
                time.sleep(BULK_BATCH_PAUSE)

            try:
                raw = yf.download(batch, **dl_kwargs)
            except Exception as e:
                # Fallimento dell'intero batch: quasi sicuramente rete/proxy, niente blacklist
                if _looks_like_rate_limit(f"{type(e).__name__}: {e}"):
                    _mark_rate_limited(reason=f"batch {bi + 1}")
                logger.warning(f"Batch {bi + 1}/{len(batches)} fallito (rete?): {e}")
                if progress_callback:
                    try:
                        progress_callback(bi + 1, len(batches), len(results))
                    except Exception:
                        pass
                if is_rate_limited():
                    break
                continue

            # Colonne piatte con batch multiplo: formato inatteso (download parzialmente
            # fallito). Per sicurezza il batch viene saltato SENZA blacklistare nulla.
            if not isinstance(raw.columns, pd.MultiIndex) and len(batch) > 1:
                logger.warning(f"Batch {bi + 1}/{len(batches)}: formato colonne inatteso, batch saltato.")
                if progress_callback:
                    try:
                        progress_callback(bi + 1, len(batches), len(results))
                    except Exception:
                        pass
                continue

            batch_empty: List[str] = []
            for t in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if t not in raw.columns.get_level_values(0):
                            batch_empty.append(t)
                            continue
                        df = raw[t].copy()
                    else:
                        # yf.download con un solo ticker ritorna colonne piatte
                        df = raw.copy()

                    df = df.dropna(how='all')
                    if df.empty:
                        batch_empty.append(t)
                        continue

                    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df.sort_index()

                    required = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required:
                        if col not in df.columns:
                            df[col] = 0.0
                    results[t] = df[required]
                except Exception:
                    batch_empty.append(t)

            # Euristica anti falsi positivi: la blacklist viene aggiornata solo se
            # il batch ha prodotto almeno un risultato (rete OK -> fallimenti reali).
            # Solo i ticker COMPLETAMENTE vuoti: chi ha poche righe potrà crescere.
            # Con il rate limit attivo (429) il vuoto NON è colpa del ticker: mai blacklistare.
            if is_rate_limited():
                if batch_empty:
                    logger.warning(f"Batch {bi + 1}/{len(batches)}: {len(batch_empty)} ticker senza dati, "
                                   f"ma rate limit attivo: niente blacklist.")
            elif batch_empty and len(results) > 0:
                self._record_failed_tickers(batch_empty, reason="no data found (bulk)")
            elif batch_empty:
                logger.warning(f"Batch {bi + 1}/{len(batches)}: {len(batch_empty)} ticker senza dati, "
                               f"ma nessun successo nel batch (rete incerta): niente blacklist.")

            if progress_callback:
                try:
                    progress_callback(bi + 1, len(batches), len(results))
                except Exception:
                    pass

        return results

    def get_infos_bulk(self, tickers: List[str], max_workers: int = None) -> Dict[str, Dict]:
        """Recupera le info (ISIN, nome, ecc.) di molti ticker in parallelo."""
        tickers = [t for t in (tickers or []) if t]
        if not tickers:
            return {}
        if is_rate_limited():
            logger.warning("get_infos_bulk sospeso: rate limit Yahoo attivo.")
            return {}
        workers = int(max_workers or BULK_INFO_WORKERS)

        def _fetch(t: str):
            try:
                return t, (self.get_info(ticker=t) or {})
            except Exception:
                return t, {}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return dict(ex.map(_fetch, tickers))

    def get_names_bulk(self, tickers: List[str], max_workers: int = None) -> Dict[str, str]:
        """Recupera {ticker: nome} di molti ticker in parallelo (fallback: il ticker stesso)."""
        infos = self.get_infos_bulk(tickers, max_workers=max_workers)
        names = {}
        for t, info in infos.items():
            name = info.get('longName') or info.get('shortName') or info.get('longname') or info.get('shortname')
            names[t] = name or t
        return names


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


def get_series_bulk(tickers: List[str], period: str = "5y", start=None, batch_size: int = None,
                    threads: Union[bool, int] = None,
                    progress_callback=None) -> Dict[str, pd.DataFrame]:
    """Download massivo in batch paralleli (vedi YahooClient.get_series_bulk).
    start="YYYY-MM-DD": download delta da quella data invece dell'intero period."""
    return _get_yahoo_client().get_series_bulk(
        tickers, period=period, start=start, batch_size=batch_size, threads=threads,
        progress_callback=progress_callback)


def get_infos_bulk(tickers: List[str], max_workers: int = None) -> Dict[str, Dict]:
    """Recupero parallelo delle info di molti ticker."""
    return _get_yahoo_client().get_infos_bulk(tickers, max_workers=max_workers)


def get_names_bulk(tickers: List[str], max_workers: int = None) -> Dict[str, str]:
    """Recupero parallelo dei nomi di molti ticker."""
    return _get_yahoo_client().get_names_bulk(tickers, max_workers=max_workers)


def is_failed_ticker(ticker: str) -> bool:
    """True se il ticker è noto come 'senza dati' su Yahoo."""
    return _get_yahoo_client().is_failed_ticker(ticker)


def get_failed_tickers() -> set:
    """Insieme dei ticker attualmente in cache negativa."""
    return _get_yahoo_client().get_failed_tickers()


def clear_failed_tickers(tickers: List[str] = None) -> int:
    """Svuota la cache negativa dei ticker falliti. Ritorna il numero di ticker rimossi."""
    return _get_yahoo_client().clear_failed_tickers(tickers)


def get_info(isin: Optional[str] = None, ticker: Optional[str] = None) -> Dict:
    return _get_yahoo_client().get_info(isin=isin, ticker=ticker)