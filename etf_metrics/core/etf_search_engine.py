# -*- coding: utf-8 -*-
import json
import logging
import os
from datetime import date
from typing import Callable, Dict, List, Iterable, Optional
from functools import lru_cache
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor

from etf_metrics.clients.yahoo_client import (
    yahoo_search, get_info, get_series_bulk, get_names_bulk)
from etf_metrics.core.data_manager import MarketDataManager
from etf_metrics.shared.utils import pick_preferred_symbol
from etf_metrics.shared.config import (
    PREFERRED_SUFFIXES, BULK_INFO_WORKERS, DEFAULT_SEED_QUERIES,
    ETF_UNIVERSE_FILE)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32768)
def _get_info_for_ticker(ticker: str) -> Dict:
    """Info (ISIN, nome) di un ticker, cached LRU."""
    if not ticker:
        return {}
    try:
        return get_info(ticker=ticker) or {}
    except Exception:
        return {}


def prefetch_ticker_infos(tickers: List[str], max_workers: int = None) -> None:
    """Precarica in parallelo le info (ISIN/nomi) riempiendo la cache LRU.

    Con universi grandi (es. tutto Xetra, ~5000 strumenti) il recupero sequenziale
    delle info è il collo di bottiglia principale: parallelizzandolo si passa da
    decine di minuti a qualche minuto (una sola volta, poi è tutto in cache/DB).
    """
    tickers = [t for t in (tickers or []) if t]
    if not tickers:
        return
    workers = int(max_workers or BULK_INFO_WORKERS)

    def _fetch(t: str):
        try:
            _get_info_for_ticker(t)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_fetch, tickers))


def _normalize_name(name: str) -> str:
    """Normalizza un nome ETF per il confronto (minuscolo, senza suffissi
    condivisi tipo 'UCITS ETF', valute, classi)."""
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r'\bucits etf\b', '', name)
    name = re.sub(r'\b(usd|eur|gbp)\b', '', name)
    name = re.sub(r'\b(acc|dist|accumulating|distributing)\b', '', name)
    name = re.sub(r'\bclass\s[a-z]\b', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    return " ".join(name.split())

@lru_cache(maxsize=32768)
def _get_isin_for_ticker(ticker: str) -> Optional[str]:
    """ISIN del ticker (dalle info cached), None se non noto."""
    info = _get_info_for_ticker(ticker)
    v = info.get("isin")
    if isinstance(v, str) and len(v) == 12:
        return v.strip().upper()
    for k, val in info.items():
        if "isin" in k.lower() and isinstance(val, str) and len(val) == 12:
            return val.strip().upper()
    return None


def get_unique_preferred_tickers(tickers: List[str]) -> List[str]:
    """Deduplica i ticker per ISIN e poi per nome ETF normalizzato, tenendo
    il ticker preferito (suffissi in config.PREFERRED_SUFFIXES) per ogni ETF."""
    if not tickers:
        return []

    # Precarica le info in parallelo: il dedup qui sotto ne fa una per ticker,
    # in sequenza sarebbe lentissimo con migliaia di strumenti.
    if len(tickers) > 8:
        prefetch_ticker_infos(tickers)

    isin_to_candidates = defaultdict(list)
    tickers_without_isin = []
    for ticker in tickers:
        isin = _get_isin_for_ticker(ticker)
        if isin:
            isin_to_candidates[isin].append(ticker)
        else:
            tickers_without_isin.append(ticker)

    preferred_tickers_from_isin = {
        pick_preferred_symbol(candidates, PREFERRED_SUFFIXES)
        for candidates in isin_to_candidates.values()
    }
    all_preferred_tickers = list(dict.fromkeys(list(preferred_tickers_from_isin) + tickers_without_isin))

    name_to_candidates = defaultdict(list)
    tickers_without_name = []

    for ticker in all_preferred_tickers:
        if not ticker: continue
        info = _get_info_for_ticker(ticker)
        long_name = info.get("longName") or info.get("shortName")
        if long_name:
            normalized_name = _normalize_name(long_name)
            if len(normalized_name) > 10:
                name_to_candidates[normalized_name].append(ticker)
            else:
                tickers_without_name.append(ticker)
        else:
            tickers_without_name.append(ticker)

    final_tickers = []
    for name, candidates in name_to_candidates.items():
        if candidates:
            preferred = pick_preferred_symbol(candidates, PREFERRED_SUFFIXES)
            if preferred:
                final_tickers.append(preferred)

    final_tickers.extend(tickers_without_name)
    return list(dict.fromkeys(final_tickers))


def discover_universe(queries: Iterable[str], quotes_per_query: int,
                      instrument_types: List[str] = None,
                      progress: Optional[Callable] = None) -> List[str]:
    """Universo di strumenti (ETF/ETP/ETN) da ricerche Yahoo, deduplicato.

    progress(done, total, msg) opzionale per la UI.
    """
    if instrument_types is None:
        instrument_types = ["ETF", "ETP", "ETN"]
    queries = list(queries)
    tickers_seen, ticker_list = set(), []
    for i, query in enumerate(queries, 1):
        if progress:
            try:
                progress(i, len(queries), f"Ricerca: {query}")
            except Exception:
                pass
        try:
            quotes = yahoo_search(query, quotes_count=quotes_per_query)
            for q in quotes:
                ticker = q.get("symbol")
                quote_type = (q.get("quoteType") or "").upper()
                if ticker and ticker not in tickers_seen and quote_type in instrument_types:
                    logger.info(f"Ticker trovato: {ticker} (Tipo: {quote_type})")
                    tickers_seen.add(ticker)
                    ticker_list.append(ticker)
        except Exception as e:
            logger.warning(f"Errore ricerca per '{query}': {e}")
    return ticker_list


# --- Persistenza dell'universo ETF (file JSON accanto agli altri stati) ---

def save_etf_universe(tickers: List[str]) -> Dict:
    """Salva l'universo ETF (unione con quello esistente) su file JSON.

    L'unione evita che una discovery parziale (o una risposta povera di Yahoo)
    faccia sparire strumenti già trovati. Ritorna il riepilogo salvato.
    """
    new = {str(t).strip().upper() for t in (tickers or []) if t and str(t).strip()}
    prev = set(load_etf_universe())
    merged = sorted(prev | new)
    payload = {
        "tickers": merged,
        "count": len(merged),
        "last_discovery": date.today().isoformat(),
    }
    try:
        with open(ETF_UNIVERSE_FILE, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Errore salvataggio {ETF_UNIVERSE_FILE}: {e}")
    return payload


def load_etf_universe() -> List[str]:
    """Ticker dell'universo ETF scoperto ([] se non ancora scoperto)."""
    try:
        with open(ETF_UNIVERSE_FILE, "r") as f:
            data = json.load(f)
        return [str(t).strip().upper() for t in (data.get("tickers") or [])
                if t and str(t).strip()]
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning(f"Errore lettura {ETF_UNIVERSE_FILE}: {e}")
        return []


def etf_universe_info() -> Dict:
    """Metadati dell'universo (ultima discovery, numero strumenti)."""
    try:
        with open(ETF_UNIVERSE_FILE, "r") as f:
            data = json.load(f)
        return {
            "count": int(data.get("count") or len(data.get("tickers") or [])),
            "last_discovery": data.get("last_discovery"),
            "file": os.path.abspath(ETF_UNIVERSE_FILE),
        }
    except FileNotFoundError:
        return {"count": 0, "last_discovery": None, "file": os.path.abspath(ETF_UNIVERSE_FILE)}
    except Exception as e:
        logger.warning(f"Errore lettura {ETF_UNIVERSE_FILE}: {e}")
        return {"count": 0, "last_discovery": None, "file": os.path.abspath(ETF_UNIVERSE_FILE)}


def load_universe_to_db(queries: Iterable[str] = None, quotes_per_query: int = 200,
                         period: str = "5y", instrument_types: List[str] = None,
                         progress: Optional[Callable] = None) -> Dict:
    """Discovery dell'universo ETF/ETP/ETN e salvataggio nel DB locale.

    Flusso (centralizzato: è l'UNICA via di popolamento dell'universo):
    1. ricerche Yahoo (query seed in config) -> strumenti ETF/ETP/ETN
    2. dedup per ISIN e nome, tenendo il listato preferito (.DE prima)
    3. lista persistita su ETF_UNIVERSE_FILE (unione con la precedente)
    4. download delta/full dei ticker non aggiornati oggi -> DB Parquet

    progress(done, total, msg) opzionale per la UI. Ritorna un report:
    {found, unique, downloaded, cached_today, failed, skipped, rate_limited,
     universe_count, last_discovery}
    """
    if queries is None:
        queries = DEFAULT_SEED_QUERIES
    report = {
        'found': 0, 'unique': 0, 'downloaded': 0, 'cached_today': 0,
        'failed': 0, 'skipped': 0, 'rate_limited': False,
        'universe_count': 0, 'last_discovery': None,
    }

    # 1. Discovery via ricerche Yahoo
    raw_tickers = discover_universe(queries, quotes_per_query=quotes_per_query,
                                    instrument_types=instrument_types,
                                    progress=progress)
    report['found'] = len(raw_tickers)
    if not raw_tickers:
        return report

    # 2. Dedup ISIN/nome con listato preferito (.DE > .MI > .AS > .L)
    if progress:
        try:
            progress(0, 1, "Deduplicazione per ISIN e nome (info ticker in parallelo)…")
        except Exception:
            pass
    unique_tickers = get_unique_preferred_tickers(raw_tickers)
    report['unique'] = len(unique_tickers)
    if not unique_tickers:
        return report

    # 3. Persiste la lista dell'universo (unione con la precedente)
    payload = save_etf_universe(unique_tickers)
    report['universe_count'] = payload['count']
    report['last_discovery'] = payload['last_discovery']

    # 4. Download: delta per i ticker già nel DB, full per i nuovi
    dm = MarketDataManager()
    to_download = dm.get_tickers_needing_update(unique_tickers)
    report['cached_today'] = len(unique_tickers) - len(to_download)

    if to_download:
        if progress:
            try:
                progress(0, 1, f"Download storico ({period}) per {len(to_download)} strumenti…")
            except Exception:
                pass

        def _dl_cb(batch_done, total_batches, downloaded):
            if progress:
                try:
                    progress(batch_done, total_batches,
                             f"Download storico: {downloaded} strumenti scaricati")
                except Exception:
                    pass

        try:
            downloaded = get_series_bulk(to_download, period=period,
                                         progress_callback=_dl_cb if progress else None)
        except Exception as e:
            logger.error(f"Download universo fallito: {e}")
            downloaded = {}
        if downloaded:
            names = get_names_bulk(list(downloaded.keys()))
            dm.save_bulk_data(downloaded, names_dict=names)
            report['downloaded'] = len(downloaded)
        report['failed'] = len(to_download) - len(downloaded)
        try:
            from etf_metrics.clients.yahoo_client import is_rate_limited
            report['rate_limited'] = bool(is_rate_limited())
        except Exception:
            pass

    return report