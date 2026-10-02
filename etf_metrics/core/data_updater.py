# -*- coding: utf-8 -*-
"""Aggiornamento dei dati di mercato nel DB con logica incrementale (delta).

Punto d'ingresso: update_tickers(tickers)

- Ticker nuovo (non nel DB): download completo dello storico richiesto
- Ticker già nel DB: download SOLO delle righe successive all'ultima data
  disponibile (l'ultima giornata viene riscaricata e aggiornata, per correggere
  eventuali dati parziali), poi upsert nel DB senza toccare lo storico esistente
- I ticker in cache negativa ("no data found") vengono saltati e riportati
- Rate limit Yahoo (HTTP 429): il download si interrompe in modo pulito,
  restituendo un report con i ticker non processati
"""
import logging
import time
from typing import Callable, Dict, List, Optional

from etf_metrics.core.data_manager import MarketDataManager
from etf_metrics.clients.yahoo_client import (
    get_series_bulk,
    get_names_bulk,
    get_failed_tickers,
    is_rate_limited,
)

logger = logging.getLogger(__name__)


def _normalize_tickers(tickers: List[str]) -> List[str]:
    """Deduplica preservando l'ordine, uppercase, rimuove vuoti."""
    return list(dict.fromkeys(t.strip().upper() for t in (tickers or []) if t and t.strip()))


def update_tickers(
    tickers: List[str],
    period: str = "5y",
    force_full: bool = False,
    skip_updated_today: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """Aggiorna i ticker richiesti nel DB (download completo per i nuovi, delta per gli altri).

    Args:
        tickers: lista di ticker (o testo grezzo: viene normalizzata)
        period: storico da scaricare per i ticker NUOVI (default 5y)
        force_full: se True riscarica l'intero storico anche per i ticker già nel DB
        skip_updated_today: se True salta i ticker già sincronizzati oggi (utile per
            "aggiorna tutto il DB" senza sprecare richieste)
        progress_callback(done, total, label): opzionale, per la UI

    Returns:
        {
          'requested': int,                    # ticker richiesti (dopo normalizzazione)
          'results': {ticker: {                # ticker aggiornati con successo
              'mode': 'delta'|'full',
              'new_rows': int, 'total_rows': int, 'first': str, 'last': str}},
          'skipped': {ticker: motivo},         # blacklist / già aggiornato / rate limit / no data
          'rate_limited': bool,
          'elapsed_sec': float,
        }
    """
    t0 = time.time()
    dm = MarketDataManager()

    unique = _normalize_tickers(tickers)
    report = {
        'requested': len(unique),
        'results': {},
        'skipped': {},
        'rate_limited': False,
        'elapsed_sec': 0.0,
    }
    if not unique:
        report['elapsed_sec'] = time.time() - t0
        return report

    # 1. Cache negativa: niente richieste per i ticker noti come "no data found"
    failed = get_failed_tickers()
    candidates = []
    for t in unique:
        if t in failed:
            report['skipped'][t] = "in cache negativa (no data found su Yahoo)"
        else:
            candidates.append(t)

    # 2. (Opzionale) salta i già sincronizzati oggi
    if skip_updated_today:
        updated_today = dm.get_tickers_updated_today()
        remaining = []
        for t in candidates:
            if t in updated_today:
                report['skipped'][t] = "già aggiornato oggi"
            else:
                remaining.append(t)
        candidates = remaining

    # 3. Piano di download: raggruppa per ultima data disponibile nel DB.
    #    I ticker aggiornati insieme condividono la stessa ultima data -> un solo
    #    batch yf.download per gruppo, con start = data (download delta).
    last_dates = dm.get_last_dates(candidates)
    full_group: List[str] = []
    delta_groups: Dict[str, List[str]] = {}
    for t in candidates:
        if force_full or t not in last_dates:
            full_group.append(t)
        else:
            delta_groups.setdefault(last_dates[t].strftime('%Y-%m-%d'), []).append(t)

    total = len(candidates)
    done = 0

    # Lavori ordinati: prima i delta (economici), poi il full (costoso)
    jobs = [("delta", start_key, group) for start_key, group in sorted(delta_groups.items())]
    if full_group:
        jobs.append(("full", None, full_group))

    missing_name = set(dm.get_tickers_missing_name(candidates))

    for kind, start_key, group in jobs:
        if is_rate_limited():
            report['rate_limited'] = True
            break

        label = (f"Download delta dal {start_key}: {len(group)} ticker"
                 if kind == "delta" else
                 f"Download completo ({period}): {len(group)} ticker")
        base = done
        group_len = len(group)

        def _bulk_cb(batch_done, total_batches, downloaded, base=base,
                     group_len=group_len, label=label):
            if progress_callback:
                try:
                    frac = batch_done / total_batches if total_batches else 1.0
                    progress_callback(min(total, base + frac * group_len), total, label)
                except Exception:
                    pass

        try:
            if kind == "delta":
                new_data = get_series_bulk(group, start=start_key, progress_callback=_bulk_cb)
            else:
                new_data = get_series_bulk(group, period=period, progress_callback=_bulk_cb)
        except Exception as e:
            logger.error(f"Download {'delta' if kind == 'delta' else 'full'} fallito: {e}")
            new_data = {}

        if new_data:
            # Nomi solo per i ticker senza nome nel DB (best effort: saltato se rate-limited)
            names = {}
            need_names = [t for t in new_data if t in missing_name]
            if need_names:
                try:
                    names = get_names_bulk(need_names)
                except Exception:
                    names = {}
            merged = dm.merge_bulk_data(new_data, names_dict=names)
            for t, ms in merged.items():
                report['results'][t] = {'mode': kind, **ms}

        done += group_len
        if progress_callback:
            try:
                progress_callback(min(total, done), total, label)
            except Exception:
                pass

        if is_rate_limited():
            report['rate_limited'] = True
            break

    # 4. Chi non è stato processato: motivo
    for t in candidates:
        if t not in report['results'] and t not in report['skipped']:
            report['skipped'][t] = ("rate limit Yahoo (riprova più tardi)"
                                    if report['rate_limited']
                                    else "nessun dato ricevuto da Yahoo")

    report['elapsed_sec'] = time.time() - t0
    return report
