# -*- coding: utf-8 -*-
"""Migrazione dello store dei prezzi verso Parquet partizionato per mese.

Fonte supportata: SQLite legacy (`market_data.db`).

La copia è un MERGE (upsert) nello store di destinazione: rieseguire la
migrazione è sicuro e non perde mai dati già presenti. La fonte NON viene
modificata.

Uso CLI:
    python3 scripts/migrate_to_parquet.py                      # auto-rileva la fonte
    python3 scripts/migrate_to_parquet.py --source mio.db
    python3 scripts/migrate_to_parquet.py --source mio.db --dest mio_parquet

Dalla UI: pagina 📥 Gestione Dati → pulsante "🚀 Migra a Parquet" (visibile
solo quando il backend in uso è SQLite).
"""
import logging
import os
import sqlite3
import time
from typing import Callable, Dict, Optional

import pandas as pd

from etf_metrics.core.parquet_store import ParquetStore, PRICES_COLUMNS
from etf_metrics.shared.config import PARQUET_STORE_DIR

logger = logging.getLogger(__name__)

_CHUNK_ROWS = 500_000


def _default_dest(source: str) -> str:
    """Store di destinazione di default: directory Parquet a fianco della fonte."""
    return os.path.join(os.path.dirname(os.path.abspath(source)),
                        PARQUET_STORE_DIR)


def _iter_month_blocks(chunks):
    """Raggruppa un flusso di DataFrame OHLCV (ordinati per data) in blocchi
    per mese calendario, così ogni partizione viene scritta una sola volta."""
    current_month = None
    buffer = []
    for chunk in chunks:
        if chunk is None or chunk.empty:
            continue
        chunk = chunk.copy()
        chunk['date'] = pd.to_datetime(chunk['date']).astype('datetime64[ns]')
        chunk = chunk.dropna(subset=['date'])
        if chunk.empty:
            continue
        chunk = chunk.sort_values('date', kind='stable')
        months = chunk['date'].dt.strftime('%Y-%m')
        boundaries = months.ne(months.shift()).cumsum()
        for _gid, block in chunk.groupby(boundaries, sort=False):
            month = block['date'].iloc[0].strftime('%Y-%m')
            if month != current_month and buffer:
                yield current_month, pd.concat(buffer, ignore_index=True)
                buffer = []
            current_month = month
            buffer.append(block)
    if buffer:
        yield current_month, pd.concat(buffer, ignore_index=True)


def _copy_prices(chunks, store: ParquetStore, report: dict,
                 progress_callback: Optional[Callable]) -> None:
    total = report.get('total_price_rows') or 0

    def _notify():
        if progress_callback:
            try:
                progress_callback(min(1.0, report['price_rows'] / total)
                                  if total else 1.0)
            except Exception:
                pass

    for _month, block in _iter_month_blocks(chunks):
        store.merge_prices(block)
        report['price_rows'] += len(block)
        report['months'] += 1
        _notify()


def _iso_date(value) -> Optional[str]:
    """Normalizza un valore data (str ISO, datetime, date) in 'YYYY-MM-DD'."""
    if value is None or (isinstance(value, float) and pd.isna(value)) \
            or value is pd.NaT:
        return None
    ts = pd.to_datetime(value, errors='coerce')
    return None if pd.isna(ts) else ts.strftime('%Y-%m-%d')


def _normalize_updates(df: pd.DataFrame) -> Dict[str, tuple]:
    """{ticker: (name, last_updated ISO, first_added ISO|None)} da un frame
    della tabella updates (colonne opzionali gestite)."""
    if df is None or df.empty:
        return {}
    out = {}
    for _, r in df.iterrows():
        name = r['name'] if ('name' in df.columns and pd.notna(r['name'])) else None
        first = (_iso_date(r.get('first_added'))
                 if 'first_added' in df.columns else None)
        out[str(r['ticker'])] = (name or None, _iso_date(r.get('last_updated')), first)
    return out


def _normalize_failed(df: pd.DataFrame) -> pd.DataFrame:
    """Frame failed_tickers normalizzato alle colonne/tipi dello store."""
    cols = ['ticker', 'reason', 'first_failed', 'last_failed', 'attempts']
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df[[c for c in cols if c in df.columns]].copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    for c in ('first_failed', 'last_failed'):
        out[c] = out[c].map(_iso_date)
    out['attempts'] = (pd.to_numeric(out['attempts'], errors='coerce')
                       .fillna(1).astype('int64'))
    out['ticker'] = out['ticker'].astype(str)
    return out[cols]


def migrate_sqlite_to_parquet(source: str = "market_data.db",
                              dest: Optional[str] = None,
                              progress_callback: Optional[Callable] = None) -> dict:
    """Copia un DB SQLite legacy nello store Parquet (merge/upsert, fonte intatta).

    Ritorna un report {'source', 'dest', 'kind', 'tickers', 'price_rows',
    'months', 'updates_rows', 'failed_rows', 'elapsed_sec'}.
    """
    t0 = time.time()
    source = os.path.abspath(str(source))
    if not os.path.exists(source):
        raise FileNotFoundError(f"DB SQLite di origine non trovato: {source}")
    dest = os.path.abspath(dest) if dest else _default_dest(source)
    store = ParquetStore(dest)

    report = {'source': source, 'dest': dest, 'kind': 'sqlite',
              'tickers': 0, 'price_rows': 0, 'months': 0,
              'updates_rows': 0, 'failed_rows': 0, 'total_price_rows': 0,
              'elapsed_sec': 0.0}

    conn = sqlite3.connect(source)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        if 'prices' in tables:
            report['total_price_rows'] = \
                conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] or 0

            def _chunks():
                for chunk in pd.read_sql(
                        "SELECT date, ticker, open, high, low, close, volume "
                        "FROM prices ORDER BY date", conn, chunksize=_CHUNK_ROWS):
                    yield chunk

            _copy_prices(_chunks(), store, report, progress_callback)

        if 'updates' in tables:
            upd = pd.read_sql("SELECT * FROM updates",
                              conn)
            store.upsert_updates(_normalize_updates(upd))
            report['updates_rows'] = len(upd)

        if 'failed_tickers' in tables:
            fail = _normalize_failed(pd.read_sql(
                "SELECT ticker, reason, first_failed, last_failed, attempts "
                "FROM failed_tickers", conn))
            store.upsert_failed_rows(fail)
            report['failed_rows'] = len(fail)
    finally:
        conn.close()

    report['tickers'] = len(store.distinct_tickers())
    report['elapsed_sec'] = time.time() - t0
    return report


    report['tickers'] = len(store.distinct_tickers())
    report['elapsed_sec'] = time.time() - t0
    return report


__all__ = ['PRICES_COLUMNS', 'migrate_sqlite_to_parquet']
