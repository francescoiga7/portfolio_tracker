# -*- coding: utf-8 -*-
"""Migrazione del DB dei prezzi da SQLite (legacy) a DuckDB (motore analitico).

Perché migrare: DuckDB è un motore colonnare embedded che sulle letture
analitiche (milioni di righe OHLCV) è 10-50x più veloce di SQLite e produce
DataFrame pandas in modo vettoriale, senza la conversione riga per riga di
`pd.read_sql`. Per un universo grande come tutto Xetra la differenza si sente
su ogni simulazione di trading.

Uso CLI:
    python3 scripts/migrate_to_duckdb.py                      # market_data.db -> market_data.duckdb
    python3 scripts/migrate_to_duckdb.py --source mio.db --dest mio.duckdb
    python3 scripts/migrate_to_duckdb.py --force              # sovrascrive la destinazione

Dalla UI: pagina 📥 Gestione Dati → pulsante "🚀 Migra a DuckDB".
"""
import logging
import os
import sqlite3
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Righe per chunk nella copia di `prices` (limita la memoria su DB enormi)
_CHUNK_ROWS = 500_000


def _sqlite_table_columns(conn, table: str) -> list:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def migrate_sqlite_to_duckdb(
    source: str = "market_data.db",
    dest: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> dict:
    """Copia un DB SQLite esistente in un nuovo DB DuckDB.

    Ritorna un report {'tickers', 'price_rows', 'updates_rows', 'failed_rows',
    'dest', 'elapsed_sec'}. La fonte NON viene modificata.
    """
    import duckdb
    import time

    t0 = time.time()
    source = os.path.abspath(source)
    if dest is None:
        dest = os.path.join(os.path.dirname(source), "market_data.duckdb")
    dest = os.path.abspath(dest)

    if not os.path.exists(source):
        raise FileNotFoundError(f"DB SQLite di origine non trovato: {source}")
    if os.path.exists(dest):
        if not force:
            raise FileExistsError(
                f"Il DB di destinazione esiste già ({dest}): usa force/--force per sovrascriverlo")
        os.remove(dest)

    src = sqlite3.connect(source)
    report = {'tickers': 0, 'price_rows': 0, 'updates_rows': 0, 'failed_rows': 0,
              'dest': dest, 'elapsed_sec': 0.0}

    try:
        # totale righe per la barra di avanzamento
        try:
            total_prices = src.execute("SELECT COUNT(*) FROM prices").fetchone()[0] or 0
        except sqlite3.Error:
            total_prices = 0

        con = duckdb.connect(dest)
        try:
            con.execute('''CREATE TABLE updates (
                ticker VARCHAR PRIMARY KEY, name VARCHAR, last_updated DATE)''')
            con.execute('''CREATE TABLE prices (
                date TIMESTAMP, ticker VARCHAR, open DOUBLE, high DOUBLE,
                low DOUBLE, close DOUBLE, volume DOUBLE,
                PRIMARY KEY (ticker, date))''')
            con.execute('''CREATE TABLE failed_tickers (
                ticker VARCHAR PRIMARY KEY, reason VARCHAR, first_failed DATE,
                last_failed DATE, attempts INTEGER DEFAULT 1)''')

            # --- prices: a chunk per intervallo di ticker (usa l'indice PK) ---
            tickers = [r[0] for r in src.execute(
                "SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchall()]
            report['tickers'] = len(tickers)

            def _notify():
                if progress_callback:
                    try:
                        progress_callback(min(1.0, report['price_rows'] / total_prices)
                                          if total_prices else 1.0)
                    except Exception:
                        pass

            lo = 0
            while lo < len(tickers):
                # chunk di ~CHUNK_ROWS righe stimato sul numero di ticker
                hi = min(len(tickers), lo + 1)
                expected = total_prices / max(1, len(tickers))
                while hi < len(tickers) and (hi - lo) * expected < _CHUNK_ROWS:
                    hi += 1
                batch = tickers[lo:hi]
                lo = hi
                placeholders = ','.join(['?'] * len(batch))
                df = pd.read_sql(
                    f"SELECT date, ticker, open, high, low, close, volume "
                    f"FROM prices WHERE ticker IN ({placeholders})", src, params=batch)
                if df is None or df.empty:
                    continue
                con.register("_chunk", df)
                con.execute(
                    "INSERT INTO prices SELECT CAST(date AS TIMESTAMP), ticker, "
                    "open, high, low, close, volume FROM _chunk")
                con.unregister("_chunk")
                report['price_rows'] += len(df)
                _notify()
            con.commit()

            # --- updates (piccola) ---
            cols = _sqlite_table_columns(src, "updates")
            select = "ticker" + (", name" if "name" in cols else ", NULL AS name") \
                     + (", last_updated" if "last_updated" in cols else ", NULL AS last_updated")
            dfu = pd.read_sql(f"SELECT {select} FROM updates", src)
            if not dfu.empty:
                con.register("_u", dfu)
                con.execute("INSERT INTO updates SELECT ticker, name, "
                            "CAST(last_updated AS DATE) FROM _u")
                con.unregister("_u")
                report['updates_rows'] = len(dfu)

            # --- failed_tickers (piccola) ---
            try:
                dff = pd.read_sql(
                    "SELECT ticker, reason, first_failed, last_failed, attempts "
                    "FROM failed_tickers", src)
                if not dff.empty:
                    con.register("_f", dff)
                    con.execute("INSERT INTO failed_tickers SELECT ticker, reason, "
                                "CAST(first_failed AS DATE), CAST(last_failed AS DATE), "
                                "attempts FROM _f")
                    con.unregister("_f")
                    report['failed_rows'] = len(dff)
            except sqlite3.Error:
                pass  # tabella assente nel DB di origine

            con.commit()
        finally:
            con.close()
    finally:
        src.close()

    if progress_callback:
        try:
            progress_callback(1.0)
        except Exception:
            pass
    report['elapsed_sec'] = time.time() - t0
    logger.info("Migrazione completata: %(rows)d righe in %(sec).1fs -> %(dest)s",
                {"rows": report['price_rows'], "sec": report['elapsed_sec'], "dest": dest})
    return report
