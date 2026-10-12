#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Migrazione dello store dei prezzi verso Parquet partizionato per mese.

Fonti supportate (rilevate dall'estensione):
- SQLite legacy (market_data.db)
- DuckDB legacy (market_data.duckdb)

La copia è un merge/upsert: rieseguire la migrazione è sicuro e la fonte NON
viene mai modificata. Con un file DuckDB come fonte serve il modulo `duckdb`
(che non è più una dipendenza del progetto):

    uv run --with duckdb python scripts/migrate_to_parquet.py --source market_data.duckdb

Esempi:
    python3 scripts/migrate_to_parquet.py                      # auto-rileva la fonte
    python3 scripts/migrate_to_parquet.py --source mio.db
    python3 scripts/migrate_to_parquet.py --source mio.duckdb --dest mio_parquet
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etf_metrics.core.store_migration import (  # noqa: E402
    migrate_sqlite_to_parquet,
    migrate_duckdb_to_parquet,
)


def _auto_source() -> str:
    """Fonte di default: il primo DB legacy presente nella directory corrente."""
    for candidate in ("market_data.db", "market_data.duckdb"):
        if os.path.exists(candidate):
            return candidate
    return "market_data.db"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migra il DB dei prezzi (SQLite o DuckDB legacy) nello store "
                    "Parquet partizionato per mese.")
    parser.add_argument("--source", default=None,
                        help="DB di origine (default: auto-rileva market_data.db / "
                             "market_data.duckdb nella directory corrente)")
    parser.add_argument("--dest", default=None,
                        help="Directory dello store Parquet di destinazione "
                             "(default: market_data_parquet a fianco della fonte)")
    args = parser.parse_args()

    source = args.source or _auto_source()
    if not os.path.exists(source):
        print(f"❌ DB di origine non trovato: {source}")
        return 1

    kind = "DuckDB" if source.lower().endswith(".duckdb") else "SQLite"
    print(f"Migrazione {kind} → Parquet: {source}")

    def _progress(frac):
        print(f"\r  prices: {frac * 100:5.1f}%", end="", flush=True)

    try:
        if kind == "DuckDB":
            report = migrate_duckdb_to_parquet(source, args.dest,
                                               progress_callback=_progress)
        else:
            report = migrate_sqlite_to_parquet(source, args.dest,
                                               progress_callback=_progress)
    except Exception as e:
        print(f"\n❌ Migrazione non riuscita: {e}")
        return 1
    print()

    print(f"✅ Migrazione completata in {report['elapsed_sec']:.1f}s:")
    print(f"   - righe di prezzi : {report['price_rows']:,}")
    print(f"   - ticker          : {report['tickers']}")
    print(f"   - mesi (partizioni): {report['months']}")
    print(f"   - righe updates   : {report['updates_rows']:,}")
    print(f"   - righe blacklist : {report['failed_rows']:,}")
    print(f"   - store Parquet   : {report['dest']}")
    print(f"   - fonte (intatta) : {report['source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
