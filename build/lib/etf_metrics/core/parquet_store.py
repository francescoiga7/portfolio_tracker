# -*- coding: utf-8 -*-
"""Store Parquet partizionato per mese per i dati di mercato OHLCV.

Layout su disco (partizionamento Hive, una directory per mese):

    <store_dir>/
    ├── prices/
    │   ├── month=2023-11/
    │   │   └── data.parquet
    │   ├── month=2023-12/
    │   │   └── data.parquet
    │   └── ...
    └── _meta/
        ├── updates.parquet          # registro aggiornamenti {ticker, name, last_updated}
        └── failed_tickers.parquet   # cache negativa {ticker, reason, ...}

Perché Parquet partizionato per mese:
- formato colonnare aperto (pyarrow): le letture proiettano solo le colonne
  richieste e saltano i row-group non pertinenti (statistiche + dizionari),
  con decodifica multithread nativa
- ogni upsert incrementale riscrive SOLO le partizioni dei mesi coinvolti
  (tipicamente quella corrente): lo storico dei mesi passati non viene mai
  toccato
- il layout Hive (`month=YYYY-MM`) è standard e leggibile da qualunque tool
  (pandas, polars, DuckDB, Spark) senza dipendenze dal progetto

Concorrenza: le scritture sono serializzate da un lock di processo (RLock);
le letture non richiedono lock perché i file Parquet sono immutabili una volta
pubblicati (si scrive un file temporaneo nascosto e lo si rinomina atomicamente).
"""
import logging
import os
import shutil
import threading
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Lock di processo per le scritture (l'app è multithread con Streamlit).
# RLock: le operazioni composite (prezzi + metadati) acquisiscono il lock una
# sola volta e richiamano metodi interni che lo ri-acquisiscono.
_WRITE_LOCK = threading.RLock()

# Schema canonico della tabella prezzi (colonne e ordine fissi)
PRICES_COLUMNS = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']

_PRICES_SCHEMA = pa.schema([
    ('date', pa.timestamp('ns')),
    ('ticker', pa.string()),
    ('open', pa.float64()),
    ('high', pa.float64()),
    ('low', pa.float64()),
    ('close', pa.float64()),
    ('volume', pa.float64()),
])

# Metadati piccoli (una riga per ticker): un file Parquet ciascuno
_META_ARROW_SCHEMAS = {
    'updates': pa.schema([
        ('ticker', pa.string()),
        ('name', pa.string()),
        ('last_updated', pa.string()),   # data ISO 'YYYY-MM-DD'
    ]),
    'failed_tickers': pa.schema([
        ('ticker', pa.string()),
        ('reason', pa.string()),
        ('first_failed', pa.string()),   # data ISO 'YYYY-MM-DD'
        ('last_failed', pa.string()),    # data ISO 'YYYY-MM-DD'
        ('attempts', pa.int64()),
    ]),
}

# Prefisso dei file temporanei: esclusi dalle scansioni dataset
_TMP_PREFIX = '.tmp'

# Massimo numero di ticker per espressione di filtro (parità con _SQL_CHUNK_SIZE)
_FILTER_CHUNK = 500


class ParquetStore:
    """Store Parquet dei prezzi con partizioni mensili e metadati affiancati.

    Tutti i metodi pubblici sono thread-safe: le scritture serializzano sul
    lock di processo, le letture sfruttano l'immutabilità dei file pubblicati.
    """

    def __init__(self, store_dir: str):
        self.store_dir = str(store_dir)
        self.prices_dir = os.path.join(self.store_dir, 'prices')
        self.meta_dir = os.path.join(self.store_dir, '_meta')
        os.makedirs(self.prices_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        self._cleanup_tmp_files()

    # ------------------------------------------------------------------
    # Primitive di basso livello
    # ------------------------------------------------------------------

    @property
    def write_lock(self) -> threading.RLock:
        """Lock delle scritture: usabile anche dai chiamanti per operazioni
        composite (prezzi + metadati) da tenere insieme."""
        return _WRITE_LOCK

    def _partition_dir(self, month_key: str) -> str:
        return os.path.join(self.prices_dir, f'month={month_key}')

    def _dataset(self) -> ds.Dataset:
        """Dataset pyarrow delle partizioni mensili (colonne fisiche + `month`)."""
        partitioning = ds.partitioning(pa.schema([('month', pa.string())]),
                                       flavor='hive')
        return ds.dataset(self.prices_dir, format='parquet',
                          partitioning=partitioning,
                          ignore_prefixes=[_TMP_PREFIX])

    def _cleanup_tmp_files(self) -> None:
        """Rimuove file temporanei rimasti da scritture interrotte (crash)."""
        with _WRITE_LOCK:
            for root, _dirs, files in os.walk(self.store_dir):
                for f in files:
                    if f.startswith(_TMP_PREFIX):
                        try:
                            os.remove(os.path.join(root, f))
                        except OSError:
                            pass

    def _has_data(self) -> bool:
        """True se c'è almeno una partizione mensile con dati. Su uno store
        vuoto lo schema del dataset contiene solo il campo di partizione:
        le scansioni con filtro su colonna vengono cortocircuitate."""
        return bool(self.month_partitions())

    @staticmethod
    def _canonical_prices(df: pd.DataFrame) -> pd.DataFrame:
        """Normalizza un frame OHLCV nello schema canonico dello store:
        colonne [date, ticker, open, ...], date datetime64[ns], dedup (ticker, date)."""
        out = df.copy()
        out['date'] = pd.to_datetime(out['date']).astype('datetime64[ns]')
        out['ticker'] = out['ticker'].astype(str)
        for c in ('open', 'high', 'low', 'close', 'volume'):
            if c not in out.columns:
                out[c] = 0.0
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out = out.dropna(subset=['date'])
        out = out.drop_duplicates(subset=['ticker', 'date'], keep='last')
        return out[PRICES_COLUMNS]

    def _read_partition(self, month_key: str) -> Optional[pd.DataFrame]:
        """Legge una partizione mensile (None se assente/vuota)."""
        part_dir = self._partition_dir(month_key)
        if not os.path.isdir(part_dir):
            return None
        dset = ds.dataset(part_dir, format='parquet',
                          ignore_prefixes=[_TMP_PREFIX])
        tbl = dset.to_table(columns=PRICES_COLUMNS)
        if tbl.num_rows == 0:
            return None
        return tbl.to_pandas()

    def _write_partition(self, month_key: str, df: pd.DataFrame) -> None:
        """Scrive (sovrascrive) una partizione mensile in modo atomico:
        file temporaneo nascosto + rename. Se `df` è vuoto rimuove la partizione."""
        part_dir = self._partition_dir(month_key)
        if df is None or len(df) == 0:
            if os.path.isdir(part_dir):
                shutil.rmtree(part_dir)
                logger.debug(f"Partizione month={month_key} rimossa (vuota).")
            return
        os.makedirs(part_dir, exist_ok=True)
        df = df.sort_values(['ticker', 'date'], kind='stable')[PRICES_COLUMNS]
        table = pa.Table.from_pandas(df, schema=_PRICES_SCHEMA,
                                     preserve_index=False)
        tmp_path = os.path.join(part_dir, f'{_TMP_PREFIX}-data.parquet')
        final_path = os.path.join(part_dir, 'data.parquet')
        pq.write_table(table, tmp_path, compression='zstd')
        if os.path.exists(final_path):
            os.remove(final_path)
        os.replace(tmp_path, final_path)

    # ------------------------------------------------------------------
    # Prezzi: upsert incrementale / replace / letture
    # ------------------------------------------------------------------

    def merge_prices(self, new_df: pd.DataFrame) -> int:
        """Upsert incrementale: aggiorna/aggiunge le righe di `new_df` riscrivendo
        SOLO le partizioni mensili coinvolte (gli altri mesi non vengono toccati).
        Le righe con (ticker, date) già presenti vengono sovrascritte.
        Ritorna il numero di righe processate."""
        if new_df is None or new_df.empty:
            return 0
        with _WRITE_LOCK:
            canonical = self._canonical_prices(new_df)
            if canonical.empty:
                return 0
            canonical = canonical.assign(
                _month=canonical['date'].dt.strftime('%Y-%m'))
            written = 0
            for month_key, part in canonical.groupby('_month', sort=True):
                existing = self._read_partition(month_key)
                if existing is None:
                    combined = part
                else:
                    combined = pd.concat([existing, part[PRICES_COLUMNS]],
                                         ignore_index=True)
                    combined = combined.drop_duplicates(subset=['ticker', 'date'],
                                                        keep='last')
                self._write_partition(month_key, combined)
                written += len(part)
            return written

    def replace_prices(self, new_df: pd.DataFrame) -> int:
        """SOSTITUISCE lo storico dei ticker presenti in `new_df` (DELETE + INSERT)
        riscrivendo solo le partizioni toccate: i mesi con sole righe dei ticker
        sostituiti vengono ripuliti (o rimossi se vuoti). Ritorna le righe scritte."""
        if new_df is None or new_df.empty:
            return 0
        with _WRITE_LOCK:
            canonical = self._canonical_prices(new_df)
            if canonical.empty:
                return 0
            replaced = set(canonical['ticker'].unique())
            canonical = canonical.assign(
                _month=canonical['date'].dt.strftime('%Y-%m'))
            new_months = set(canonical['_month'])
            old_months = self._months_with_tickers(replaced)
            for month_key in sorted(new_months | old_months):
                part_new = canonical.loc[canonical['_month'] == month_key,
                                         PRICES_COLUMNS]
                existing = self._read_partition(month_key)
                if existing is not None and not existing.empty:
                    existing = existing[~existing['ticker'].isin(replaced)]
                if existing is None or existing.empty:
                    combined = part_new
                else:
                    combined = pd.concat([existing, part_new], ignore_index=True)
                self._write_partition(month_key, combined)
            return len(canonical)

    def _months_with_tickers(self, tickers: set) -> set:
        """Mesi in cui almeno uno dei ticker indicati ha già righe nello store."""
        if not tickers or not self._has_data():
            return set()
        months = set()
        ticker_list = list(tickers)
        dset = self._dataset()
        for i in range(0, len(ticker_list), _FILTER_CHUNK):
            chunk = ticker_list[i:i + _FILTER_CHUNK]
            tbl = dset.to_table(columns=['ticker', 'date'],
                                filter=ds.field('ticker').isin(chunk))
            if tbl.num_rows:
                df = tbl.to_pandas()
                months.update(df['date'].dt.strftime('%Y-%m').unique())
        return months

    def _scan_ticker_dates(self, tickers: Optional[List[str]] = None) -> Optional[pa.Table]:
        """Tabella arrow [ticker, date]: filtrata per i ticker richiesti
        (a chunk) oppure completa. None se non ci sono righe."""
        if not self._has_data():
            return None
        dset = self._dataset()
        if not tickers:
            tbl = dset.to_table(columns=['ticker', 'date'])
            return tbl if tbl.num_rows else None
        tables = []
        for i in range(0, len(tickers), _FILTER_CHUNK):
            chunk = list(tickers)[i:i + _FILTER_CHUNK]
            tbl = dset.to_table(columns=['ticker', 'date'],
                                filter=ds.field('ticker').isin(chunk))
            if tbl.num_rows:
                tables.append(tbl)
        return pa.concat_tables(tables) if tables else None

    def load_frames(self, tickers: List[str]) -> pd.DataFrame:
        """Righe OHLCV dei ticker richiesti ( filtro a chunk per universi grandi)."""
        if not tickers or not self._has_data():
            return pd.DataFrame(columns=PRICES_COLUMNS)
        dset = self._dataset()
        frames = []
        for i in range(0, len(tickers), _FILTER_CHUNK):
            chunk = list(tickers)[i:i + _FILTER_CHUNK]
            tbl = dset.to_table(columns=PRICES_COLUMNS,
                                filter=ds.field('ticker').isin(chunk))
            if tbl.num_rows:
                frames.append(tbl.to_pandas())
        if not frames:
            return pd.DataFrame(columns=PRICES_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def last_dates(self, tickers: List[str]) -> Dict[str, pd.Timestamp]:
        """{ticker: ultima data disponibile} per i ticker richiesti."""
        if not tickers:
            return {}
        tbl = self._scan_ticker_dates(list(tickers))
        if tbl is None:
            return {}
        agg = tbl.group_by('ticker').aggregate([('date', 'max')])
        return {r.ticker: pd.Timestamp(r.date_max)
                for r in agg.to_pandas().itertuples(index=False)}

    def ticker_row_stats(self, tickers: Optional[List[str]] = None
                         ) -> Dict[str, Tuple[int, pd.Timestamp, pd.Timestamp]]:
        """{ticker: (righe, prima data, ultima data)} per i ticker richiesti
        (o per tutti, se tickers è None)."""
        tbl = self._scan_ticker_dates(tickers)
        if tbl is None:
            return {}
        agg = tbl.group_by('ticker').aggregate(
            [('date', 'count'), ('date', 'min'), ('date', 'max')])
        out = {}
        for r in agg.to_pandas().itertuples(index=False):
            out[r.ticker] = (int(r.date_count),
                             pd.Timestamp(r.date_min),
                             pd.Timestamp(r.date_max))
        return out

    def tickers_with_min_rows(self, min_rows: int) -> List[str]:
        """Ticker con almeno `min_rows` righe di prezzi."""
        if not self._has_data():
            return []
        tbl = self._dataset().to_table(columns=['ticker'])
        if not tbl.num_rows:
            return []
        agg = tbl.group_by('ticker').aggregate([('ticker', 'count')])
        df = agg.to_pandas()
        sel = df.loc[df['ticker_count'] >= int(min_rows), 'ticker']
        return sorted(sel.tolist())

    def distinct_tickers(self) -> List[str]:
        """Tutti i ticker distinti presenti nelle partizioni dei prezzi."""
        if not self._has_data():
            return []
        tbl = self._dataset().to_table(columns=['ticker'])
        if not tbl.num_rows:
            return []
        return sorted(pc.unique(tbl.column('ticker')).to_pylist())

    def count_rows(self) -> int:
        """Righe totali (conteggio dai metadati Parquet, senza decodificare)."""
        try:
            return self._dataset().count_rows()
        except Exception:
            return 0

    def month_partitions(self) -> List[str]:
        """Chiavi dei mesi presenti ('YYYY-MM', ordinate)."""
        try:
            return sorted(d[len('month='):] for d in os.listdir(self.prices_dir)
                          if d.startswith('month='))
        except OSError:
            return []

    def stats(self) -> dict:
        """Statistiche globali dello store (per la UI Gestione Dati)."""
        updates = self._read_meta('updates')
        failed = self._read_meta('failed_tickers')
        last_sync = None
        if not updates.empty:
            lu = updates['last_updated']
            if lu.notna().any():
                last_sync = str(lu.max())
        size_bytes = 0
        for root, _dirs, files in os.walk(self.store_dir):
            for f in files:
                try:
                    size_bytes += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return {
            'tickers': len(self.distinct_tickers()),
            'rows': self.count_rows(),
            'last_sync': last_sync,
            'failed': int(len(failed)),
            'db_mb': size_bytes / (1024 * 1024),
            'partitions': len(self.month_partitions()),
        }

    # ------------------------------------------------------------------
    # Metadati: registro aggiornamenti (updates)
    # ------------------------------------------------------------------

    def _meta_path(self, name: str) -> str:
        return os.path.join(self.meta_dir, f'{name}.parquet')

    def _read_meta(self, name: str) -> pd.DataFrame:
        """Legge un file di metadati (df vuoto con le colonne attese se assente).
        Un file corrotto propaga l'errore: meglio fallire che credere vuoto."""
        path = self._meta_path(name)
        cols = _META_ARROW_SCHEMAS[name].names
        if not os.path.exists(path):
            return pd.DataFrame(columns=cols)
        df = pd.read_parquet(path)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols]

    def _write_meta(self, name: str, df: pd.DataFrame) -> None:
        """Scrive (sovrascrive) un file di metadati in modo atomico."""
        schema = _META_ARROW_SCHEMAS[name]
        cols = schema.names
        if df is None or df.empty:
            df = pd.DataFrame(columns=cols)
        else:
            df = df[cols].copy()
        if name == 'failed_tickers':
            df['attempts'] = (pd.to_numeric(df['attempts'], errors='coerce')
                              .fillna(0).astype('int64'))
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        tmp_path = self._meta_path(name) + f'{_TMP_PREFIX}.parquet'
        pq.write_table(table, tmp_path, compression='zstd')
        os.replace(tmp_path, self._meta_path(name))

    def read_updates(self) -> pd.DataFrame:
        """Tabella updates: {ticker, name, last_updated}."""
        return self._read_meta('updates')

    def upsert_updates(self, updates: Dict[str, Tuple[Optional[str], str]]) -> None:
        """Upsert nel registro aggiornamenti.

        updates: {ticker: (name, last_updated ISO)}. name None/vuoto mantiene
        il nome già presente (stessa semantica del COALESCE SQL)."""
        if not updates:
            return
        with _WRITE_LOCK:
            df = self._read_meta('updates')
            df = df.set_index('ticker') if not df.empty else \
                pd.DataFrame(columns=['name', 'last_updated'])
            tickers = list(updates.keys())
            new_name = pd.Series(
                [(updates[t][0] or None) if updates[t][0] else None
                 for t in tickers], index=tickers, dtype='object')
            new_date = pd.Series([updates[t][1] for t in tickers],
                                 index=tickers, dtype='object')
            if not df.empty:
                old_name = df['name'].reindex(tickers)
                coalesced = new_name.where(new_name.notna() & (new_name != ''),
                                           old_name)
                keep = df.loc[~df.index.isin(tickers)]
            else:
                coalesced = new_name
                keep = df
            block = pd.DataFrame({'name': coalesced, 'last_updated': new_date})
            block.index.name = 'ticker'
            out = pd.concat([keep, block]) if len(keep) else block
            self._write_meta('updates', out.reset_index())

    # ------------------------------------------------------------------
    # Metadati: cache negativa (failed_tickers)
    # ------------------------------------------------------------------

    def read_failed(self) -> pd.DataFrame:
        """Tabella failed_tickers: {ticker, reason, first_failed, last_failed, attempts}."""
        return self._read_meta('failed_tickers')

    def record_failed(self, tickers: List[str], reason: str, today: str) -> None:
        """Registra i ticker senza dati: nuovo inserimento o incremento dei tentativi."""
        if not tickers:
            return
        with _WRITE_LOCK:
            df = self._read_meta('failed_tickers')
            df = df.set_index('ticker') if not df.empty else \
                pd.DataFrame(columns=['reason', 'first_failed', 'last_failed',
                                      'attempts'])
            for t in dict.fromkeys(tickers):
                if not df.empty and t in df.index:
                    df.at[t, 'reason'] = reason
                    df.at[t, 'last_failed'] = today
                    try:
                        attempts = int(df.at[t, 'attempts'])
                    except (TypeError, ValueError):
                        attempts = 0
                    df.at[t, 'attempts'] = attempts + 1
                else:
                    df.loc[t] = [reason, today, today, 1]
            df.index.name = 'ticker'
            self._write_meta('failed_tickers', df.reset_index())

    def upsert_failed_rows(self, df: pd.DataFrame) -> None:
        """Upsert di righe complete di failed_tickers (usato dalle migrazioni:
        conserva first_failed/attempts della fonte)."""
        if df is None or df.empty:
            return
        cols = _META_ARROW_SCHEMAS['failed_tickers'].names
        new = df[cols].drop_duplicates(subset='ticker', keep='last')
        new = new.set_index('ticker')
        with _WRITE_LOCK:
            cur = self._read_meta('failed_tickers')
            cur = cur.set_index('ticker') if not cur.empty else \
                pd.DataFrame(columns=cols[1:])
            keep = cur.loc[~cur.index.isin(new.index)] if not cur.empty else cur
            out = pd.concat([keep, new]) if len(keep) else new
            out.index.name = 'ticker'
            self._write_meta('failed_tickers', out.reset_index())

    def remove_failed(self, tickers: Optional[List[str]] = None) -> int:
        """Rimuove dalla cache negativa i ticker indicati (o tutti).
        Ritorna il numero di righe rimosse."""
        with _WRITE_LOCK:
            df = self._read_meta('failed_tickers')
            if df.empty:
                return 0
            if tickers is None:
                removed = len(df)
                self._write_meta('failed_tickers', df.iloc[0:0])
                return removed
            mask = df['ticker'].isin(set(tickers))
            removed = int(mask.sum())
            if removed:
                self._write_meta('failed_tickers', df[~mask])
            return removed
