import sqlite3
import pandas as pd
from datetime import date, timedelta
from typing import Optional, Dict
import logging
import os
import threading

from etf_metrics.shared.config import (
    FAILED_TICKER_RETRY_DAYS,
    PARQUET_STORE_DIR,
    resolve_market_db_path,
)
from etf_metrics.core.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

# Numero massimo di parametri per query SQL (compatibilità con SQLite vecchi, limite 999)
_SQL_CHUNK_SIZE = 500

# Serializza la conversione one-shot del file DuckDB legacy (l'app è multithread)
_CONVERT_LOCK = threading.Lock()


class MarketDataManager:
    """Gestore del DB dei prezzi con doppio backend: Parquet (default) o SQLite (legacy).

    Backend scelto dal percorso (`.db`/`.sqlite` = SQLite, directory = Parquet).
    Un file `.duckdb` legacy viene convertito one-shot a Parquet al primo uso
    (originale rinominato `.pre-parquet.bak`). API identica per entrambi i
    backend. Layout e vantaggi dello store Parquet: vedi parquet_store.py.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = resolve_market_db_path()
        db_path = str(db_path)

        if db_path.lower().endswith('.duckdb'):
            # Legacy DuckDB: conversione one-shot allo store Parquet
            db_path = self._convert_legacy_duckdb(db_path)

        self.db_path = db_path
        if db_path.lower().endswith(('.db', '.sqlite', '.sqlite3')) \
                or os.path.isfile(db_path):
            self.backend = 'sqlite'
        else:
            self.backend = 'parquet'
            self._store = ParquetStore(db_path)
        self._init_db()

    def _convert_legacy_duckdb(self, db_path: str) -> str:
        """Converte un DB DuckDB legacy in Parquet (one-shot); ritorna il
        percorso dello store. L'originale viene rinominato
        `<nome>.duckdb.pre-parquet.bak` (dati intatti)."""
        if not os.path.exists(db_path):
            raise RuntimeError(
                "Il percorso del DB dei prezzi punta a un file DuckDB legacy che "
                "non esiste. Il backend DuckDB è stato sostituito da Parquet: punti "
                "la variabile d'ambiente ETF_METRICS_DB a una directory Parquet "
                "(o a un file .db SQLite), oppure rimuovila per usare il default.")
        from etf_metrics.core.store_migration import migrate_duckdb_to_parquet
        with _CONVERT_LOCK:
            dest = os.path.join(os.path.dirname(os.path.abspath(db_path)),
                                PARQUET_STORE_DIR)
            report = migrate_duckdb_to_parquet(db_path, dest)
            backup = db_path + '.pre-parquet.bak'
            try:
                if os.path.exists(backup):
                    os.remove(backup)
                os.rename(db_path, backup)
            except OSError as e:
                logger.warning(
                    f"Impossibile rinominare {db_path} in {backup}: {e}")
            logger.info(
                f"DB DuckDB legacy convertito in Parquet ({report['price_rows']:,} "
                f"righe, {report['tickers']} ticker) → {dest}")
            return dest

# Connessioni / primitive (backend SQLite legacy) ---

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Con WAL, synchronous=NORMAL e' sicuro e molto piu' veloce
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        return conn

    def _query_df(self, query: str, params: Optional[list] = None) -> pd.DataFrame:
        """SELECT SQLite -> DataFrame."""
        conn = self._get_conn()
        try:
            return pd.read_sql(query, conn, params=params)
        finally:
            conn.close()

    def _query_rows(self, query: str, params: Optional[list] = None) -> list:
        """SELECT SQLite -> lista di tuple."""
        conn = self._get_conn()
        try:
            return conn.execute(query, params or []).fetchall()
        finally:
            conn.close()

# Inizializzazione schema ---

    def _init_db(self):
        """Inizializza lo storage se non esiste (schema identico per backend)."""
        if self.backend == 'sqlite':
            self._init_db_sqlite()
        # Parquet: le directory vengono create da ParquetStore.__init__

    def _init_db_sqlite(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        # Modalita' WAL: letture concorrenti durante le scritture
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
        except Exception as e:
            logger.debug(f"WAL non attivabile: {e}")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS updates (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                last_updated DATE,
                first_added DATE
            )
        ''')

        cursor.execute("PRAGMA table_info(updates)")
        columns = [info[1] for info in cursor.fetchall()]
        for col, col_type in (('name', 'TEXT'), ('first_added', 'DATE')):
            if col not in columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE updates ADD COLUMN {col} {col_type}")
                    logger.info(f"Colonna '{col}' aggiunta alla tabella updates.")
                except Exception as e:
                    logger.warning(f"Impossibile aggiungere colonna {col}: {e}")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                date TIMESTAMP,
                ticker TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            )
        ''')

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON prices (ticker)")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failed_tickers (
                ticker TEXT PRIMARY KEY,
                reason TEXT,
                first_failed DATE,
                last_failed DATE,
                attempts INTEGER DEFAULT 1
            )
        ''')
        conn.commit()
        conn.close()

# Registro aggiornamenti ---

    def get_tickers_needing_update(self, tickers: list) -> list:
        """Ritorna la lista dei ticker che NON sono stati aggiornati oggi,
        escludendo quelli in cache negativa ("no data found")."""
        if not tickers: return []

        today = date.today().isoformat()
        updated_set = set()
        try:
            if self.backend == 'parquet':
                updates = self._store.read_updates()
                if not updates.empty:
                    updated_set = set(
                        updates.loc[updates['last_updated'] == today, 'ticker'])
            else:
                for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                    chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                    placeholders = ','.join(['?'] * len(chunk))
                    query = (f"SELECT ticker FROM updates WHERE last_updated = ? "
                             f"AND ticker IN ({placeholders})")
                    updated_df = self._query_df(query, [today] + chunk)
                    if not updated_df.empty:
                        updated_set.update(updated_df['ticker'].values)
        except Exception:
            updated_set = set()

        input_set = set(tickers)
        candidates = list(input_set - updated_set)

        # Cache negativa: salta i ticker senza dati (policy di retry da config)
        failed = self.get_failed_tickers()
        if failed:
            candidates = [t for t in candidates if t not in failed]

        return candidates

    def _normalize_ohlcv_frame(self, ticker: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalizza un frame OHLCV (output yfinance) nello schema della
        tabella prices: colonne [date, ticker, open, high, low, close, volume].

        Le date diventano stringhe ISO 'YYYY-MM-DD HH:MM:SS' (formato scritto
        da to_sql: fa combaciare la PRIMARY KEY (ticker, date) negli upsert
        SQLite; il backend Parquet riconverte in datetime prima di scrivere).
        """
        if df is None or df.empty:
            return None

        df_to_save = df.copy()

        # yfinance puo' restituire un MultiIndex (es. ('Close', 'NVDA')):
        # si tiene l'ultimo livello
        if isinstance(df_to_save.columns, pd.MultiIndex):
            df_to_save.columns = df_to_save.columns.get_level_values(-1)

        if 'Date' not in df_to_save.columns:
            df_to_save = df_to_save.reset_index()

        # Pulisce i nomi colonne (residui di tuple -> lowercase)
        new_cols = []
        for c in df_to_save.columns:
            col_str = str(c[0]) if isinstance(c, tuple) else str(c)
            new_cols.append(col_str.lower())
        df_to_save.columns = new_cols

        df_to_save['ticker'] = ticker

        cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c not in df_to_save.columns:
                df_to_save[c] = 0.0
        df_to_save = df_to_save[cols]

        # Data uniforme + dedup + ordinamento (robustezza per il merge)
        df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_save = df_to_save.drop_duplicates(subset=['date']).sort_values('date')
        return df_to_save

    @staticmethod
    def _frames_to_prices_df(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Concatena i frame normalizzati in un unico DataFrame prezzi con
        date datetime64 (input del backend Parquet)."""
        return pd.concat(
            [f.assign(date=pd.to_datetime(f['date'])) for f in frames.values()],
            ignore_index=True)

    def _upsert_updates(self, conn, ticker: str, asset_name: Optional[str], today: str):
        """Upsert della riga del ticker nella tabella updates (backend SQLite).
        first_added è scritto solo al primo INSERT, mai sovrascritto."""
        conn.execute("""
            INSERT INTO updates (ticker, name, last_updated, first_added)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                last_updated=excluded.last_updated,
                name=COALESCE(excluded.name, updates.name)
        """, (ticker, asset_name, today, today))

    def save_bulk_data(self, data_dict: dict, names_dict: dict = None):
        """Salva i dati nel DB SOSTITUENDO lo storico dei ticker e aggiornando
        data di update e nome.
        data_dict: {'TICKER': DataFrame}; names_dict: {'TICKER': 'Nome'} opzionale.
        """
        if not data_dict: return

        today = date.today().isoformat()
        if names_dict is None: names_dict = {}

        frames = {}
        for ticker, df in data_dict.items():
            norm = self._normalize_ohlcv_frame(ticker, df)
            if norm is not None and not norm.empty:
                frames[ticker] = norm
        if not frames:
            return

        if self.backend == 'parquet':
            try:
                new_df = self._frames_to_prices_df(frames)
                with self._store.write_lock:
                    self._store.replace_prices(new_df)
                    self._store.remove_failed(list(frames.keys()))
                    self._store.upsert_updates(
                        {t: (names_dict.get(t), today) for t in frames})
                logger.info(f"Salvati {len(frames)} ticker nello store Parquet "
                            f"({len(new_df):,} righe).")
            except Exception as e:
                logger.error(f"Errore salvataggio store Parquet: {e}")
            return

        # --- SQLite (percorso originale) ---
        conn = self._get_conn()
        try:
            for ticker, df in frames.items():
                # sostituzione: si rimuove lo storico del ticker per evitare duplicati
                conn.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
                df.to_sql('prices', conn, if_exists='append', index=False)

                # un ticker fallito che ora ha dati esce dalla blacklist
                conn.execute("DELETE FROM failed_tickers WHERE ticker = ?", (ticker,))

                self._upsert_updates(conn, ticker, names_dict.get(ticker), today)

            conn.commit()
            logger.info(f"Salvati {len(frames)} ticker nel database.")

        except Exception as e:
            logger.error(f"Errore salvataggio DB: {e}")
            conn.rollback()
        finally:
            conn.close()

    def merge_bulk_data(self, data_dict: dict, names_dict: dict = None) -> dict:
        """Upsert incrementale: aggiunge solo le righe nuove e aggiorna l'ultima,
        SENZA cancellare lo storico già presente (aggiornamento "delta").

        Ritorna {ticker: {'new_rows', 'total_rows', 'first', 'last'}} dove:
        - new_rows: righe con data successiva all'ultima già presente nel DB
        - total_rows/first/last: stato del ticker dopo il merge
        """
        if not data_dict:
            return {}

        today = date.today().isoformat()
        names_dict = names_dict or {}
        stats = {}

        frames = {}
        for ticker, df in data_dict.items():
            norm = self._normalize_ohlcv_frame(ticker, df)
            if norm is not None and not norm.empty:
                frames[ticker] = norm
        if not frames:
            return {}

        if self.backend == 'parquet':
            try:
                with self._store.write_lock:
                    # Ultima data presente PRIMA del merge: serve a contare le righe nuove
                    last_before = self._store.last_dates(list(frames.keys()))
                    # Upsert nelle sole partizioni mensili coinvolte
                    self._store.merge_prices(self._frames_to_prices_df(frames))
                    self._store.remove_failed(list(frames.keys()))
                    self._store.upsert_updates(
                        {t: (names_dict.get(t), today) for t in frames})
                    after = self._store.ticker_row_stats(list(frames.keys()))
                for t, norm in frames.items():
                    lb = last_before.get(t)
                    cnt, first, last = after.get(t, (0, None, None))
                    dates = pd.to_datetime(norm['date'])
                    new_rows = int((dates > lb).sum()) if lb is not None else len(norm)
                    stats[t] = {
                        'new_rows': new_rows,
                        'total_rows': int(cnt or 0),
                        'first': first,
                        'last': last,
                    }
                logger.info(
                    f"Merge incrementale di {len(stats)} ticker completato (Parquet).")
            except Exception as e:
                logger.error(f"Errore merge incrementale store Parquet: {e}")
            return stats

        # --- SQLite (legacy) ---
        conn = self._get_conn()
        try:
            for ticker, norm in frames.items():
                # ultima data presente PRIMA del merge: per contare le righe nuove
                row = conn.execute(
                    "SELECT MAX(date) FROM prices WHERE ticker = ?", (ticker,)).fetchone()
                last_before = row[0] if row and row[0] else None

                # INSERT OR REPLACE: le date gia' presenti (es. ultima giornata
                # parziale) vengono aggiornate col valore definitivo
                conn.executemany(
                    "INSERT OR REPLACE INTO prices "
                    "(date, ticker, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    list(norm.itertuples(index=False, name=None)))

                # un ticker fallito che ora produce dati viene riabilitato
                conn.execute("DELETE FROM failed_tickers WHERE ticker = ?", (ticker,))

                self._upsert_updates(conn, ticker, names_dict.get(ticker), today)

                cnt, first, last = conn.execute(
                    "SELECT COUNT(*), MIN(date), MAX(date) FROM prices WHERE ticker = ?",
                    (ticker,)).fetchone()
                new_rows = (int((norm['date'] > last_before).sum())
                            if last_before else len(norm))
                stats[ticker] = {
                    'new_rows': new_rows,
                    'total_rows': int(cnt or 0),
                    'first': first,
                    'last': last,
                }

            conn.commit()
            logger.info(f"Merge incrementale di {len(stats)} ticker completato.")

        except Exception as e:
            logger.error(f"Errore merge incrementale DB: {e}")
            conn.rollback()
        finally:
            conn.close()
        return stats

    def get_last_dates(self, tickers: list) -> dict:
        """{ticker: Timestamp dell'ultima data disponibile} per i ticker con dati nel DB
        (query a chunk per universi grandi). Base per gli aggiornamenti delta."""
        if not tickers:
            return {}
        if self.backend == 'parquet':
            try:
                return self._store.last_dates(tickers)
            except Exception as e:
                logger.error(f"Errore get_last_dates: {e}")
                return {}
        out = {}
        try:
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                df = self._query_df(
                    f"SELECT ticker, MAX(date) AS last_date FROM prices "
                    f"WHERE ticker IN ({placeholders}) GROUP BY ticker", chunk)
                for _, r in df.iterrows():
                    if r['last_date']:
                        out[r['ticker']] = pd.to_datetime(r['last_date'])
        except Exception as e:
            logger.error(f"Errore get_last_dates: {e}")
        return out

    def get_tickers_updated_today(self) -> set:
        """Insieme dei ticker con last_updated = oggi nella tabella updates."""
        today = date.today().isoformat()
        if self.backend == 'parquet':
            try:
                updates = self._store.read_updates()
                if updates.empty:
                    return set()
                return set(updates.loc[updates['last_updated'] == today, 'ticker'])
            except Exception as e:
                logger.error(f"Errore get_tickers_updated_today: {e}")
                return set()
        try:
            rows = self._query_df("SELECT ticker FROM updates WHERE last_updated = ?",
                                  [today])
            return set(rows['ticker'].values) if not rows.empty else set()
        except Exception as e:
            logger.error(f"Errore get_tickers_updated_today: {e}")
            return set()

    def get_tickers_missing_name(self, tickers: list) -> list:
        """Sottoinsieme dei ticker indicati che NON hanno un nome nel DB: sia i ticker
        assenti dalla tabella updates (nuovi) sia quelli con nome vuoto."""
        if not tickers:
            return []
        if self.backend == 'parquet':
            try:
                updates = self._store.read_updates()
                if updates.empty:
                    return list(tickers)
                have_name = set(
                    updates.loc[updates['name'].notna() & (updates['name'] != ''),
                                'ticker'])
            except Exception as e:
                logger.error(f"Errore get_tickers_missing_name: {e}")
                have_name = set()
            return [t for t in tickers if t not in have_name]
        have_name = set()
        try:
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                df = self._query_df(
                    f"SELECT ticker FROM updates WHERE ticker IN ({placeholders}) "
                    f"AND name IS NOT NULL AND name != ''", chunk)
                if not df.empty:
                    have_name.update(df['ticker'].values)
        except Exception as e:
            logger.error(f"Errore get_tickers_missing_name: {e}")
        return [t for t in tickers if t not in have_name]

    def get_all_tickers(self) -> list:
        """Tutti i ticker noti al DB (dalla tabella updates, con fallback su prices)."""
        if self.backend == 'parquet':
            try:
                updates = self._store.read_updates()
                if updates.empty:
                    return self._store.distinct_tickers()
                return sorted(updates['ticker'].tolist())
            except Exception as e:
                logger.error(f"Errore get_all_tickers: {e}")
                return []
        try:
            rows = self._query_df("SELECT ticker FROM updates")
            if rows.empty:
                rows = self._query_df("SELECT DISTINCT ticker FROM prices")
            return sorted(rows['ticker'].values) if not rows.empty else []
        except Exception as e:
            logger.error(f"Errore get_all_tickers: {e}")
            return []

    def get_tickers_with_min_rows(self, min_rows: int = 200) -> list:
        """Ticker con almeno `min_rows` righe di prezzi (storico sufficiente per
        scanner e backtest). Usato dal selettore universo della UI."""
        if self.backend == 'parquet':
            try:
                return self._store.tickers_with_min_rows(min_rows)
            except Exception as e:
                logger.error(f"Errore get_tickers_with_min_rows: {e}")
                return []
        try:
            rows = self._query_df(
                "SELECT ticker FROM prices GROUP BY ticker HAVING COUNT(*) >= ?",
                [int(min_rows)])
            return sorted(rows['ticker'].tolist()) if not rows.empty else []
        except Exception as e:
            logger.error(f"Errore get_tickers_with_min_rows: {e}")
            return []

    def get_all_tickers_info(self) -> pd.DataFrame:
        """Tabella riassuntiva del contenuto del DB: ticker, nome, righe, copertura
        date, prezzo dell'ultima barra, data dell'ultimo aggiornamento e data del
        primo inserimento nel DB (per la UI Gestione Dati)."""
        if self.backend == 'parquet':
            try:
                updates = self._store.read_updates()
                if updates.empty:
                    return pd.DataFrame(
                        columns=['ticker', 'name', 'first_added', 'last_updated',
                                 'rows', 'first_date', 'last_date', 'last_price'])
                price_stats = self._store.ticker_row_stats()
                last_prices = self._store.last_closes()
                rows = []
                for r in updates.sort_values('ticker').itertuples(index=False):
                    cnt, first, last = price_stats.get(r.ticker, (0, None, None))
                    rows.append({
                        'ticker': r.ticker,
                        'name': r.name if pd.notna(r.name) else None,
                        'first_added': (r.first_added
                                        if pd.notna(r.first_added) else None),
                        'last_updated': (r.last_updated
                                         if pd.notna(r.last_updated) else None),
                        'rows': cnt,
                        'first_date': first,
                        'last_date': last,
                        'last_price': (round(last_prices[r.ticker], 4)
                                       if r.ticker in last_prices else None),
                    })
                return pd.DataFrame(
                    rows, columns=['ticker', 'name', 'first_added', 'last_updated',
                                   'rows', 'first_date', 'last_date', 'last_price'])
            except Exception as e:
                logger.error(f"Errore get_all_tickers_info: {e}")
                return pd.DataFrame()
        try:
            return self._query_df("""
                SELECT u.ticker, u.name, u.first_added, u.last_updated,
                       COUNT(p.date) AS rows,
                       MIN(p.date) AS first_date, MAX(p.date) AS last_date,
                       (SELECT p2.close FROM prices p2 WHERE p2.ticker = u.ticker
                        ORDER BY p2.date DESC LIMIT 1) AS last_price
                FROM updates u LEFT JOIN prices p ON p.ticker = u.ticker
                GROUP BY u.ticker, u.name, u.first_added, u.last_updated
                ORDER BY u.ticker
            """)
        except Exception as e:
            logger.error(f"Errore get_all_tickers_info: {e}")
            return pd.DataFrame()

    def get_db_stats(self) -> dict:
        """Statistiche globali del DB (per la UI Gestione Dati)."""
        if self.backend == 'parquet':
            try:
                st = self._store.stats()
                return {
                    'tickers': st['tickers'], 'rows': st['rows'],
                    'last_sync': st['last_sync'], 'failed': st['failed'],
                    'db_mb': st['db_mb'], 'partitions': st['partitions'],
                    'backend': self.backend, 'path': self.db_path,
                }
            except Exception as e:
                logger.error(f"Errore get_db_stats: {e}")
                return {'tickers': 0, 'rows': 0, 'last_sync': None, 'failed': 0,
                        'db_mb': 0.0, 'partitions': 0,
                        'backend': self.backend, 'path': self.db_path}
        try:
            row = self._query_rows(
                "SELECT COUNT(DISTINCT ticker), COUNT(*) FROM prices")[0]
            tickers, rows = row[0], row[1]
            last_sync = self._query_rows("SELECT MAX(last_updated) FROM updates")[0][0]
            failed = self._query_rows("SELECT COUNT(*) FROM failed_tickers")[0][0]
            try:
                size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
            except Exception:
                size_mb = 0.0
            return {
                'tickers': int(tickers or 0),
                'rows': int(rows or 0),
                'last_sync': str(last_sync) if last_sync is not None else None,
                'failed': int(failed or 0),
                'db_mb': size_mb,
                'partitions': None,
                'backend': self.backend,
                'path': self.db_path,
            }
        except Exception as e:
            logger.error(f"Errore get_db_stats: {e}")
            return {'tickers': 0, 'rows': 0, 'last_sync': None, 'failed': 0,
                    'db_mb': 0.0, 'partitions': None,
                    'backend': self.backend, 'path': self.db_path}

    def load_data(self, tickers: list) -> dict:
        """Carica i dati dal DB per i ticker richiesti (query a chunk per universi grandi)."""
        if not tickers: return {}

        try:
            if self.backend == 'parquet':
                df_all = self._store.load_frames(tickers)
            else:
                frames = []
                for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                    chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                    placeholders = ','.join(['?'] * len(chunk))
                    query = (f"SELECT date, ticker, open, high, low, close, volume "
                             f"FROM prices WHERE ticker IN ({placeholders})")
                    conn = self._get_conn()
                    try:
                        frames.append(pd.read_sql(query, conn, params=chunk,
                                                  parse_dates=['date']))
                    finally:
                        conn.close()
                df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        except Exception as e:
            logger.error(f"Errore load_data: {e}")
            return {}

        if df_all is None or df_all.empty:
            return {}

        result = {}
        for ticker, group in df_all.groupby('ticker'):
            df_cleaned = group.set_index('date').sort_index()
            df_cleaned.columns = [c.capitalize() for c in df_cleaned.columns]
            if 'Ticker' in df_cleaned.columns: del df_cleaned['Ticker']
            result[ticker] = df_cleaned

        return result

    def get_ticker_names(self, tickers: list) -> dict:
        """Recupera un dizionario {ticker: name} dal database (query a chunk)."""
        if not tickers: return {}

        if self.backend == 'parquet':
            try:
                updates = self._store.read_updates()
                if updates.empty:
                    return {t: t for t in tickers}
                updates = updates[updates['ticker'].isin(set(tickers))]
                names = {}
                for r in updates.itertuples(index=False):
                    # nome nullo/NaN -> ticker
                    names[r.ticker] = r.name if (pd.notna(r.name) and r.name) else r.ticker
                return {t: names.get(t, t) for t in tickers}
            except Exception as e:
                logger.error(f"Errore recupero nomi: {e}")
                return {t: t for t in tickers}

        rows = []
        try:
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                query = f"SELECT ticker, name FROM updates WHERE ticker IN ({placeholders})"
                rows.extend(self._query_df(query, chunk).to_dict('records'))
        except Exception as e:
            logger.error(f"Errore recupero nomi: {e}")
            return {t: t for t in tickers}

        # nome nullo/NaN -> ticker
        return {row['ticker']: (row['name'] if pd.notna(row['name']) and row['name']
                                else row['ticker']) for row in rows}

# Cache negativa: ticker senza dati su Yahoo ("no data found") ---

    def record_failed_tickers(self, tickers: list, reason: str = "no data found") -> None:
        """Registra i ticker senza dati, per non riscaricarli. Se gia' presenti
        incrementa i tentativi e aggiorna la data."""
        if not tickers: return
        today = date.today().isoformat()
        if self.backend == 'parquet':
            try:
                self._store.record_failed(tickers, reason, today)
            except Exception as e:
                logger.error(f"Errore registrazione failed_tickers: {e}")
            return

        conn = self._get_conn()
        try:
            for t in tickers:
                conn.execute("""
                    INSERT INTO failed_tickers (ticker, reason, first_failed, last_failed, attempts)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(ticker) DO UPDATE SET
                        last_failed=excluded.last_failed,
                        attempts=attempts + 1,
                        reason=excluded.reason
                """, (t, reason, today, today))
            conn.commit()
        except Exception as e:
            logger.error(f"Errore registrazione failed_tickers: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_failed_tickers(self, retry_days=None) -> set:
        """Ritorna l'insieme dei ticker attualmente in blacklist.

        retry_days=None (default da config FAILED_TICKER_RETRY_DAYS): blacklist permanente.
        retry_days=N: i ticker falliti da più di N giorni NON sono più in blacklist
        (verranno ritentati al prossimo giro).
        """
        if retry_days is None:
            retry_days = FAILED_TICKER_RETRY_DAYS
        if self.backend == 'parquet':
            try:
                failed = self._store.read_failed()
                if failed.empty:
                    return set()
                if retry_days is None:
                    return set(failed['ticker'])
                # Blacklist attiva solo se il fallimento e' dentro la finestra
                # di retry: retry_days=0 -> retry sempre
                cutoff = (date.today() - timedelta(days=int(retry_days))).isoformat()
                return set(failed.loc[failed['last_failed'] > cutoff, 'ticker'])
            except Exception as e:
                logger.error(f"Errore lettura failed_tickers: {e}")
                return set()
        try:
            if retry_days is None:
                rows = self._query_df("SELECT ticker FROM failed_tickers")
            else:
                cutoff = (date.today() - timedelta(days=int(retry_days))).isoformat()
                rows = self._query_df(
                    "SELECT ticker FROM failed_tickers WHERE last_failed > ?",
                    [cutoff])
            return set(rows['ticker'].values) if not rows.empty else set()
        except Exception as e:
            logger.error(f"Errore lettura failed_tickers: {e}")
            return set()

    def get_failed_tickers_info(self) -> pd.DataFrame:
        """Dettaglio dei ticker in blacklist (per UI/debug)."""
        if self.backend == 'parquet':
            try:
                failed = self._store.read_failed()
                if failed.empty:
                    return failed
                return failed.sort_values(
                    ['last_failed', 'ticker'], ascending=[False, True])
            except Exception as e:
                logger.error(f"Errore lettura failed_tickers: {e}")
                return pd.DataFrame()
        try:
            return self._query_df(
                "SELECT ticker, reason, first_failed, last_failed, attempts "
                "FROM failed_tickers ORDER BY last_failed DESC, ticker")
        except Exception as e:
            logger.error(f"Errore lettura failed_tickers: {e}")
            return pd.DataFrame()

    def clear_failed_tickers(self, tickers: list = None) -> int:
        """Svuota la blacklist (tutti i ticker o solo quelli indicati). Ritorna il numero rimossi."""
        if self.backend == 'parquet':
            try:
                return self._store.remove_failed(tickers)
            except Exception as e:
                logger.error(f"Errore pulizia failed_tickers: {e}")
                return 0

        conn = self._get_conn()
        try:
            if tickers:
                removed = 0
                for t in tickers:
                    cur = conn.execute("DELETE FROM failed_tickers WHERE ticker = ?", (t,))
                    removed += cur.rowcount
                conn.commit()
                return removed
            else:
                cur = conn.execute("DELETE FROM failed_tickers")
                conn.commit()
                return cur.rowcount
        except Exception as e:
            logger.error(f"Errore pulizia failed_tickers: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
