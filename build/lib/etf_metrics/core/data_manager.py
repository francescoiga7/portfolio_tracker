import sqlite3
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional, Dict, List, Tuple
import logging
import os
import threading

from etf_metrics.shared.config import FAILED_TICKER_RETRY_DAYS, resolve_market_db_path

logger = logging.getLogger(__name__)

# Numero massimo di parametri per query SQL (compatibilità con SQLite vecchi, limite 999)
_SQL_CHUNK_SIZE = 500

# Lock globale per le scritture DuckDB: le connessioni allo stesso file sono
# coordinate dall'engine, ma le transazioni di scrittura vengono serializzate
# esplicitamente (l'app è multithread con Streamlit).
_DUCK_WRITE_LOCK = threading.Lock()


def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False


class MarketDataManager:
    """Gestore del DB dei prezzi con doppio backend: SQLite (legacy) o DuckDB.

    Il backend è scelto dal percorso del DB (estensione `.duckdb`) oppure, con
    il percorso di default, dal file presente su disco (`market_data.duckdb`
    ha la precedenza non appena esiste, es. dopo la migrazione). L'API pubblica
    è identica per entrambi i backend.

    Perché DuckDB: è un motore colonnare embedded che sulle letture analitiche
    (milioni di righe OHLCV, es. tutto Xetra) è 10-50x più veloce di SQLite e
    restituisce DataFrame pandas in modo nativo, senza la conversione riga per
    riga di `pd.read_sql`. La migrazione si fa con `scripts/migrate_to_duckdb.py`
    o dal pulsante dedicato nella pagina 📥 Gestione Dati.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = resolve_market_db_path()
        self.db_path = db_path
        if str(db_path).lower().endswith('.duckdb'):
            if not _duckdb_available():
                raise RuntimeError(
                    "Il DB dei prezzi è DuckDB ma il modulo 'duckdb' non è installato "
                    "(pip install duckdb). In alternativa rinomini/rimuova il file "
                    ".duckdb per tornare al DB SQLite.")
            self.backend = 'duckdb'
        else:
            self.backend = 'sqlite'
        self._init_db()

    # ------------------------------------------------------------------
    # Connessioni / primitive
    # ------------------------------------------------------------------

    def _get_conn(self):
        if self.backend == 'duckdb':
            import duckdb
            return duckdb.connect(self.db_path)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Ottimizzazione commit: con WAL, synchronous=NORMAL è sicuro e molto più veloce
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        return conn

    def _query_df(self, query: str, params: Optional[list] = None) -> pd.DataFrame:
        """SELECT -> DataFrame (nativo per DuckDB, pd.read_sql per SQLite)."""
        conn = self._get_conn()
        try:
            if self.backend == 'duckdb':
                return conn.execute(query, params or []).df()
            return pd.read_sql(query, conn, params=params)
        finally:
            conn.close()

    def _query_rows(self, query: str, params: Optional[list] = None) -> list:
        """SELECT -> lista di tuple."""
        conn = self._get_conn()
        try:
            return conn.execute(query, params or []).fetchall()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Inizializzazione schema
    # ------------------------------------------------------------------

    def _init_db(self):
        """Inizializza le tabelle se non esistono (schema identico per backend)."""
        if self.backend == 'duckdb':
            self._init_db_duckdb()
        else:
            self._init_db_sqlite()

    def _init_db_duckdb(self):
        import duckdb
        with _DUCK_WRITE_LOCK:
            conn = duckdb.connect(self.db_path)
            try:
                conn.execute('''CREATE TABLE IF NOT EXISTS updates (
                    ticker VARCHAR PRIMARY KEY, name VARCHAR, last_updated DATE)''')
                conn.execute('''CREATE TABLE IF NOT EXISTS prices (
                    date TIMESTAMP, ticker VARCHAR, open DOUBLE, high DOUBLE,
                    low DOUBLE, close DOUBLE, volume DOUBLE,
                    PRIMARY KEY (ticker, date))''')
                conn.execute('''CREATE TABLE IF NOT EXISTS failed_tickers (
                    ticker VARCHAR PRIMARY KEY, reason VARCHAR, first_failed DATE,
                    last_failed DATE, attempts INTEGER DEFAULT 1)''')
                conn.commit()
            finally:
                conn.close()

    def _init_db_sqlite(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        # Modalità WAL: letture concorrenti durante le scritture + commit più rapidi
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
        except Exception as e:
            logger.debug(f"WAL non attivabile: {e}")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS updates (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                last_updated DATE
            )
        ''')

        cursor.execute("PRAGMA table_info(updates)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'name' not in columns:
            try:
                cursor.execute("ALTER TABLE updates ADD COLUMN name TEXT")
                logger.info("Colonna 'name' aggiunta alla tabella updates.")
            except Exception as e:
                logger.warning(f"Impossibile aggiungere colonna name: {e}")

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

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON prices (ticker)')

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

    # ------------------------------------------------------------------
    # Registro aggiornamenti
    # ------------------------------------------------------------------

    def get_tickers_needing_update(self, tickers: list) -> list:
        """Ritorna la lista dei ticker che NON sono stati aggiornati oggi,
        escludendo quelli in cache negativa ("no data found")."""
        if not tickers: return []

        today = date.today().isoformat()
        date_eq = "last_updated = CAST(? AS DATE)" if self.backend == 'duckdb' else "last_updated = ?"
        updated_set = set()
        try:
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                query = (f"SELECT ticker FROM updates WHERE {date_eq} "
                         f"AND ticker IN ({placeholders})")
                updated_df = self._query_df(query, [today] + chunk)
                if not updated_df.empty:
                    updated_set.update(updated_df['ticker'].values)
        except Exception:
            updated_set = set()

        input_set = set(tickers)
        candidates = list(input_set - updated_set)

        # Cache negativa: salta i ticker che non hanno mai dato dati (policy di retry da config)
        failed = self.get_failed_tickers()
        if failed:
            candidates = [t for t in candidates if t not in failed]

        return candidates

    def _normalize_ohlcv_frame(self, ticker: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalizza un frame OHLCV (output yfinance) nello schema della tabella prices:
        colonne [date, ticker, open, high, low, close, volume], date come stringa ISO
        'YYYY-MM-DD HH:MM:SS' (stesso formato scritto da to_sql: serve per far
        combaciare la PRIMARY KEY (ticker, date) negli upsert incrementali)."""
        if df is None or df.empty:
            return None

        df_to_save = df.copy()

        # FIX MULTIINDEX/TUPLE ERROR:
        # Se yfinance restituisce un MultiIndex (es. ('Close', 'NVDA')), prendiamo l'ultimo livello
        if isinstance(df_to_save.columns, pd.MultiIndex):
            df_to_save.columns = df_to_save.columns.get_level_values(-1)

        if 'Date' not in df_to_save.columns:
            df_to_save = df_to_save.reset_index()

        # Pulisce i nomi colonne (rimuove eventuali residui di tuple e fa lowercase)
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

        # Formato data uniforme + dedup + ordinamento (robustezza per il merge)
        df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_save = df_to_save.drop_duplicates(subset=['date']).sort_values('date')
        return df_to_save

    def _upsert_updates(self, conn, ticker: str, asset_name: Optional[str], today: str):
        """Upsert della riga del ticker nella tabella updates.

        Nota SQL: SQLite non ha il tipo DATE (CAST(? AS DATE) troncherebbe la
        stringa ISO a intero!), quindi il parametro viene passato tale e quale;
        DuckDB invece esige il cast esplicito verso DATE.
        """
        if self.backend == 'duckdb':
            conn.execute("""
                INSERT INTO updates (ticker, name, last_updated)
                VALUES (?, ?, CAST(? AS DATE))
                ON CONFLICT(ticker) DO UPDATE SET
                    last_updated=excluded.last_updated,
                    name=COALESCE(excluded.name, updates.name)
            """, (ticker, asset_name, today))
        else:
            conn.execute("""
                INSERT INTO updates (ticker, name, last_updated)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    last_updated=excluded.last_updated,
                    name=COALESCE(excluded.name, updates.name)
            """, (ticker, asset_name, today))

    def save_bulk_data(self, data_dict: dict, names_dict: dict = None):
        """
        Salva i dati scaricati nel DB (SOSTITUISCE lo storico del ticker) e aggiorna
        la data di update e il nome.
        data_dict: { 'TICKER': pd.DataFrame, ... }
        names_dict: { 'TICKER': 'Nome Esteso', ... } (Opzionale)
        """
        if not data_dict: return

        today = date.today().isoformat()
        if names_dict is None: names_dict = {}

        if self.backend == 'duckdb':
            with _DUCK_WRITE_LOCK:
                conn = self._get_conn()
                try:
                    for ticker, df in data_dict.items():
                        df_to_save = self._normalize_ohlcv_frame(ticker, df)
                        if df_to_save is None or df_to_save.empty:
                            continue
                        conn.execute("DELETE FROM prices WHERE ticker = ?", [ticker])
                        conn.register("_bulk", df_to_save)
                        conn.execute(
                            "INSERT INTO prices SELECT CAST(date AS TIMESTAMP), ticker, "
                            "open, high, low, close, volume FROM _bulk")
                        conn.unregister("_bulk")
                        conn.execute("DELETE FROM failed_tickers WHERE ticker = ?", [ticker])
                        self._upsert_updates(conn, ticker, names_dict.get(ticker), today)
                    conn.commit()
                    logger.info(f"Salvati {len(data_dict)} ticker nel database (DuckDB).")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Errore salvataggio DB: {e}")
                finally:
                    conn.close()
            return

        # --- SQLite (percorso originale) ---
        conn = self._get_conn()
        try:
            for ticker, df in data_dict.items():
                df_to_save = self._normalize_ohlcv_frame(ticker, df)
                if df_to_save is None or df_to_save.empty:
                    continue

                # Rimuovi vecchi prezzi per questo ticker per evitare duplicati sporchi
                conn.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
                df_to_save.to_sql('prices', conn, if_exists='append', index=False)

                # Se un ticker precedentemente fallito ora ha dati, toglierlo dalla blacklist
                conn.execute("DELETE FROM failed_tickers WHERE ticker = ?", (ticker,))

                self._upsert_updates(conn, ticker, names_dict.get(ticker), today)

            conn.commit()
            logger.info(f"Salvati {len(data_dict)} ticker nel database.")

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

        if self.backend == 'duckdb':
            with _DUCK_WRITE_LOCK:
                conn = self._get_conn()
                try:
                    for ticker, df in data_dict.items():
                        norm = self._normalize_ohlcv_frame(ticker, df)
                        if norm is None or norm.empty:
                            continue

                        row = conn.execute(
                            "SELECT MAX(date) FROM prices WHERE ticker = ?", [ticker]).fetchone()
                        last_before = row[0] if row and row[0] else None
                        # confronto come stringa ISO (stesso formato delle righe normalizzate)
                        last_before_str = (last_before.strftime('%Y-%m-%d %H:%M:%S')
                                           if last_before else None)

                        conn.register("_bulk", norm)
                        conn.execute("""
                            INSERT INTO prices SELECT CAST(date AS TIMESTAMP), ticker,
                                open, high, low, close, volume FROM _bulk
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                open=excluded.open, high=excluded.high, low=excluded.low,
                                close=excluded.close, volume=excluded.volume""")
                        conn.unregister("_bulk")

                        conn.execute("DELETE FROM failed_tickers WHERE ticker = ?", [ticker])
                        self._upsert_updates(conn, ticker, names_dict.get(ticker), today)

                        cnt, first, last = conn.execute(
                            "SELECT COUNT(*), MIN(date), MAX(date) FROM prices WHERE ticker = ?",
                            [ticker]).fetchone()
                        new_rows = (int((norm['date'] > last_before_str).sum())
                                    if last_before_str else len(norm))
                        stats[ticker] = {
                            'new_rows': new_rows,
                            'total_rows': int(cnt or 0),
                            'first': first,
                            'last': last,
                        }
                    conn.commit()
                    logger.info(f"Merge incrementale di {len(stats)} ticker completato (DuckDB).")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Errore merge incrementale DB: {e}")
                finally:
                    conn.close()
            return stats

        # --- SQLite (percorso originale) ---
        conn = self._get_conn()
        try:
            for ticker, df in data_dict.items():
                norm = self._normalize_ohlcv_frame(ticker, df)
                if norm is None or norm.empty:
                    continue

                # Ultima data presente PRIMA del merge: serve a contare le righe nuove
                row = conn.execute(
                    "SELECT MAX(date) FROM prices WHERE ticker = ?", (ticker,)).fetchone()
                last_before = row[0] if row and row[0] else None

                # INSERT OR REPLACE: le date già presenti (es. ultima giornata parziale)
                # vengono aggiornate con il valore definitivo, le altre aggiunte
                conn.executemany(
                    "INSERT OR REPLACE INTO prices "
                    "(date, ticker, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    list(norm.itertuples(index=False, name=None)))

                # Un ticker prima fallito che ora produce dati viene riabilitato
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
        date_eq = "last_updated = CAST(? AS DATE)" if self.backend == 'duckdb' else "last_updated = ?"
        try:
            rows = self._query_df(f"SELECT ticker FROM updates WHERE {date_eq}", [today])
            return set(rows['ticker'].values) if not rows.empty else set()
        except Exception as e:
            logger.error(f"Errore get_tickers_updated_today: {e}")
            return set()

    def get_tickers_missing_name(self, tickers: list) -> list:
        """Sottoinsieme dei ticker indicati che NON hanno un nome nel DB: sia i ticker
        assenti dalla tabella updates (nuovi) sia quelli con nome vuoto."""
        if not tickers:
            return []
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
        date e data dell'ultimo aggiornamento (per la UI Gestione Dati)."""
        last_updated = ("CAST(u.last_updated AS VARCHAR)" if self.backend == 'duckdb'
                        else "u.last_updated")
        try:
            return self._query_df(f"""
                SELECT u.ticker, u.name, {last_updated} AS last_updated,
                       COUNT(p.date) AS rows,
                       MIN(p.date) AS first_date, MAX(p.date) AS last_date
                FROM updates u LEFT JOIN prices p ON p.ticker = u.ticker
                GROUP BY u.ticker, u.name, u.last_updated
                ORDER BY u.ticker
            """)
        except Exception as e:
            logger.error(f"Errore get_all_tickers_info: {e}")
            return pd.DataFrame()

    def get_db_stats(self) -> dict:
        """Statistiche globali del DB (per la UI Gestione Dati)."""
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
                'backend': self.backend,
                'path': self.db_path,
            }
        except Exception as e:
            logger.error(f"Errore get_db_stats: {e}")
            return {'tickers': 0, 'rows': 0, 'last_sync': None, 'failed': 0,
                    'db_mb': 0.0, 'backend': self.backend, 'path': self.db_path}

    def load_data(self, tickers: list) -> dict:
        """Carica i dati dal DB per i ticker richiesti (query a chunk per universi grandi)."""
        if not tickers: return {}

        frames = []
        try:
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                query = (f"SELECT date, ticker, open, high, low, close, volume "
                         f"FROM prices WHERE ticker IN ({placeholders})")
                if self.backend == 'duckdb':
                    conn = self._get_conn()
                    try:
                        frames.append(conn.execute(query, chunk).df())
                    finally:
                        conn.close()
                else:
                    conn = self._get_conn()
                    try:
                        frames.append(pd.read_sql(query, conn, params=chunk,
                                                  parse_dates=['date']))
                    finally:
                        conn.close()
        except Exception as e:
            logger.error(f"Errore load_data: {e}")
            return {}

        df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if df_all.empty:
            return {}

        result = {}
        # Ottimizzazione: raggruppamento pandas
        for ticker, group in df_all.groupby('ticker'):
            df_cleaned = group.set_index('date').sort_index()
            df_cleaned.columns = [c.capitalize() for c in df_cleaned.columns]
            if 'Ticker' in df_cleaned.columns: del df_cleaned['Ticker']
            result[ticker] = df_cleaned

        return result

    def get_ticker_names(self, tickers: list) -> dict:
        """Recupera un dizionario {ticker: name} dal database (query a chunk)."""
        if not tickers: return {}

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

        # Crea dizionario, se il nome è nullo/NaN usa il ticker
        # (NULL torna come NaN da DuckDB e da pd.read_sql)
        return {row['ticker']: (row['name'] if pd.notna(row['name']) and row['name']
                                else row['ticker']) for row in rows}

    # ------------------------------------------------------------------
    # Cache negativa: ticker senza dati su Yahoo ("no data found")
    # ------------------------------------------------------------------

    def record_failed_tickers(self, tickers: list, reason: str = "no data found") -> None:
        """Registra i ticker che non hanno prodotto dati, per non riscaricarli in futuro.
        Se un ticker è già presente, incrementa i tentativi e aggiorna la data."""
        if not tickers: return
        today = date.today().isoformat()
        if self.backend == 'duckdb':
            with _DUCK_WRITE_LOCK:
                conn = self._get_conn()
                try:
                    for t in tickers:
                        conn.execute("""
                            INSERT INTO failed_tickers (ticker, reason, first_failed, last_failed, attempts)
                            VALUES (?, ?, CAST(? AS DATE), CAST(? AS DATE), 1)
                            ON CONFLICT(ticker) DO UPDATE SET
                                last_failed=excluded.last_failed,
                                attempts=failed_tickers.attempts + 1,
                                reason=excluded.reason
                        """, (t, reason, today, today))
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Errore registrazione failed_tickers: {e}")
                finally:
                    conn.close()
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
        try:
            if retry_days is None:
                rows = self._query_df("SELECT ticker FROM failed_tickers")
            else:
                # Blacklist attiva solo se il fallimento è più recente della finestra di retry:
                # retry_days=0 -> nessun ticker è blacklistato (retry sempre)
                cutoff = (date.today() - timedelta(days=int(retry_days))).isoformat()
                cmp_sql = ("last_failed > CAST(? AS DATE)" if self.backend == 'duckdb'
                           else "last_failed > ?")
                rows = self._query_df(
                    f"SELECT ticker FROM failed_tickers WHERE {cmp_sql}", [cutoff])
            return set(rows['ticker'].values) if not rows.empty else set()
        except Exception as e:
            logger.error(f"Errore lettura failed_tickers: {e}")
            return set()

    def get_failed_tickers_info(self) -> pd.DataFrame:
        """Dettaglio dei ticker in blacklist (per UI/debug)."""
        try:
            return self._query_df(
                "SELECT ticker, reason, first_failed, last_failed, attempts "
                "FROM failed_tickers ORDER BY last_failed DESC, ticker")
        except Exception as e:
            logger.error(f"Errore lettura failed_tickers: {e}")
            return pd.DataFrame()

    def clear_failed_tickers(self, tickers: list = None) -> int:
        """Svuota la blacklist (tutti i ticker o solo quelli indicati). Ritorna il numero rimossi."""
        if self.backend == 'duckdb':
            with _DUCK_WRITE_LOCK:
                conn = self._get_conn()
                try:
                    if tickers:
                        removed = 0
                        for t in tickers:
                            # il rowcount di DuckDB vale -1: si conta prima di cancellare
                            n = conn.execute("SELECT COUNT(*) FROM failed_tickers "
                                             "WHERE ticker = ?", [t]).fetchone()[0]
                            conn.execute("DELETE FROM failed_tickers WHERE ticker = ?", [t])
                            removed += int(n or 0)
                        conn.commit()
                        return removed
                    n = conn.execute("SELECT COUNT(*) FROM failed_tickers").fetchone()[0]
                    conn.execute("DELETE FROM failed_tickers")
                    conn.commit()
                    return int(n or 0)
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Errore pulizia failed_tickers: {e}")
                    return 0
                finally:
                    conn.close()

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
