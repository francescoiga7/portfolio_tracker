import sqlite3
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional
import logging
import os

from etf_metrics.shared.config import MARKET_DATA_DB, FAILED_TICKER_RETRY_DAYS

logger = logging.getLogger(__name__)

# Numero massimo di parametri per query SQL (compatibilità con SQLite vecchi, limite 999)
_SQL_CHUNK_SIZE = 500


class MarketDataManager:
    def __init__(self, db_path=MARKET_DATA_DB):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Ottimizzazione commit: con WAL, synchronous=NORMAL è sicuro e molto più veloce
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        return conn

    def _init_db(self):
        """Inizializza le tabelle se non esistono e gestisce migrazioni schema semplici."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Modalità WAL: letture concorrenti durante le scritture + commit più rapidi
        # (persistente nel file DB, vale per tutte le connessioni future)
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
        except Exception as e:
            logger.debug(f"WAL non attivabile: {e}")

        # Tabella Updates (Registro Ticker)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS updates (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                last_updated DATE
            )
        ''')

        # Controllo se la colonna 'name' esiste (per chi ha già il DB creato)
        cursor.execute("PRAGMA table_info(updates)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'name' not in columns:
            try:
                cursor.execute("ALTER TABLE updates ADD COLUMN name TEXT")
                logger.info("Colonna 'name' aggiunta alla tabella updates.")
            except Exception as e:
                logger.warning(f"Impossibile aggiungere colonna name: {e}")

        # Tabella Prices (Dati Storici)
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

        # Tabella Failed Tickers (cache negativa: "no data found" su Yahoo)
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

    def get_tickers_needing_update(self, tickers: list) -> list:
        """Ritorna la lista dei ticker che NON sono stati aggiornati oggi,
        escludendo quelli in cache negativa ("no data found")."""
        if not tickers: return []

        today = date.today().isoformat()
        conn = self._get_conn()

        updated_set = set()
        try:
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                # Controlliamo solo la data, non se il nome esiste (lo aggiorneremo se necessario)
                query = f"SELECT ticker FROM updates WHERE last_updated = ? AND ticker IN ({placeholders})"
                params = [today] + chunk
                updated_df = pd.read_sql(query, conn, params=params)
                if not updated_df.empty:
                    updated_set.update(updated_df['ticker'].values)
        except Exception:
            updated_set = set()
        finally:
            conn.close()

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

    def save_bulk_data(self, data_dict: dict, names_dict: dict = None):
        """
        Salva i dati scaricati nel DB (SOSTITUISCE lo storico del ticker) e aggiorna
        la data di update e il nome.
        data_dict: { 'TICKER': pd.DataFrame, ... }
        names_dict: { 'TICKER': 'Nome Esteso', ... } (Opzionale)
        """
        if not data_dict: return

        conn = self._get_conn()
        today = date.today().isoformat()
        if names_dict is None: names_dict = {}

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

                asset_name = names_dict.get(ticker, None)

                # Upsert logica per la tabella updates
                conn.execute("""
                    INSERT INTO updates (ticker, name, last_updated)
                    VALUES (?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        last_updated=excluded.last_updated,
                        name=COALESCE(excluded.name, updates.name)
                """, (ticker, asset_name, today))

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

        conn = self._get_conn()
        today = date.today().isoformat()
        names_dict = names_dict or {}
        stats = {}

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

                conn.execute("""
                    INSERT INTO updates (ticker, name, last_updated)
                    VALUES (?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        last_updated=excluded.last_updated,
                        name=COALESCE(excluded.name, updates.name)
                """, (ticker, names_dict.get(ticker), today))

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
        conn = self._get_conn()
        out = {}
        try:
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                df = pd.read_sql(
                    f"SELECT ticker, MAX(date) AS last_date FROM prices "
                    f"WHERE ticker IN ({placeholders}) GROUP BY ticker",
                    conn, params=chunk)
                for _, r in df.iterrows():
                    if r['last_date']:
                        out[r['ticker']] = pd.to_datetime(r['last_date'])
        except Exception as e:
            logger.error(f"Errore get_last_dates: {e}")
        finally:
            conn.close()
        return out

    def get_tickers_updated_today(self) -> set:
        """Insieme dei ticker con last_updated = oggi nella tabella updates."""
        conn = self._get_conn()
        try:
            rows = pd.read_sql(
                "SELECT ticker FROM updates WHERE last_updated = ?",
                conn, params=[date.today().isoformat()])
            return set(rows['ticker'].values) if not rows.empty else set()
        except Exception as e:
            logger.error(f"Errore get_tickers_updated_today: {e}")
            return set()
        finally:
            conn.close()

    def get_tickers_missing_name(self, tickers: list) -> list:
        """Sottoinsieme dei ticker indicati che NON hanno un nome nel DB: sia i ticker
        assenti dalla tabella updates (nuovi) sia quelli con nome vuoto."""
        if not tickers:
            return []
        conn = self._get_conn()
        have_name = set()
        try:
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                df = pd.read_sql(
                    f"SELECT ticker FROM updates WHERE ticker IN ({placeholders}) "
                    f"AND name IS NOT NULL AND name != ''",
                    conn, params=chunk)
                if not df.empty:
                    have_name.update(df['ticker'].values)
        except Exception as e:
            logger.error(f"Errore get_tickers_missing_name: {e}")
        finally:
            conn.close()
        return [t for t in tickers if t not in have_name]

    def get_all_tickers(self) -> list:
        """Tutti i ticker noti al DB (dalla tabella updates, con fallback su prices)."""
        conn = self._get_conn()
        try:
            rows = pd.read_sql("SELECT ticker FROM updates", conn)
            if rows.empty:
                rows = pd.read_sql("SELECT DISTINCT ticker FROM prices", conn)
            return sorted(rows['ticker'].values) if not rows.empty else []
        except Exception as e:
            logger.error(f"Errore get_all_tickers: {e}")
            return []
        finally:
            conn.close()

    def get_all_tickers_info(self) -> pd.DataFrame:
        """Tabella riassuntiva del contenuto del DB: ticker, nome, righe, copertura
        date e data dell'ultimo aggiornamento (per la UI Gestione Dati)."""
        conn = self._get_conn()
        try:
            return pd.read_sql("""
                SELECT u.ticker, u.name, u.last_updated,
                       COUNT(p.date) AS rows,
                       MIN(p.date) AS first_date, MAX(p.date) AS last_date
                FROM updates u LEFT JOIN prices p ON p.ticker = u.ticker
                GROUP BY u.ticker, u.name, u.last_updated
                ORDER BY u.ticker
            """, conn)
        except Exception as e:
            logger.error(f"Errore get_all_tickers_info: {e}")            
            return pd.DataFrame()
        finally:
            conn.close()

    def get_db_stats(self) -> dict:
        """Statistiche globali del DB (per la UI Gestione Dati)."""
        conn = self._get_conn()
        try:
            tickers, rows = conn.execute(
                "SELECT COUNT(DISTINCT ticker), COUNT(*) FROM prices").fetchone()
            last_sync = conn.execute(
                "SELECT MAX(last_updated) FROM updates").fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM failed_tickers").fetchone()[0]
            try:
                size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
            except Exception:
                size_mb = 0.0
            return {
                'tickers': int(tickers or 0),
                'rows': int(rows or 0),
                'last_sync': last_sync,
                'failed': int(failed or 0),
                'db_mb': size_mb,
            }
        except Exception as e:
            logger.error(f"Errore get_db_stats: {e}")
            return {'tickers': 0, 'rows': 0, 'last_sync': None, 'failed': 0, 'db_mb': 0.0}
        finally:
            conn.close()

    def load_data(self, tickers: list) -> dict:
        """Carica i dati dal DB per i ticker richiesti (query a chunk per universi grandi)."""
        if not tickers: return {}

        conn = self._get_conn()
        try:
            frames = []
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                query = f"SELECT * FROM prices WHERE ticker IN ({placeholders})"
                frames.append(pd.read_sql(query, conn, params=chunk, parse_dates=['date']))
        finally:
            conn.close()

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

        conn = self._get_conn()
        rows = []
        try:
            for i in range(0, len(tickers), _SQL_CHUNK_SIZE):
                chunk = tickers[i:i + _SQL_CHUNK_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                query = f"SELECT ticker, name FROM updates WHERE ticker IN ({placeholders})"
                rows.extend(pd.read_sql(query, conn, params=chunk).to_dict('records'))
        except Exception as e:
            logger.error(f"Errore recupero nomi: {e}")
            return {t: t for t in tickers}
        finally:
            conn.close()

        # Crea dizionario, se il nome è nullo usa il ticker
        return {row['ticker']: (row['name'] if row['name'] else row['ticker']) for row in rows}

    # ------------------------------------------------------------------
    # Cache negativa: ticker senza dati su Yahoo ("no data found")
    # ------------------------------------------------------------------

    def record_failed_tickers(self, tickers: list, reason: str = "no data found") -> None:
        """Registra i ticker che non hanno prodotto dati, per non riscaricarli in futuro.
        Se un ticker è già presente, incrementa i tentativi e aggiorna la data."""
        if not tickers: return
        today = date.today().isoformat()
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
        conn = self._get_conn()
        try:
            if retry_days is None:
                rows = pd.read_sql("SELECT ticker FROM failed_tickers", conn)
            else:
                # Blacklist attiva solo se il fallimento è più recente della finestra di retry:
                # retry_days=0 -> nessun ticker è blacklistato (retry sempre)
                cutoff = (date.today() - timedelta(days=int(retry_days))).isoformat()
                rows = pd.read_sql(
                    "SELECT ticker FROM failed_tickers WHERE last_failed > ?", conn,
                    params=[cutoff])
            return set(rows['ticker'].values) if not rows.empty else set()
        except Exception as e:
            logger.error(f"Errore lettura failed_tickers: {e}")
            return set()
        finally:
            conn.close()

    def get_failed_tickers_info(self) -> pd.DataFrame:
        """Dettaglio dei ticker in blacklist (per UI/debug)."""
        conn = self._get_conn()
        try:
            return pd.read_sql(
                "SELECT ticker, reason, first_failed, last_failed, attempts "
                "FROM failed_tickers ORDER BY last_failed DESC, ticker", conn)
        except Exception as e:
            logger.error(f"Errore lettura failed_tickers: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def clear_failed_tickers(self, tickers: list = None) -> int:
        """Svuota la blacklist (tutti i ticker o solo quelli indicati). Ritorna il numero rimossi."""
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