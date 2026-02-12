import sqlite3
import pandas as pd
import numpy as np
from datetime import date
import logging
import os

logger = logging.getLogger(__name__)


class MarketDataManager:
    def __init__(self, db_path="market_data.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        """Inizializza le tabelle se non esistono."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS updates (
                ticker TEXT PRIMARY KEY,
                last_updated DATE
            )
        ''')

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
        conn.commit()
        conn.close()

    def get_tickers_needing_update(self, tickers: list) -> list:
        """Ritorna la lista dei ticker che NON sono stati aggiornati oggi."""
        if not tickers: return []

        today = date.today().isoformat()
        conn = self._get_conn()

        placeholders = ','.join(['?'] * len(tickers))
        query = f"SELECT ticker FROM updates WHERE last_updated = ? AND ticker IN ({placeholders})"
        params = [today] + tickers

        try:
            updated_df = pd.read_sql(query, conn, params=params)
            updated_set = set(updated_df['ticker'].values) if not updated_df.empty else set()
        except Exception:
            updated_set = set()
        finally:
            conn.close()

        input_set = set(tickers)
        return list(input_set - updated_set)

    def save_bulk_data(self, data_dict: dict):
        """
        Salva i dati scaricati nel DB e aggiorna la data di update.
        data_dict: { 'TICKER': pd.DataFrame, ... }
        """
        if not data_dict: return

        conn = self._get_conn()
        today = date.today().isoformat()

        try:
            for ticker, df in data_dict.items():
                if df.empty: continue

                df_to_save = df.copy()
                if 'Date' not in df_to_save.columns:
                    df_to_save = df_to_save.reset_index()

                df_to_save.columns = [c.lower() for c in df_to_save.columns]
                df_to_save['ticker'] = ticker

                cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                for c in cols:
                    if c not in df_to_save.columns:
                        df_to_save[c] = 0.0

                df_to_save = df_to_save[cols]

                conn.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))

                df_to_save.to_sql('prices', conn, if_exists='append', index=False)

                conn.execute(
                    "INSERT OR REPLACE INTO updates (ticker, last_updated) VALUES (?, ?)",
                    (ticker, today)
                )

            conn.commit()
            logger.info(f"Salvati {len(data_dict)} ticker nel database.")

        except Exception as e:
            logger.error(f"Errore salvataggio DB: {e}")
            conn.rollback()
        finally:
            conn.close()

    def load_data(self, tickers: list) -> dict:
        """Carica i dati dal DB per i ticker richiesti."""
        if not tickers: return {}

        conn = self._get_conn()

        placeholders = ','.join(['?'] * len(tickers))
        query = f"SELECT * FROM prices WHERE ticker IN ({placeholders})"

        try:
            df_all = pd.read_sql(query, conn, params=tickers, parse_dates=['date'])
        finally:
            conn.close()

        if df_all.empty:
            return {}

        result = {}
        for ticker, group in df_all.groupby('ticker'):
            df_cleaned = group.set_index('date').sort_index()
            df_cleaned.columns = [c.capitalize() for c in df_cleaned.columns]
            if 'Ticker' in df_cleaned.columns: del df_cleaned['Ticker']
            result[ticker] = df_cleaned

        return result