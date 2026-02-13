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
        """Inizializza le tabelle se non esistono e gestisce migrazioni schema semplici."""
        conn = self._get_conn()
        cursor = conn.cursor()

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
        conn.commit()
        conn.close()

    def get_tickers_needing_update(self, tickers: list) -> list:
        """Ritorna la lista dei ticker che NON sono stati aggiornati oggi."""
        if not tickers: return []

        today = date.today().isoformat()
        conn = self._get_conn()

        placeholders = ','.join(['?'] * len(tickers))
        # Controlliamo solo la data, non se il nome esiste (lo aggiorneremo se necessario)
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

    def save_bulk_data(self, data_dict: dict, names_dict: dict = None):
        """
        Salva i dati scaricati nel DB e aggiorna la data di update e il nome.
        data_dict: { 'TICKER': pd.DataFrame, ... }
        names_dict: { 'TICKER': 'Nome Esteso', ... } (Opzionale)
        """
        if not data_dict: return

        conn = self._get_conn()
        today = date.today().isoformat()
        if names_dict is None: names_dict = {}

        try:
            for ticker, df in data_dict.items():
                if df.empty: continue

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

                # Rimuovi vecchi prezzi per questo ticker per evitare duplicati sporchi
                conn.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
                df_to_save.to_sql('prices', conn, if_exists='append', index=False)

                # Recupera il nome se presente, altrimenti cerca di mantenerlo o usa il ticker
                asset_name = names_dict.get(ticker, None)

                # Upsert logica per la tabella updates
                if asset_name:
                    conn.execute(
                        "INSERT OR REPLACE INTO updates (ticker, name, last_updated) VALUES (?, ?, ?)",
                        (ticker, asset_name, today)
                    )
                else:
                    # Se non abbiamo il nome nuovo, proviamo ad aggiornare solo la data preservando il nome se esiste
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
        # Ottimizzazione: raggruppamento pandas
        for ticker, group in df_all.groupby('ticker'):
            df_cleaned = group.set_index('date').sort_index()
            df_cleaned.columns = [c.capitalize() for c in df_cleaned.columns]
            if 'Ticker' in df_cleaned.columns: del df_cleaned['Ticker']
            result[ticker] = df_cleaned

        return result

    def get_ticker_names(self, tickers: list) -> dict:
        """Recupera un dizionario {ticker: name} dal database."""
        if not tickers: return {}

        conn = self._get_conn()
        placeholders = ','.join(['?'] * len(tickers))
        query = f"SELECT ticker, name FROM updates WHERE ticker IN ({placeholders})"

        try:
            df = pd.read_sql(query, conn, params=tickers)
            # Crea dizionario, se il nome è nullo usa il ticker
            return {row['ticker']: (row['name'] if row['name'] else row['ticker']) for _, row in df.iterrows()}
        except Exception as e:
            logger.error(f"Errore recupero nomi: {e}")
            return {t: t for t in tickers}
        finally:
            conn.close()