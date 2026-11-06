# -*- coding: utf-8 -*-
"""Config & costanti per il progetto ETF Metrics.

Contenuto:
  - File di stato / persistenza: PORTFOLIO_FILE (portafogli salvati),
    ALGO_STATE_FILE (stato del portafoglio algoritmico live),
    ETF_UNIVERSE_FILE (universo ETF scoperto via discovery Yahoo, con
    ticker .DE preferiti: alimenta la pagina Satelliti PAC) e
    PARQUET_STORE_DIR (directory dello store Parquet dei prezzi, partizioni
    mensili).
  - resolve_market_db_path(): risoluzione a runtime del backend dei prezzi.
  - PREFERRED_SUFFIXES: preferenze per i suffissi dei ticker Yahoo;
    MANUAL_ISIN_MAP: mappatura manuale ISIN -> ticker candidati.
  - FAILED_TICKER_RETRY_DAYS: cache negativa dei ticker senza dati
    ("no data found"). None = mai più scaricati (blacklist permanente);
    un numero N concede un nuovo tentativo dopo N giorni (utile per ETF
    appena quotati). Reset sempre possibile con
    MarketDataManager.clear_failed_tickers().
  - Download massivo (universi grandi, es. tutto Xetra):
    BULK_DOWNLOAD_BATCH_SIZE (ticker per chiamata yf.download, batch
    isolati e ripristinabili), BULK_DOWNLOAD_THREADS (thread paralleli per
    batch), BULK_INFO_WORKERS (thread paralleli per il recupero di
    nomi/ISIN via get_info), BULK_BATCH_PAUSE (secondi di pausa tra i
    batch: riduce il rischio di 429; 0 = off) e YAHOO_RATE_LIMIT_COOLDOWN
    (quando Yahoo risponde 429 i download vengono sospesi per questi
    secondi invece di martellare il server; durante il cooldown nessun
    ticker finisce in blacklist; default 10 minuti).
  - Discovery universo ETF: DEFAULT_SEED_QUERIES, query generate a matrice
    (asset class × geografia, temi, fattori, emittenti, commodity).
  - FAMOUS_PORTFOLIOS: portafogli modello per il backtesting.
"""
from typing import Dict, List, Tuple
import logging
import os

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = "saved_portfolio.json"
ALGO_STATE_FILE = "live_algo_portfolio.json"
ETF_UNIVERSE_FILE = "etf_universe.json"

PARQUET_STORE_DIR = "market_data_parquet"


def resolve_market_db_path() -> str:
    """Percorso dello store dei prezzi, risolto a RUNTIME (non a import).

    Precedenza:
    1. variabile d'ambiente ETF_METRICS_DB (scelta esplicita: directory
       Parquet oppure file .db SQLite legacy)
    2. market_data_parquet/, se esiste (backend Parquet: è il backend di
       default, formato colonnare aperto con partizioni mensili)
    3. market_data.db (SQLite, legacy: usato solo se è l'unico DB presente
       e lo store Parquet non è ancora stato creato)
    4. market_data_parquet nuovo (installazioni da zero: il Gestore Dati
       crea e popola direttamente lo store Parquet)

    La risoluzione a runtime permette il cambio di backend senza riavviare
    l'app (basta un refresh della pagina Streamlit).
    """
    explicit = os.environ.get("ETF_METRICS_DB")
    if explicit:
        return explicit
    if os.path.isdir(PARQUET_STORE_DIR):
        return PARQUET_STORE_DIR
    if os.path.exists("market_data.db"):
        return "market_data.db"
    return PARQUET_STORE_DIR


PREFERRED_SUFFIXES = [".DE", ".MI", ".AS", ".L", ""]

MANUAL_ISIN_MAP: Dict[str, List[str]] = {
    "IE00BK5BQT80": ["VWCE.MI", "VWCE.DE", "VWRA.L"],
    "IE00BKM4GZ66": ["EIMI.MI", "EMIM.AS", "EIMI.L"],
    "IE00BDBRDM35": ["AGGH.MI", "AGGH.AS", "AGGH.L"],
    "IE00B3RBWM25": ["VDEV.MI", "VHVG.DE"],
    "IE00B3YCGJ38": ["SWDA.MI", "IWDA.AS"],
}

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; etf-metrics/1.0)"}

FAILED_TICKER_RETRY_DAYS = None

BULK_DOWNLOAD_BATCH_SIZE = 200
BULK_DOWNLOAD_THREADS = 16
BULK_INFO_WORKERS = 16
BULK_BATCH_PAUSE = 1.0
YAHOO_RATE_LIMIT_COOLDOWN = 600

asset_classes = ["Equity", "Government Bond", "Corporate Bond", "Aggregate Bond", "Commodity", "Real Estate"]
geographies = ["Global", "Europe", "USA", "Eurozone", "Emerging Markets", "Japan", "Pacific", "Germany", "UK", "China"]
themes_sectors = [
    "Technology", "Healthcare", "Financials", "Energy", "Clean Energy", "AI & Robotics", "Cybersecurity",
    "Digitalisation", "Water", "Uranium", "Nuclear", "Defense", "Infrastructure", "Semiconductors",
    "Biotechnology", "Automation", "ESG", "SRI", "Climate", "Megatrends", "Megatrend Equal Weight",
    "Aerospace", "Quantum Computing", "Innovation", "Artificial Intelligence & Robotics",
    "Active ETF", "Disruptive Innovation", "Genomics", "Space Exploration", "Metaverse", "Fintech",
    "Future of Food", "Next Generation Internet", "E-commerce", "Gaming & Esports"
]
factors = ["Value", "Growth", "Momentum", "Quality", "Minimum Volatility", "Size"]
providers = ["iShares", "Xtrackers", "Amundi", "Lyxor", "Invesco", "VanEck", "Vanguard", "SPDR", "HANetf", "ARK", "JPMorgan"]
specific_commodities = ["Gold ETC", "Silver ETC", "Oil ETC", "Bitcoin ETP", "Ethereum ETP", "Crypto ETP"]

generated_queries = set()
for ac in asset_classes:
    for geo in geographies:
        generated_queries.add(f'"{ac} {geo} UCITS ETF"')
for theme in themes_sectors:
    generated_queries.add(f'"{theme} UCITS ETF"')
for factor in factors:
    generated_queries.add(f'"{factor} Factor UCITS ETF"')
for provider in providers:
    generated_queries.add(f'"{provider} UCITS ETF"')
for comm in specific_commodities:
    generated_queries.add(f'"{comm}"')


DEFAULT_SEED_QUERIES: Tuple[str, ...] = tuple(sorted(list(generated_queries)))


FAMOUS_PORTFOLIOS = {
    "LifeStrategy 80 (LS80)": {
        "VTI": 0.40, "VEA": 0.24, "VWO": 0.16,
        "AGG": 0.14, "BNDX": 0.06
    },
    "LifeStrategy 60 (LS60)": {
        "VTI": 0.30, "VEA": 0.18, "VWO": 0.12,
        "AGG": 0.28, "BNDX": 0.12
    },
    "Classic 60/40": {
        "VTI": 0.60,
        "AGG": 0.40
    },
    "All-Weather (Bridgewater)": {
        "VTI": 0.30,
        "TLT": 0.40,
        "IEF": 0.15,
        "GLD": 0.075,
        "DBC": 0.075
    },
    "Golden Butterfly": {
        "VTI": 0.20,
        "SHY": 0.20,
        "TLT": 0.20,
        "GLD": 0.20,
        "VIOV": 0.20
    }
}
