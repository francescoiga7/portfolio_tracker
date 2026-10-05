# -*- coding: utf-8 -*-
"""
Config & costanti per il progetto ETF Metrics.
"""
from typing import Dict, List, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# --- File di stato / persistenza ---
PORTFOLIO_FILE = "saved_portfolio.json"
ALGO_STATE_FILE = "live_algo_portfolio.json"


def resolve_market_db_path() -> str:
    """Percorso del DB dei prezzi, risolto a RUNTIME (non a import).

    Precedenza:
    1. variabile d'ambiente ETF_METRICS_DB (scelta esplicita)
    2. market_data.duckdb, se esiste e duckdb è installato (backend analitico,
       10-50x più veloce sulle letture: viene preferito appena disponibile,
       es. subito dopo la migrazione dalla pagina 📥 Gestione Dati)
    3. market_data.db (SQLite, legacy)

    La risoluzione a runtime permette il cambio di backend senza riavviare
    l'app (basta un refresh della pagina Streamlit).
    """
    explicit = os.environ.get("ETF_METRICS_DB")
    if explicit:
        return explicit
    if os.path.exists("market_data.duckdb"):
        try:
            import duckdb  # noqa: F401
            return "market_data.duckdb"
        except Exception:
            logger.warning("Trovato market_data.duckdb ma il modulo 'duckdb' non è "
                           "installato: si usa il DB SQLite. (pip install duckdb)")
    return "market_data.db"


# Percorso del DB dei prezzi (compatibilità: usare resolve_market_db_path()
# per leggere il valore aggiornato, es. dopo una migrazione a DuckDB)
MARKET_DATA_DB = resolve_market_db_path()

# Preferenze per i suffissi dei ticker Yahoo
PREFERRED_SUFFIXES = [".DE", ".MI", ".AS", ".L", ""]

# Mappatura manuale ISIN -> ticker candidati
MANUAL_ISIN_MAP: Dict[str, List[str]] = {
    "IE00BK5BQT80": ["VWCE.MI", "VWCE.DE", "VWRA.L"],
    "IE00BKM4GZ66": ["EIMI.MI", "EMIM.AS", "EIMI.L"],
    "IE00BDBRDM35": ["AGGH.MI", "AGGH.AS", "AGGH.L"],
    "IE00B3RBWM25": ["VDEV.MI", "VHVG.DE"],
    "IE00B3YCGJ38": ["SWDA.MI", "IWDA.AS"],
}

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; etf-metrics/1.0)"}

# --- Cache negativa dei ticker senza dati ("no data found") ---
# None = i ticker che non hanno dati non vengono MAI più scaricati (blacklist permanente).
# Imposta un numero N di giorni per concedere un nuovo tentativo dopo N giorni
# (utile per ETF appena quotati che potrebbero ottenere dati più avanti).
# È sempre possibile resettare la blacklist con MarketDataManager.clear_failed_tickers().
FAILED_TICKER_RETRY_DAYS = None

# --- Download massivo (universi grandi, es. tutto Xetra) ---
BULK_DOWNLOAD_BATCH_SIZE = 200   # ticker per chiamata yf.download (batch isolati e ripristinabili)
BULK_DOWNLOAD_THREADS = 16       # thread paralleli per batch
BULK_INFO_WORKERS = 16           # thread paralleli per il recupero di nomi/ISIN (get_info)
BULK_BATCH_PAUSE = 1.0           # secondi di pausa tra i batch: riduce il rischio di 429 (0 = off)
# Quando Yahoo risponde 429 (Too Many Requests) i download vengono sospesi per
# questo numero di secondi invece di martellare il server: durante il cooldown
# nessun ticker finisce in blacklist. Default: 10 minuti.
YAHOO_RATE_LIMIT_COOLDOWN = 600

# --- NUOVA STRATEGIA DI DISCOVERY A MATRICE ---
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

# Generazione delle query a matrice
generated_queries = set()
# 1. Asset Class x Geografia
for ac in asset_classes:
    for geo in geographies:
        generated_queries.add(f'"{ac} {geo} UCITS ETF"')
# 2. Tematici e Settoriali
for theme in themes_sectors:
    generated_queries.add(f'"{theme} UCITS ETF"')
# 3. Fattori
for factor in factors:
    generated_queries.add(f'"{factor} Factor UCITS ETF"')
# 4. Emittenti
for provider in providers:
    generated_queries.add(f'"{provider} UCITS ETF"')
# 5. Commodities Specifiche
for comm in specific_commodities:
    generated_queries.add(f'"{comm}"')


DEFAULT_SEED_QUERIES: Tuple[str, ...] = tuple(sorted(list(generated_queries)))


# Portafogli modello per backtesting
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
