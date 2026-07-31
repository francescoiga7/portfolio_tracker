# -*- coding: utf-8 -*-
"""
Config & costanti per il progetto ETF Metrics.
"""
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# --- File di stato / persistenza ---
PORTFOLIO_FILE = "saved_portfolio.json"
ALGO_STATE_FILE = "live_algo_portfolio.json"
MARKET_DATA_DB = "market_data.db"

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
