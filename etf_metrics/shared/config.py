# -*- coding: utf-8 -*-
"""
Config & costanti per il progetto ETF Metrics.
"""
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

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

# Parole chiave -> proxy benchmark (ticker Yahoo)
BENCHMARK_KEYWORDS_TO_PROXY: Dict[str, str] = {
    "ftse all-world": "VWRA.L",
    "ftse all world": "VWRA.L",
    "msci acwi imi": "SSAC.L",
    "msci acwi": "ACWI",
    "msci world": "IWDA.AS",
    "ftse developed world": "VEVE.L",
    "s&p 500": "SPY",
    "sp 500": "SPY",
    "stoxx europe 600": "EXSA.DE",
    "euro stoxx 50": "EXW1.DE",
    "msci emerging markets imi": "EIMI.L",
    "msci emerging markets": "EIMI.L",
    "bloomberg global aggregate": "AGGG.L",
    "global aggregate hedged eur": "AGGH.MI",
    "global aggregate eur hedged": "AGGH.MI",
    "bloomberg barclays global aggregate": "AGGG.L",
    "treasury": "IEF",
    "inflation-linked": "TIP",
    "gold": "GC=F",
    "bitcoin": "BTC-USD",
    "commodities": "DBC",
}

# ETF noti -> proxy benchmark
TICKER_TO_PROXY: Dict[str, str] = {
    "VWCE": "VWRA.L",
    "VWRL": "VWRA.L",
    "SWDA": "IWDA.AS",
    "IWDA": "IWDA.AS",
    "EIMI": "EIMI.L",
    "AGGH": "AGGH.MI",
    "AGGG": "AGGG.L",
}

# Mappatura completa ISIN -> (ticker_benchmark, nome_benchmark)
ISIN_TO_BENCHMARK = {
    "IE00BK5BQT80": ("VWRA.L", "FTSE All-World Index"),
    "IE000YYE6WK5": ("^GSPC", "S&P 500 Index"),
    "IE00B4L5Y983": ("^GSPC", "S&P 500 Index"),
    "IE00B3YCGJ38": ("IWDA.AS", "MSCI World Index"),
    "IE00B1YZSC51": ("IEUR.AS", "MSCI Europe Index"),
    "IE00BKM4GZ66": ("EIMI.L", "MSCI Emerging Markets IMI Index"),
    "IE00BDBRDM35": ("AGGG.L", "Bloomberg Global Aggregate Bond Index"),
    "IE00B3F81409": ("AGGG.L", "Bloomberg Global Aggregate Bond Index"),
    "IE00B6R52259": ("GC=F", "Gold Spot Price"),
    "IE00BFNM3K80": ("WSML.L", "MSCI World SRI Index"),
    "IE00BN4Q0370": ("SUSW.L", "MSCI World ESG Universal Index"),
    "IE00BFNM3D14": ("SUAG.L", "Bloomberg MSCI Global Green Bond Index"),
    "LU0274208692": ("^GSPC", "S&P 500 Index"),
    "IE00B0M62Q58": ("IUES.DE", "MSCI Europe Index"),
    "IE00B1FZS467": ("CSPX.L", "S&P 500 Index"),
    "IE00B52VJ196": ("INRG.L", "S&P Global Clean Energy Index"),
    "IE00BYZK4552": ("RBOT.L", "ROBO Global Robotics Index"),
    "IE00B14X4N27": ("CSBGE0.L", "iBoxx EUR Sovereigns Eurozone Index"),
}

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; etf-metrics/1.0)"}

# Periodi supportati (Yahoo Finance)
PERIODS_ALL = [
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "3y", "5y", "10y", "ytd", "max"
]

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

TACTICAL_TRADING_SEED_QUERIES: Tuple[str, ...] = (
    # Big Cap & Tech Giants
    "Apple Inc.", "Microsoft Corporation", "NVIDIA Corporation", "Amazon.com, Inc.",
    "Alphabet Inc.", "Meta Platforms, Inc.", "Tesla, Inc.",
    # Liste dei più attivi (molto efficaci per la scoperta)
    "most active stocks NASDAQ",
    "most active stocks NYSE",
    "trending stocks USA",
    # Settori Chiave
    "Semiconductor stocks",
    "AI stocks",
    "Cloud computing stocks",
    "Cybersecurity stocks",
    "Biotechnology stocks",
    # Indici Principali (come fallback)
    "S&P 100",
    "NASDAQ 100",
    "Dow Jones Industrial Average"
)