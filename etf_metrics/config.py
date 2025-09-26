# -*- coding: utf-8 -*-
"""
Config & costanti per il progetto ETF Metrics.
"""
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Preferenze per i suffissi dei ticker Yahoo
PREFERRED_SUFFIXES = [".MI", ".DE", ".AS", ".L", ""]

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
    "IE000YYE6WK5": ("^GSPC", "S&P 500 Index"),  # SPDR S&P 500 UCITS ETF
    "IE00B4L5Y983": ("^GSPC", "S&P 500 Index"),  # Core S&P 500 UCITS ETF
    "IE00B3YCGJ38": ("IWDA.AS", "MSCI World Index"),  # Core MSCI World UCITS ETF
    "IE00B1YZSC51": ("IEUR.AS", "MSCI Europe Index"),  # Core MSCI Europe UCITS ETF
    "IE00BKM4GZ66": ("EIMI.L", "MSCI Emerging Markets IMI Index"),  # Core MSCI EM IMI UCITS ETF
    "IE00BDBRDM35": ("AGGG.L", "Bloomberg Global Aggregate Bond Index"),  # Core Global Aggregate Bond UCITS ETF
    "IE00B3F81409": ("AGGG.L", "Bloomberg Global Aggregate Bond Index"),  # Core Global Bond UCITS ETF
    "IE00B6R52259": ("GC=F", "Gold Spot Price"),  # Physical Gold ETC
    "IE00BFNM3K80": ("WSML.L", "MSCI World SRI Index"),  # MSCI World SRI UCITS ETF
    "IE00BN4Q0370": ("SUSW.L", "MSCI World ESG Universal Index"),  # MSCI World ESG Universal UCITS ETF
    "IE00BFNM3D14": ("SUAG.L", "Bloomberg MSCI Global Green Bond Index"),  # Global Green Bond UCITS ETF
    "LU0274208692": ("^GSPC", "S&P 500 Index"),  # db x-trackers S&P 500 UCITS ETF
    "IE00B0M62Q58": ("IUES.DE", "MSCI Europe Index"),  # iShares Core MSCI Europe UCITS ETF
    "IE00B1FZS467": ("CSPX.L", "S&P 500 Index"),  # iShares Core S&P 500 UCITS ETF
    "IE00B52VJ196": ("INRG.L", "S&P Global Clean Energy Index"),  # iShares Global Clean Energy UCITS ETF
    "IE00BYZK4552": ("RBOT.L", "ROBO Global Robotics Index"),  # iShares Automation & Robotics UCITS ETF
    "IE00B14X4N27": ("CSBGE0.L", "iBoxx EUR Sovereigns Eurozone Index"),  # iShares Core EUR Govt Bond UCITS ETF
}

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; etf-metrics/1.0)"}

# Periodi supportati (Yahoo Finance)
PERIODS_ALL = [
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "3y", "5y", "10y", "ytd", "max"
]

DEFAULT_SEED_QUERIES: Tuple[str, ...] = (
    # Azionari Globali e Regionali
    "MSCI World", "FTSE All-World", "Global Equity", "Developed Markets",
    "S&P 500", "NASDAQ 100", "STOXX Europe 600", "MSCI Europe",
    "MSCI Emerging Markets", "MSCI Japan", "FTSE 100", "DAX", "CAC 40", "FTSE MIB",

    # Tematici e Settoriali
    "Technology Sector", "Healthcare Sector", "Financial Sector", "Energy Sector",
    "Clean Energy", "AI & Robotics", "Cybersecurity", "Digitalisation", "Water",
    "Uranium", "Nuclear", "Defense", "Quantum Computing", "Infrastructure", "Property", "REIT",
    "ESG", "SRI", "Climate Change", "Semiconductors", "LifeStrategy"

    # Fattoriali
    "Value Factor", "Growth Factor", "Momentum Factor", "Quality Factor", "Minimum Volatility",

'''
    # Obbligazionari
    "Global Aggregate Bond", "Government Bond", "Treasury Bond", "Corporate Bond",
    "High Yield Bond", "Inflation-Linked Bond", "Green Bond", "Floating Rate Note",
    "Money Market", "Short Term",
'''

    # Commodities e Alternativi
    "Gold ETC", "Silver ETC", "Broad Commodities ETC", "Bitcoin ETP", "Ethereum ETP",
    "Real Estate", "Infrastructure",

    # Stili di investimento e provider
    "Vanguard LifeStrategy", "iShares", "Xtrackers", "Amundi", "Lyxor", "Invesco", "VanEck", "HANetf"

    # Aggiunta diretta di Ticker e ISIN specifici per garantirne la cattura
    "XEON", "IE00B3VTMJ91", "LU1650487413", "IE00BDBRDM35", "IWDE", "IE00BK5BQT80",
    "IE00B4L5Y983", "NDXH", "CSSX5E", "CSMIB", "BTCE", "VNGA80", "XQUI", "IE000YYE6WK5"
)


# Porffoli modello per backtesting
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