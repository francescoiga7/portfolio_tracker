# -*- coding: utf-8 -*-
"""
Config & costanti per il progetto ETF Metrics.
"""
from typing import Dict, List
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
    "IE000U58J0M1": ("^GSPC", "S&P 500 Index"),  # iShares Core S&P 500 UCITS ETF USD (Acc)
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

# Nome commerciale -> nome indice "canonico"
BENCHMARK_NAME_MAP = {
    "FTSE All-World": "FTSE ALL WORLD NET TR",
    "FTSE All World": "FTSE ALL WORLD NET TR",
    "MSCI ACWI IMI": "MSCI ACWI IMI NET TR",
    "MSCI ACWI": "MSCI ACWI NET TR",
    "MSCI World": "MSCI WORLD NET TR",
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

DEFAULT_CSV_PATH = "etf_metrics_results.csv"

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
    "IE00B14X4N27": ("CSBGE0.L", "iBoxx EUR Sovereigns Eurozone Index"),  # iShares Core EUR Govt Bond UCITS ETF    "LU0480132876": ("VWCE", "MSCI Emerging Markets. L'indice MSCI Emerging Markets replica i titoli azionari dei mercati emergenti di tutto il mondo. &nbsp; L’indice di spesa complessiva (TER) dell'ETF è pari allo 0,15% annuo . L’ETF replica la performance dell’indice sottostante con replica a campionamento (acquistando solo i componenti più importanti dello stesso). I dividendi dell'ETF sono distribuiti agli investitori (Semestralmente). &nbsp; L’ETF UBS Core MSCI EM UCITS ETF USD dis è un ETF di dimensioni molto grandi con un patrimonio gestito pari a 1.750 mln di Euro . L’ETF è stato lanciato il 12 novembre 2010 ed ha domicilio fiscale in Lussemburgo . Mostra di più Mostra di meno MSCI Emerging Markets (19) Azioni (1654) Mercati emergenti (93) Documenti Scheda informativa (IT) KID (IT) Prospetto (EN) Relazione annuale (EN) Relazione semestrale (EN) Mostra di più Grafico Notizie ed analisi sugli ETF La difesa dell'Europa &ndash; Dopo le “notizie di pace” Big tech ed energia nucleare &ndash; Ecco le mosse audaci che stanno compiendo Dal retail all’istituzionale: &ndash; L’ascesa del basis trading Ulteriori Notizie Articoli più popolari I migliori broker online per investire in ETF in Italia ETF: che cosa sono? Semplice spiegazione sugli ETF Come generare una rendita passiva con gli ETF da dividendo Nozioni di base Data Indice MSCI Emerging Markets"),  # Auto-added via override
    "IE000U58J0M1": ("VWCE", "S&P Global Clean Energy Transition. L'indice S&P Global Clean Energy Transition replica i 30 titoli azionari più grandi e liquidi di tutto il mondo che sono impegnati nell'economia delle energie pulite. &nbsp; L’indice di spesa complessiva (TER) dell'ETF è pari allo 0,65% annuo . Il iShares Global Clean Energy Transition UCITS ETF USD (Acc) è l’ETF più economico che replica l'indice S&P Global Clean Energy Transition. L’ETF replica la performance dell’indice sottostante con replica fisica totale (acquistando tutti i componenti dello stesso). I dividendi dell'ETF sono accumulati e reinvestiti nell'ETF. &nbsp; L’ETF iShares Global Clean Energy Transition UCITS ETF USD (Acc) gestisce un patrimonio pari a 180 mln di Euro . L’ETF è stato lanciato il 23 febbraio 2022 ed ha domicilio fiscale in Irlanda . Mostra di più Mostra di meno S&amp;P Global Clean Energy Transition (2) Azioni (1654) Globale (523) Servizi di pubblica utilità (13) Energia pulita (20) Documenti Scheda informativa (IT) KID (IT) Prospetto (EN) Relazione annuale (EN) Relazione semestrale (EN) Mostra di più Grafico Notizie ed analisi sugli ETF La difesa dell'Europa &ndash; Dopo le “notizie di pace” Big tech ed energia nucleare &ndash; Ecco le mosse audaci che stanno compiendo Dal retail all’istituzionale: &ndash; L’ascesa del basis trading Ulteriori Notizie Articoli più popolari I migliori broker online per investire in ETF in Italia ETF: che cosa sono? Semplice spiegazione sugli ETF Come generare una rendita passiva con gli ETF da dividendo Nozioni di base Data Indice S&amp;P Global Clean Energy Transition"),  # Auto-added via override

}

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; etf-metrics/1.0)"}

# Periodi supportati (Yahoo Finance)
PERIODS_ALL = [
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "3y", "5y", "10y", "ytd", "max"
]


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