# -*- coding: utf-8 -*-
from typing import Dict, List, Optional
import re
import logging
import yfinance as yf

from .yahoo_client import resolve_isin_one
from .justetf_client import fetch_justetf_page, parse_etf_details_from_html, parse_ter_from_html

logger = logging.getLogger(__name__)

UNKNOWN = "Sconosciuto"
# Regex condivise per inferire la distribuzione (evita duplicazioni)
ACCUM_RE = re.compile(r"\bacc(?:umulating)?\b", re.IGNORECASE)
DIST_RE = re.compile(r"\bdist(?:ributing|ribuzione)?\b", re.IGNORECASE)


def get_etf_extended_info(isin: str) -> Dict:
    """Recupera informazioni estese su un ETF con approccio universale"""
    info: Dict = {}

    # 1) Yahoo Finance
    yf_info = get_yahoo_info(isin)
    if yf_info:
        info.update(yf_info)

    # 2) JustETF (parsing HTML più robusto)
    try:
        html = fetch_justetf_page(isin, timeout=15)
        if html:
            jt = parse_etf_details_from_html(html)
            # aggiorna solo i campi mancanti o "Sconosciuto"
            for k in ("category", "fund_size", "replication_method", "distribution", "provider"):
                v = jt.get(k)
                if v and (k not in info or not info[k] or info[k] == UNKNOWN):
                    info[k] = v
    except Exception as e:
        logger.debug(f"Errore nel recupero informazioni JustETF per {isin}: {e}")

    # 3) Fallback leggeri
    if "category" not in info or not info["category"] or info["category"] == UNKNOWN:
        info["category"] = get_category_from_name(info.get("longName", "") or info.get("shortName", ""))

    # Distribuzione dai nomi se non presente
    if "distribution" not in info or not info["distribution"] or info["distribution"] == UNKNOWN:
        nm = f"{info.get('longName', '')} {info.get('shortName', '')}".lower()
        if ACCUM_RE.search(nm):
            info["distribution"] = "Ad accumulazione"
        elif DIST_RE.search(nm):
            info["distribution"] = "A distribuzione"
        else:
            info["distribution"] = UNKNOWN

    # fund_size derivato se assente e abbiamo totalAssets
    if "fund_size" not in info and "totalAssets" in info and info["totalAssets"]:
        try:
            ta = float(info["totalAssets"])
            if ta > 1e9:
                info["fund_size"] = f"{ta / 1e9:.2f} Miliardi USD"
            else:
                info["fund_size"] = f"{ta / 1e6:.2f} Milioni USD"
        except Exception:
            pass
    # Placeholder per alternative ETF con TD migliore (richiede dati esterni)
    info["alternative_etfs"] = get_better_td_alternatives(isin, info.get("category", ""))

    return info


def get_yahoo_info(isin: str) -> Dict:
    """Recupera informazioni da Yahoo Finance"""
    info: Dict = {}
    try:
        ticker = resolve_isin_one(isin)
        if ticker:
            yf_ticker = yf.Ticker(ticker)
            yf_info = yf_ticker.info
            info["provider"] = yf_info.get("fundFamily", UNKNOWN)
            info["category"] = yf_info.get("category", UNKNOWN)
            info["morningStarOverallRating"] = yf_info.get("morningStarOverallRating", UNKNOWN)

            # Preserva anche totalAssets per derivazioni successive
            total_assets = yf_info.get("totalAssets")
            if total_assets is not None:
                info["totalAssets"] = total_assets
                # Formatta la dimensione del fondo
                try:
                    ta = float(total_assets)
                    if ta > 1e9:
                        info["fund_size"] = f"{ta / 1e9:.2f} Miliardi USD"
                    else:
                        info["fund_size"] = f"{ta / 1e6:.2f} Milioni USD"
                except Exception:
                    pass

            # Evita di dedurre la distribuzione solo dal "Yield": controlla nomi
            long_short = f"{yf_info.get('longName','')} {yf_info.get('shortName','')}".lower()
            if ACCUM_RE.search(long_short):
                info["distribution"] = "Ad accumulazione"
            elif DIST_RE.search(long_short):
                info["distribution"] = "A distribuzione"
    except Exception as e:
        logger.debug(f"Errore nel recupero informazioni Yahoo per {isin}: {e}")
    return info


def get_category_from_name(name: str) -> str:
    """Determina la categoria dal nome dell'ETF"""
    name_lower = name.lower()
    category_keywords = {
        "azionario": ["equity", "stock", "azionario", "index", "msci", "ftse", "stoxx"],
        "obbligazionario": ["bond", "obbligazionario", "treasury", "government", "corporate"],
        "commodity": ["commodity", "gold", "silver", "oil", "energy", "metals"],
        "settoriale": ["technology", "healthcare", "financial", "energy", "sector"],
        "reale": ["real estate", "property", "reit"],
        "monetario": ["money market", "cash", "liquidità"],
    }
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category.capitalize()
    return "Diversificato"


def get_better_td_alternatives(isin: str, category: str) -> List[Dict]:
    """Trova ETF alternativi nella stessa categoria con tracking difference migliore"""
    # In un'implementazione reale, questi dati verrebbero da un database
    # Per ora restituiamo una lista vuota
    return []


def fetch_ter_justetf(isin: str, timeout: int = 12) -> Optional[float]:
    html = fetch_justetf_page(isin, timeout=timeout)
    if not html:
        return None
    return parse_ter_from_html(html)


def fallback_ter_from_yahoo_info(etf_info: Dict) -> Optional[float]:
    keys = ["annualReportExpenseRatio", "annualExpenseRatio", "expenseRatio", "feesExpensesInvestment"]
    for k in keys:
        v = etf_info.get(k)
        if v is None:
            continue
        try:
            # Gestisci sia valori decimali (0.0025) sia stringhe con '%'
            s = str(v).strip().replace("%", "")
            f = float(s)
            if f <= 0.05:  # probabile valore frazionario
                f *= 100.0
            if 0.0 < f < 3.0:
                return round(f, 4)
        except Exception:
            continue
    return None