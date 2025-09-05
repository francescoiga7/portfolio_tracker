# -*- coding: utf-8 -*-
from typing import Dict, List, Optional
import re
import yfinance as yf
from .yahoo_client import resolve_isin_one
from .justetf_client import fetch_justetf_page, parse_etf_details_from_html, parse_ter_from_html

def get_etf_extended_info(isin: str) -> Dict:
    """Recupera informazioni estese su un ETF con approccio universale"""
    info = {}

    # Prima prova a recuperare da Yahoo Finance
    yf_info = get_yahoo_info(isin)
    if yf_info:
        info.update(yf_info)

    # Poi prova con JustETF (più robusto via BeautifulSoup)
    try:
        html = fetch_justetf_page(isin, timeout=15)
        if html:
            jt = parse_etf_details_from_html(html)
            # aggiorna solo i campi mancanti o "Sconosciuto"
            for k in ("category", "fund_size", "replication_method", "distribution", "provider"):
                v = jt.get(k)
                if v and (k not in info or not info[k] or info[k] == "Sconosciuto"):
                    info[k] = v
    except Exception as e:
        print(f"Errore nel recupero informazioni JustETF per {isin}: {e}")

    # Infine, se ancora mancano informazioni, usa fallback leggeri
    if "category" not in info or not info["category"] or info["category"] == "Sconosciuto":
        info["category"] = get_category_from_name(info.get("longName", "") or info.get("shortName", ""))

    # Migliora inferenza "Distribuzione" dai nomi se non presente
    if "distribution" not in info or not info["distribution"] or info["distribution"] == "Sconosciuto":
        nm = f"{info.get('longName', '')} {info.get('shortName', '')}".lower()
        if re.search(r"\bacc(?:umulating)?\b", nm):
            info["distribution"] = "Ad accumulazione"
        elif re.search(r"\bdist(?:ributing|ribuzione)?\b", nm):
            info["distribution"] = "A distribuzione"
        else:
            info["distribution"] = "Sconosciuto"

    if "fund_size" not in info and "totalAssets" in info:
        info["fund_size"] = f"{info['totalAssets'] / 1e9:.2f} Miliardi USD"

    # Aggiungi alternative con tracking difference migliore (dinamico)
    info["alternative_etfs"] = get_better_td_alternatives(isin, info.get("category", ""))

    return info

def get_yahoo_info(isin: str) -> Dict:
    """Recupera informazioni da Yahoo Finance"""
    info = {}
    try:
        ticker = resolve_isin_one(isin)
        if ticker:
            yf_ticker = yf.Ticker(ticker)
            yf_info = yf_ticker.info

            info["provider"] = yf_info.get("fundFamily", "Sconosciuto")
            info["category"] = yf_info.get("category", "Sconosciuto")

            # Formatta la dimensione del fondo
            total_assets = yf_info.get("totalAssets")
            if total_assets:
                if total_assets > 1e9:
                    info["fund_size"] = f"{total_assets / 1e9:.2f} Miliardi USD"
                else:
                    info["fund_size"] = f"{total_assets / 1e6:.2f} Milioni USD"

            # Evita di dedurre la distribuzione solo dal "Yield"
            long_short = f"{yf_info.get('longName','')} {yf_info.get('shortName','')}".lower()
            if re.search(r"\bacc(?:umulating)?\b", long_short):
                info["distribution"] = "Ad accumulazione"
            elif re.search(r"\bdist(?:ributing)?\b", long_short):
                info["distribution"] = "A distribuzione"

    except Exception as e:
        print(f"Errore nel recupero informazioni Yahoo per {isin}: {e}")

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
            f = float(v)
            if f <= 0.05:
                f *= 100.0
            if 0.0 < f < 3.0:
                return round(f, 4)
        except Exception:
            continue
    return None