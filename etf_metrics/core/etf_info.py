# -*- coding: utf-8 -*-
from typing import Dict, Optional
import re
import logging
import yfinance as yf

from etf_metrics.clients.yahoo_client import resolve_isin_one
from etf_metrics.clients.justetf_client import fetch_justetf_page, parse_etf_details_from_html, parse_ter_from_html

logger = logging.getLogger(__name__)

UNKNOWN = "Sconosciuto"
ACCUM_RE = re.compile(r"\bacc(?:umulating)?\b", re.IGNORECASE)
DIST_RE = re.compile(r"\bdist(?:ributing|ribuzione|inc)?\b", re.IGNORECASE)

def get_etf_extended_info(isin: str) -> Dict:
    """Recupera informazioni estese con fallback aggressivi."""
    info: Dict = {}

    yf_info = get_yahoo_info(isin)
    if yf_info:
        info.update(yf_info)

    try:
        html = fetch_justetf_page(isin, timeout=10)
        if html:
            jt = parse_etf_details_from_html(html)
            for k in ("category", "fund_size", "replication_method", "distribution", "provider"):
                v = jt.get(k)
                if v and (k not in info or not info[k] or info[k] == UNKNOWN):
                    info[k] = v
    except Exception:
        pass

    if info.get("category", UNKNOWN) == UNKNOWN:
        full_name = info.get("longName", "") or info.get("shortName", "")
        info["category"] = get_category_from_name(full_name)

    if info.get("distribution", UNKNOWN) == UNKNOWN:
        nm = f"{info.get('longName', '')} {info.get('shortName', '')}".lower()
        if ACCUM_RE.search(nm):
            info["distribution"] = "Ad accumulazione"
        elif DIST_RE.search(nm):
            info["distribution"] = "A distribuzione"
        else:
            if info.get('yield', 0) > 0.01:
                info["distribution"] = "A distribuzione (dedotto da yield)"
            else:
                info["distribution"] = UNKNOWN

    if "fund_size" not in info and "totalAssets" in info and info["totalAssets"]:
        try:
            ta = float(info["totalAssets"])
            if ta > 1e9:
                info["fund_size"] = f"{ta / 1e9:.2f} Mld USD"
            else:
                info["fund_size"] = f"{ta / 1e6:.2f} Mln USD"
        except Exception:
            pass

    return info

def get_yahoo_info(isin: str) -> Dict:
    info: Dict = {}
    try:
        ticker = resolve_isin_one(isin)
        if ticker:
            t = yf.Ticker(ticker)
            yf_data = t.info
            info["provider"] = yf_data.get("fundFamily", UNKNOWN)
            info["category"] = yf_data.get("category", UNKNOWN)
            info["longName"] = yf_data.get("longName")
            info["totalAssets"] = yf_data.get("totalAssets")
            info["yield"] = yf_data.get("yield", 0)
    except Exception:
        pass
    return info

def get_category_from_name(name: str) -> str:
    name_lower = name.lower()
    maps = {
        "Azionario Globale": ["world", "global", "acwi"],
        "Azionario USA": ["s&p", "nasdaq", "usa", "america"],
        "Azionario Europa": ["europe", "stoxx", "dax", "cac", "mib"],
        "Azionario Emergenti": ["emerging", "china", "india"],
        "Obbligazionario Gov": ["govt", "treasury", "bund", "btp"],
        "Obbligazionario Corp": ["corporate", "corp", "credit"],
        "Tech": ["tech", "robot", "cyber", "digital"],
        "Materie Prime": ["gold", "silver", "oil", "commodity"]
    }
    for cat, keys in maps.items():
        if any(k in name_lower for k in keys):
            return cat
    return "Azionario/Altro"

def fetch_ter_justetf(isin: str, timeout: int = 12) -> Optional[float]:
    html = fetch_justetf_page(isin, timeout=timeout)
    if not html: return None
    return parse_ter_from_html(html)

def fallback_ter_from_yahoo_info(etf_info: Dict) -> Optional[float]:
    keys = ["annualReportExpenseRatio", "annualExpenseRatio", "expenseRatio", "feesExpensesInvestment"]
    for k in keys:
        v = etf_info.get(k)
        if v is not None:
            try:
                val = float(v)
                return val * 100 if val < 0.1 else val
            except: continue
    return None