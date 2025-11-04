# -*- coding: utf-8 -*-
import re
from typing import Dict, Optional, Tuple
import logging
from etf_metrics.shared.config import BENCHMARK_KEYWORDS_TO_PROXY, TICKER_TO_PROXY, ISIN_TO_BENCHMARK
from etf_metrics.clients.yahoo_client import get_info

logger = logging.getLogger(__name__)

# ============================================================
# DEFAULT BENCHMARK AZIONARIO GLOBALE
# - Indice: MSCI ACWI (sviluppati + emergenti)
# - Proxy Yahoo Finance: iShares MSCI ACWI ETF -> "ACWI"
# Se preferisci MSCI World, puoi impostare:
# DEFAULT_EQUITY_BENCHMARK_PROXY_TICKER = "URTH" (iShares MSCI World ETF)
# DEFAULT_EQUITY_BENCHMARK_NAME = "MSCI World"
# Oppure FTSE Global All Cap (proxy "VT")
# ============================================================
DEFAULT_EQUITY_BENCHMARK_PROXY_TICKER = "ACWI"
DEFAULT_EQUITY_BENCHMARK_NAME = "MSCI ACWI"


def lookup_proxy_for_benchmark(
    bench_name: Optional[str],
    etf_ticker: str,
    etf_info: Dict
) -> Tuple[Optional[str], Optional[str]]:
    """
    Trova un proxy (ticker Yahoo) per il benchmark di un ETF, utilizzando una logica
    a cascata che privilegia la precisione. Se non viene trovato nulla, imposta un
    benchmark azionario globale di default (MSCI ACWI, proxy ACWI).

    Args:
        bench_name (Optional[str]): Nome del benchmark dell'ETF (può provenire da UI).
        etf_ticker (str): Ticker dell'ETF.
        etf_info (Dict): Informazioni aggiuntive sull'ETF.

    Returns:
        Tuple[Optional[str], Optional[str]]: (ticker proxy del benchmark, nome del benchmark).
    """
    etf_isin = etf_info.get("isin")
    if etf_isin and etf_isin in ISIN_TO_BENCHMARK:
        proxy_ticker, benchmark_name = ISIN_TO_BENCHMARK[etf_isin]
        logger.info(f"Benchmark trovato da mappatura manuale per {etf_isin}: {proxy_ticker} -> {benchmark_name}")
        return proxy_ticker, benchmark_name

    proxy_found: Optional[str] = None
    final_bench_name: Optional[str] = bench_name
    if bench_name:
        low = bench_name.lower()
        for key, proxy in BENCHMARK_KEYWORDS_TO_PROXY.items():
            if key in low:
                proxy_found = proxy
                logger.debug(f"Proxy individuato da bench_name '{bench_name}' con keyword '{key}': {proxy_found}")
                break

    if not proxy_found and etf_ticker:
        base = re.sub(r"\.[A-Z]+$", "", etf_ticker.upper())
        for k, proxy in TICKER_TO_PROXY.items():
            if k in base:
                proxy_found = proxy
                logger.debug(f"Proxy individuato da ticker ETF '{etf_ticker}' (base '{base}') con chiave '{k}': {proxy_found}")
                break

    if not proxy_found:
        text = " ".join(
            str(etf_info.get(k, ""))
            for k in ("fundFamily", "category", "longName", "shortName", "longBusinessSummary")
        ).lower()
        for key, proxy in BENCHMARK_KEYWORDS_TO_PROXY.items():
            if key in text:
                proxy_found = proxy
                logger.debug(f"Proxy individuato da descrizione ETF con keyword '{key}': {proxy_found}")
                if not final_bench_name:
                    final_bench_name = key.title()
                break

    if proxy_found and not final_bench_name:
        try:
            proxy_info = get_info(ticker=proxy_found)
            final_bench_name = (
                proxy_info.get("longName")
                or proxy_info.get("shortName")
                or bench_name
            )
            logger.debug(f"Nome benchmark migliorato da Yahoo per '{proxy_found}': '{final_bench_name}'")
        except Exception as e:
            logger.warning(f"Impossibile recuperare info Yahoo per '{proxy_found}': {e}")

    if not proxy_found:
        proxy_found = DEFAULT_EQUITY_BENCHMARK_PROXY_TICKER
        if not final_bench_name:
            final_bench_name = DEFAULT_EQUITY_BENCHMARK_NAME
        logger.info(f"Nessun benchmark rilevato. Uso default globale: {proxy_found} -> {final_bench_name}")

        try:
            proxy_info = get_info(ticker=proxy_found)
            _ = proxy_info.get("longName") or proxy_info.get("shortName")
        except Exception as e:
            logger.debug(f"Skip miglioramento nome per default '{proxy_found}': {e}")

    return proxy_found, final_bench_name