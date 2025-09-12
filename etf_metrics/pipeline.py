# -*- coding: utf-8 -*-
import datetime as dt
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .yahoo_client import resolve_isin_one, get_series, get_info
from .metrics import compute_metrics_from_series, compute_sharpe_ratio
from .trackingdiff_client import fetch_tracking_difference
from .etf_info import get_etf_extended_info, fetch_ter_justetf, fallback_ter_from_yahoo_info
from .benchmark import lookup_proxy_for_benchmark

ND = "n.d."  # evita ripetizioni letterali della stringa "n.d."


def compute_etf_over_periods(
    isin: str,
    periods: List[str],
    csv_path: Optional[str] = None,  # mantenuto per compatibilità, anche se non utilizzato
    bench_override: Optional[str] = None,
    rf_ann: float = 0.0,
) -> Tuple[Dict[str, Optional[str]], List[Dict], Dict[str, Optional[pd.DataFrame]]]:
    """
    Calcola le metriche di un ETF su periodi definiti.
    Ritorna: (info_out, rows, aligned_frames)
    """

    # 1) TD esterna, ticker e TER
    td_external = fetch_tracking_difference(isin)
    ticker = resolve_isin_one(isin)
    if not ticker:
        raise RuntimeError(f"Nessun ticker Yahoo trovato per ISIN {isin}")

    yf_info = get_info(isin=isin)
    ter_pct = fetch_ter_justetf(isin)
    if ter_pct is None:
        ter_pct = fallback_ter_from_yahoo_info(yf_info)

    # 2) Determina il benchmark (mappatura manuale o heuristica di fallback)
    from .config import ISIN_TO_BENCHMARK  # import locale per evitare import circolari
    bench_symbol: Optional[str] = None
    bench_name: Optional[str] = None

    if isin in ISIN_TO_BENCHMARK:
        bench_symbol, bench_name = ISIN_TO_BENCHMARK[isin]
    else:
        etf_info_for_lookup = {**yf_info, "isin": isin}
        bench_symbol, bench_name = lookup_proxy_for_benchmark(None, ticker, etf_info_for_lookup)

    # Override esplicito
    if bench_override:
        bench_symbol = bench_override.strip()
        bench_name = f"Override: {bench_symbol}"

    # 3) Informazioni estese
    extended_info = get_etf_extended_info(isin)

    # 4) Output info
    info_out: Dict[str, Optional[str]] = {
        "isin": isin,
        "yahoo_symbol": ticker,
        "benchmark_symbol": bench_symbol,
        "benchmark_name": bench_name or "N/A",
        "ter_pct": f"{ter_pct:.2f}%" if ter_pct is not None else ND,
        "td_external": f"{td_external:.2f}%" if td_external is not None else ND,
        "category": extended_info.get("category", ND),
        "fund_size": extended_info.get("fund_size", ND),
        "replication_method": extended_info.get("replication_method", ND),
        "distribution": extended_info.get("distribution", ND),
        "alternative_etfs": extended_info.get("alternative_etfs", []),
        **yf_info,
    }

    # 5) Metriche per ciascun periodo
    rows: List[Dict] = []
    aligned_frames: Dict[str, Optional[pd.DataFrame]] = {}
    timestamp = dt.datetime.now().isoformat(timespec="seconds")

    for per in periods:
        etf_s = get_series(ticker, per)

        # Calcolo metriche ETF solo se la serie è valida
        if etf_s is not None and len(etf_s) >= 2:
            etf_metrics = compute_metrics_from_series(etf_s)
            sharpe = compute_sharpe_ratio(etf_s, rf_ann)
        else:
            etf_metrics = {}
            sharpe = None

        calmar = None
        cagr = etf_metrics.get("cagr")
        mdd = etf_metrics.get("mdd")
        if cagr is not None and mdd is not None and mdd != 0:
            # Manteniamo la convenzione pre-esistente (mdd presumibilmente negativo)
            calmar = -cagr / mdd

        td_val: Optional[float] = None
        aligned_df: Optional[pd.DataFrame] = None

        if bench_symbol:
            bench_s = get_series(bench_symbol, per)
            if (
                etf_s is not None and bench_s is not None
                and len(etf_s) > 1 and len(bench_s) > 1
            ):
                aligned_df = (
                    pd.concat(
                        [etf_s.rename("ETF"), bench_s.rename("Benchmark")],
                        axis=1, join="inner"
                    )
                    .dropna()
                )
                if aligned_df.shape[0] >= 2:
                    etf_return = (aligned_df["ETF"].iloc[-1] / aligned_df["ETF"].iloc[0] - 1.0) * 100.0
                    bench_return = (aligned_df["Benchmark"].iloc[-1] / aligned_df["Benchmark"].iloc[0] - 1.0) * 100.0
                    td_val = float(etf_return - bench_return)
            else:
                aligned_df = None

        aligned_frames[per] = aligned_df

        rows.append(
            {
                "timestamp": timestamp,
                "isin": isin,
                "yahoo_symbol": ticker,
                "period": per,
                "total_return_pct": etf_metrics.get("total"),
                "cagr_pct": cagr,
                "vol_ann_pct": etf_metrics.get("vol_ann"),
                "mdd_pct": mdd,
                "ter_pct": ter_pct,
                "benchmark_symbol": bench_symbol or "",
                "benchmark_name": bench_name or "",
                "tracking_diff_pct": td_val,
                "sharpe_ratio": sharpe,
                "calmar_ratio": calmar,
            }
        )

    return info_out, rows, aligned_frames