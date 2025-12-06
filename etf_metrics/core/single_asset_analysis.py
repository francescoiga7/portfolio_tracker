# -*- coding: utf-8 -*-
import datetime as dt
from typing import Dict, List, Optional, Tuple
import pandas as pd

from etf_metrics.clients.yahoo_client import resolve_isin_one, get_series, get_info
from etf_metrics.core.metrics import (
    compute_metrics_from_series,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_omega_ratio,
    compute_var
)
from etf_metrics.core.trading import analyze_ticker
from etf_metrics.clients.trackingdiff_client import fetch_tracking_difference
from etf_metrics.core.etf_info import get_etf_extended_info, fetch_ter_justetf, fallback_ter_from_yahoo_info
from etf_metrics.core.benchmark import lookup_proxy_for_benchmark

ND = "n.d."


def compute_single_asset_analysis(
        isin: str,
        periods: List[str],
        bench_override: Optional[str] = None,
        rf_ann: float = 0.0,
) -> Tuple[Dict, List[Dict], Dict]:
    ticker = resolve_isin_one(isin)
    if not ticker:
        raise RuntimeError(f"Nessun ticker Yahoo trovato per ISIN {isin}")

    yf_info = get_info(isin=isin)
    ter_pct = fetch_ter_justetf(isin) or fallback_ter_from_yahoo_info(yf_info)
    td_external = fetch_tracking_difference(isin)
    extended_info = get_etf_extended_info(isin)

    bench_symbol, bench_name = lookup_proxy_for_benchmark(None, ticker, {**yf_info, "isin": isin})
    if bench_override:
        bench_symbol = bench_override.strip()
        bench_name = f"Override: {bench_symbol}"

    full_series_df = get_series(ticker, period="10y", as_dataframe=True)
    trading_signal = {}

    if full_series_df is not None and not full_series_df.empty:
        trading_signal = analyze_ticker(ticker, full_series_df)

    info_out = {
        "isin": isin,
        "yahoo_symbol": ticker,
        "name": extended_info.get("longName") or yf_info.get("longName", ticker),
        "benchmark_symbol": bench_symbol,
        "benchmark_name": bench_name or "N/A",
        "ter_pct": f"{ter_pct:.2f}%" if ter_pct is not None else ND,
        "td_external": f"{td_external:.2f}%" if td_external is not None else ND,
        "category": extended_info.get("category", ND),
        "fund_size": extended_info.get("fund_size", ND),
        "replication_method": extended_info.get("replication_method", ND),
        "distribution": extended_info.get("distribution", ND),
        "trading_signal": trading_signal,
        **yf_info,
    }

    rows: List[Dict] = []
    aligned_frames: Dict = {}
    timestamp = dt.datetime.now().isoformat(timespec="seconds")

    for per in periods:
        etf_s = get_series(ticker, per)

        metrics, sharpe, sortino, omega, var, calmar, td_val, aligned_df = [None] * 8

        if etf_s is not None and len(etf_s) >= 2:
            metrics = compute_metrics_from_series(etf_s)
            sharpe = compute_sharpe_ratio(etf_s, rf_ann)
            sortino = compute_sortino_ratio(etf_s, rf_ann)
            omega = compute_omega_ratio(etf_s)
            var = compute_var(etf_s, confidence_level=0.95)

            cagr = metrics.get("cagr")
            mdd = metrics.get("mdd")
            if cagr is not None and mdd is not None and mdd != 0:
                calmar = -cagr / mdd
        else:
            metrics = {}

        if bench_symbol:
            bench_s = get_series(bench_symbol, per)
            if etf_s is not None and bench_s is not None and len(etf_s) > 1 and len(bench_s) > 1:
                aligned_df = pd.concat([etf_s.rename("ETF"), bench_s.rename("Benchmark")], axis=1,
                                       join="inner").dropna()
                if aligned_df.shape[0] >= 2:
                    etf_ret = (aligned_df["ETF"].iloc[-1] / aligned_df["ETF"].iloc[0] - 1.0) * 100.0
                    bench_ret = (aligned_df["Benchmark"].iloc[-1] / aligned_df["Benchmark"].iloc[0] - 1.0) * 100.0
                    td_val = etf_ret - bench_ret

        aligned_frames[per] = aligned_df

        rows.append({
            "timestamp": timestamp, "isin": isin, "yahoo_symbol": ticker, "period": per,
            "total_return_pct": metrics.get("total"), "cagr_pct": metrics.get("cagr"),
            "vol_ann_pct": metrics.get("vol_ann"), "mdd_pct": metrics.get("mdd"),
            "ter_pct": ter_pct,
            "tracking_diff_pct": td_val, "sharpe_ratio": sharpe, "calmar_ratio": calmar,
            "sortino_ratio": sortino, "omega_ratio": omega, "var_95_1d_pct": var
        })

    return info_out, rows, aligned_frames