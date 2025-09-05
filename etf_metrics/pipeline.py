# -*- coding: utf-8 -*-
import datetime as dt
from typing import Dict, List, Optional
import pandas as pd

from .yahoo_client import resolve_isin_one, get_series, get_info
from .metrics import compute_metrics_from_series, compute_sharpe_ratio
from .io_csv import upsert_results_csv
from .trackingdiff_client import fetch_tracking_difference
from .etf_info import get_etf_extended_info, fetch_ter_justetf, fallback_ter_from_yahoo_info


def compute_etf_over_periods(
        isin: str,
        periods: List[str],
        csv_path: Optional[str] = None,
        bench_override: Optional[str] = None,
        rf_ann: float = 0.0,
):
    """
    Calcola le metriche di un ETF su periodi definiti.
    """
    # 1. Recupera informazioni di base e TER
    td_external = fetch_tracking_difference(isin)
    ticker = resolve_isin_one(isin)
    if not ticker:
        raise RuntimeError(f"Nessun ticker Yahoo trovato per ISIN {isin}")

    yf_info = get_info(isin=isin)
    ter_pct = fetch_ter_justetf(isin)
    if ter_pct is None:
        ter_pct = fallback_ter_from_yahoo_info(yf_info)

    # 2. Determina il benchmark solo da configurazione statica
    from .config import ISIN_TO_BENCHMARK
    bench_symbol = None
    bench_name = None  # Variabile per il nome finale del benchmark

    # Usa solo mappatura manuale ISIN_TO_BENCHMARK
    if isin in ISIN_TO_BENCHMARK:
        bench_symbol, bench_name = ISIN_TO_BENCHMARK[isin]
    else:
        # Fallback: cerca per ticker nei mapping
        from .benchmark import lookup_proxy_for_benchmark
        etf_info_for_lookup = {**yf_info, "isin": isin}
        bench_symbol, bench_name = lookup_proxy_for_benchmark(None, ticker, etf_info_for_lookup)

    # Se override esplicito, rispetta l'override per il simbolo e aggiorna il nome
    if bench_override:
        bench_symbol = bench_override.strip()
        bench_name = f"Override: {bench_symbol}"

    # 3. Recupera informazioni estese
    extended_info = get_etf_extended_info(isin)

    # 4. Popola il dizionario di output
    info_out: Dict[str, Optional[str]] = {
        "isin": isin,
        "yahoo_symbol": ticker,
        "benchmark_symbol": bench_symbol,
        "benchmark_name": bench_name or "N/A",
        "ter_pct": f"{ter_pct:.2f}%" if ter_pct is not None else "n.d.",
        "td_external": f"{td_external:.2f}%" if td_external is not None else "n.d.",
        "category": extended_info.get("category", "n.d."),
        "fund_size": extended_info.get("fund_size", "n.d."),
        "replication_method": extended_info.get("replication_method", "n.d."),
        "distribution": extended_info.get("distribution", "n.d."),
        "alternative_etfs": extended_info.get("alternative_etfs", []),
        **yf_info
    }

    # 5. Calcola le metriche per ciascun periodo
    rows: List[Dict] = []
    aligned_frames: Dict[str, Optional[pd.DataFrame]] = {}
    timestamp = dt.datetime.now().isoformat(timespec="seconds")

    for per in periods:
        etf_s = get_series(ticker, per)
        etf_metrics = compute_metrics_from_series(etf_s)

        sharpe = compute_sharpe_ratio(etf_s, rf_ann) if etf_s is not None else None
        calmar = None
        cagr = etf_metrics.get("cagr")
        mdd = etf_metrics.get("mdd")
        if cagr is not None and mdd is not None and mdd != 0:
            calmar = -cagr / mdd

        td_val, aligned_df = None, None
        if bench_symbol:
            bench_s = get_series(bench_symbol, per)
            if etf_s is not None and bench_s is not None and len(etf_s) > 1 and len(bench_s) > 1:
                aligned_df = pd.concat([etf_s.rename('ETF'), bench_s.rename('Benchmark')], axis=1,
                                       join='inner').dropna()
                if aligned_df.shape[0] >= 2:
                    etf_return = (aligned_df['ETF'].iloc[-1] / aligned_df['ETF'].iloc[0] - 1.0) * 100.0
                    bench_return = (aligned_df['Benchmark'].iloc[-1] / aligned_df['Benchmark'].iloc[0] - 1.0) * 100.0
                    td_val = float(etf_return - bench_return)
                else:
                    aligned_df = None
        aligned_frames[per] = aligned_df

        rows.append({
            "timestamp": timestamp, "isin": isin, "yahoo_symbol": ticker, "period": per,
            "total_return_pct": etf_metrics.get("total"), "cagr_pct": cagr, "vol_ann_pct": etf_metrics.get("vol_ann"),
            "mdd_pct": mdd, "ter_pct": ter_pct, "benchmark_symbol": bench_symbol or "",
            "benchmark_name": bench_name or "", "tracking_diff_pct": td_val,
            "sharpe_ratio": sharpe, "calmar_ratio": calmar
        })

    # 6. Salva i risultati se richiesto
    if csv_path:
        try:
            df_to_save = pd.DataFrame(rows).round(6)
            for col in df_to_save.columns:
                if 'pct' in col or 'ratio' in col:
                    df_to_save[col] = df_to_save[col].astype(object).where(pd.notna(df_to_save[col]), None)
            upsert_results_csv(csv_path, df_to_save.to_dict('records'))
        except Exception as e:
            print(f"Errore nel salvataggio CSV: {e}")

    return info_out, rows, aligned_frames