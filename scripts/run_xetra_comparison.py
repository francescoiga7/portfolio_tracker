# -*- coding: utf-8 -*-
"""Confronto strategie sui dati Xetra reali del DB locale (nessuna rete).

- Universo point-in-time: i ticker entrano solo quando hanno dati (listing
  date); gli indicatori (SMA200 ecc.) richiedono comunque lo storico minimo.
- Analisi per anno civile: se il vantaggio di una strategia dipende solo da
  un regime (es. 2022 bear), e' probabile overfitting.
- Benchmark: SPY se presente nel DB, altrimenti proxy equal-weight universo.

Uso: python3 scripts/run_xetra_comparison.py [keys_csv] [start] [end]
"""
import logging
import sys
import time

import pandas as pd

logging.basicConfig(level=logging.WARNING)

from etf_metrics.core.automated_backtest import (
    CONFIG, STRATEGY_ORDER, prepare_market_data, run_strategy_comparison,
    compute_buy_and_hold_curves, BENCH_PROXY_KEY,
)
from etf_metrics.core.metrics import compute_backtest_performance_metrics
from etf_metrics.core.parquet_store import ParquetStore

UNIVERSE_MIN_ROWS = 200     # come il selettore "Tutto il DB" della UI
INITIAL_CAPITAL = 10000.0


def load_universe():
    """Ticker del DB con storico sufficiente (>= UNIVERSE_MIN_ROWS righe)."""
    store = ParquetStore('market_data_parquet')
    stats = store.ticker_row_stats()
    # solo ticker con abbastanza storico per gli indicatori del motore
    tickers = sorted(t for t, (r, _f, _l) in stats.items() if r >= UNIVERSE_MIN_ROWS)
    return tickers


def yearly_returns(equity: pd.Series) -> pd.Series:
    """Rendimenti % per anno civile da una curva equity."""
    eq = equity.dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    years = eq.index.year
    out = {}
    prev = eq.iloc[0]
    for y in sorted(set(years)):
        vals = eq[years == y]
        out[y] = (vals.iloc[-1] / prev - 1.0) * 100.0
        prev = vals.iloc[-1]
    return pd.Series(out)


def main(strategies, start_date, label, universe_data, tickers, params=None):
    t0 = time.time()
    res = run_strategy_comparison(
        tickers, strategies=strategies, start_date=start_date,
        initial_capital=INITIAL_CAPITAL, preloaded_data=universe_data,
        tax_rate=26.0, allow_fractional=True, strategy_params=params,
    )
    print(f"\n=== {label} | start {start_date} | {len(tickers)} ticker | {time.time()-t0:.0f}s ===")

    bh = compute_buy_and_hold_curves(universe_data, tickers, start_date,
                                     INITIAL_CAPITAL, benchmark_ticker=CONFIG['SPY_TICKER'])
    bench_key = 'benchmark' if 'benchmark' in bh else 'benchmark_proxy'
    bench_lbl = 'S&P500' if bench_key == 'benchmark' else 'Universo (proxy EW)'
    bench = bh.get(bench_key)
    eqw = bh.get('equal_weight')

    rows = []
    curves = {}
    if bench is not None:
        curves[f"{bench_lbl} B&H"] = bench
        rows.append(_row(f"{bench_lbl} B&H", bench, bench.iloc[-1], 0))
    if eqw is not None:
        curves["Equal weight B&H"] = eqw
        rows.append(_row("Equal weight B&H", eqw, eqw.iloc[-1], 0))

    for k, r in res.items():
        m = compute_backtest_performance_metrics(r['trades'], INITIAL_CAPITAL,
                                                 nav_daily=r.get('nav'))
        if m and m.get('Equity_Curve') is not None:
            curves[r['name']] = m['Equity_Curve']
        rows.append(_row(r['name'], m['Equity_Curve'] if m else None, r['final'],
                         len(r['trades']), m))
    df = pd.DataFrame(rows).set_index('Strategia')
    print(df.round(1).to_string())
    return curves


def _row(name, equity, final, n_trades, m=None):
    row = {
        'Strategia': name,
        'Finale': final,
        'Rend.%': (final / INITIAL_CAPITAL - 1) * 100,
        'CAGR%': m['CAGR'] * 100 if m else float('nan'),
        'MaxDD%': m['Max_Drawdown'] * 100 if m else float('nan'),
        'Sharpe': m['Sharpe'] if m else float('nan'),
        'Trades': n_trades,
        'Commissioni': float('nan'),
    }
    if equity is not None and len(equity) > 250:
        yr = yearly_returns(equity)
        for y, v in yr.items():
            row[str(y)] = round(v, 1)
    return row


if __name__ == '__main__':
    strategies = sys.argv[1].split(',') if len(sys.argv) > 1 else STRATEGY_ORDER
    start = sys.argv[2] if len(sys.argv) > 2 else '2021-01-04'
    end = sys.argv[3] if len(sys.argv) > 3 else None
    tickers = load_universe()
    print(f"Universo: {len(tickers)} ticker (>= {UNIVERSE_MIN_ROWS} righe)")
    t0 = time.time()
    data = prepare_market_data(tickers, period='max', allow_download=False)
    print(f"Dati caricati: {len(data)} ticker ({time.time()-t0:.0f}s)")
    if end:
        end_ts = pd.to_datetime(end)
        data = {t: df[df.index <= end_ts] for t, df in data.items()
                if df is not None and len(df[df.index <= end_ts]) > 0}
        print(f"Finestra troncata a {end}: {len(data)} ticker")
    curves = main(strategies, start, 'Confronto strategie', data, tickers)
