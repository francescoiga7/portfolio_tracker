# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from etf_metrics.core.automated_backtest import (
    run_market_aware_backtest,
    prepare_market_data,
    calculate_indicators
)

logger = logging.getLogger(__name__)


def detrend_series(series: pd.Series) -> pd.Series:
    """
    Rimuove il trend (drift) usando calcoli sui Log-Returns.
    Restituisce una serie di prezzi 'Zero Mean'.
    """
    if series.empty: return series

    clean_series = series.dropna()
    if len(clean_series) < 2: return series

    prices = clean_series.values

    log_rets = np.log(prices[1:] / prices[:-1])
    avg_ret = np.mean(log_rets)
    detrended_rets = log_rets - avg_ret
    reconstructed_path = np.exp(np.cumsum(np.insert(detrended_rets, 0, 0)))
    final_prices = reconstructed_path * prices[0]

    return pd.Series(final_prices, index=clean_series.index, name=clean_series.name)


def shuffle_series(series: pd.Series) -> pd.Series:
    """
    Monte Carlo Permutation: Mescola i rendimenti giornalieri.
    Distrugge la correlazione seriale mantenendo la distribuzione.
    """
    if series.empty: return series

    rets = series.pct_change().dropna().values
    np.random.shuffle(rets)

    initial_price = series.iloc[0]
    price_path = np.cumprod(1 + rets) * initial_price
    full_path = np.insert(price_path, 0, initial_price)

    if len(full_path) > len(series):
        full_path = full_path[:len(series)]

    return pd.Series(full_path, index=series.index, name=series.name)


def _rebuild_synthetic_df(original_df: pd.DataFrame, new_close: pd.Series) -> pd.DataFrame:
    """Ricostruisce un DataFrame OHLCV coerente basato su una nuova serie Close."""
    common_index = original_df.index.intersection(new_close.index)
    if common_index.empty: return pd.DataFrame()

    df_slice = original_df.loc[common_index]
    new_close = new_close.loc[common_index]

    ratio = new_close / df_slice['Close']

    df_synth = pd.DataFrame(index=common_index)
    df_synth['Close'] = new_close
    df_synth['Open'] = df_slice['Open'] * ratio
    df_synth['High'] = df_slice['High'] * ratio
    df_synth['Low'] = df_slice['Low'] * ratio
    df_synth['Volume'] = df_slice['Volume']

    df_synth = calculate_indicators(df_synth)

    return df_synth


def run_detrended_analysis(tickers, start_date, initial_capital=1000):
    """
    Esegue il backtest su dati Detrended.
    Serve a verificare se la strategia guadagna grazie all'Alpha o solo al Trend di mercato.
    """
    real_data = prepare_market_data(tickers, period="10y")
    detrended_data = {}

    for t, df in real_data.items():
        if df.empty: continue

        detr_close = detrend_series(df['Close'])

        df_d = _rebuild_synthetic_df(df, detr_close)

        if not df_d.empty:
            detrended_data[t] = df_d

    if not detrended_data:
        return pd.DataFrame(), initial_capital, {}

    trades, end_cap = run_market_aware_backtest(
        tickers, start_date, initial_capital, preloaded_data=detrended_data
    )

    return trades, end_cap, detrended_data


def run_monte_carlo_permutation_test(tickers, start_date, n_simulations=50, initial_capital=1000):
    """
    Esegue N simulazioni Monte Carlo mescolando i rendimenti.
    Verifica se il risultato è statisticamente significativo o frutto del caso.
    """
    real_data = prepare_market_data(tickers, period="10y")
    _, real_final_cap = run_market_aware_backtest(
        tickers, start_date, initial_capital, preloaded_data=real_data
    )
    real_return = (real_final_cap - initial_capital) / initial_capital

    simulation_results = []

    for i in range(n_simulations):
        synthetic_data = {}
        for t, df in real_data.items():
            if df.empty: continue

            shuffled_close = shuffle_series(df['Close'])

            df_synth = _rebuild_synthetic_df(df, shuffled_close)

            if not df_synth.empty:
                synthetic_data[t] = df_synth

        _, synth_cap = run_market_aware_backtest(
            tickers, start_date, initial_capital, preloaded_data=synthetic_data
        )
        synth_return = (synth_cap - initial_capital) / initial_capital
        simulation_results.append(synth_return)

    better_than_real = sum(1 for r in simulation_results if r >= real_return)
    p_value = better_than_real / n_simulations if n_simulations > 0 else 1.0

    return {
        "real_return": real_return,
        "monte_carlo_returns": simulation_results,
        "p_value": p_value,
        "n_simulations": n_simulations
    }


def run_walk_forward_analysis(tickers, initial_capital=1000, train_months=24, test_months=6):
    """
    Walk-Forward Analysis (WFA) per verificare la stabilità nel tempo.
    Nota: La strategia V28 non ha parametri ottimizzabili (SMA130 fissa),
    quindi questo test verifica la consistenza dei rendimenti su finestre mobili.
    """
    full_data = prepare_market_data(tickers, period="10y")
    if not full_data: return pd.DataFrame()

    sample = list(full_data.values())[0]
    start_dt = sample.index[0]
    end_dt = sample.index[-1]

    current_dt = start_dt
    results = []

    while current_dt < end_dt:
        window_end = current_dt + pd.DateOffset(months=test_months)
        if window_end > end_dt: break

        window_data = {}
        valid_window = False
        for t, df in full_data.items():
            mask = (df.index >= current_dt) & (df.index <= window_end)
            df_slice = df.loc[mask].copy()
            if not df_slice.empty and len(df_slice) > 130:
                window_data[t] = df_slice
                valid_window = True

        if valid_window:
            trades, end_cap = run_market_aware_backtest(
                tickers,
                start_date=str(current_dt.date()),
                initial_capital=initial_capital,
                preloaded_data=window_data
            )

            pnl = end_cap - initial_capital
            roi = (pnl / initial_capital) * 100

            results.append({
                "Window Start": current_dt.date(),
                "Window End": window_end.date(),
                "PnL": pnl,
                "ROI %": roi,
                "Trades": len(trades)
            })

        current_dt = window_end

    return pd.DataFrame(results)