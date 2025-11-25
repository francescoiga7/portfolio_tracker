# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from etf_metrics.core.automated_backtest import (
    run_market_aware_backtest,
    prepare_market_data,
    calculate_atr_series,
    calculate_adx_series
)

logger = logging.getLogger(__name__)


def detrend_series(series: pd.Series) -> pd.Series:
    """
    Rimuove il trend (drift) usando calcoli vettoriali Numpy.
    Metodo Log-Returns per evitare problemi matematici.
    """
    if series.empty: return series

    # Pulizia dati (rimuovi NaN iniziali)
    clean_series = series.dropna()
    if len(clean_series) < 2: return series

    prices = clean_series.values

    # 1. Calcola Log Returns: Log(P_t / P_t-1)
    log_rets = np.log(prices[1:] / prices[:-1])

    # 2. Calcola il Drift medio (Componente Trend)
    avg_ret = np.mean(log_rets)

    # 3. Sottrai il Drift per ottenere rendimenti 'Zero Mean'
    detrended_rets = log_rets - avg_ret

    # 4. Ricostruisci la curva dei prezzi (Cumulative Product)
    # P_t = P_0 * exp( cumsum(r_detrended) )
    # Inseriamo 0 all'inizio per riallinearci con la lunghezza originale
    reconstructed_path = np.exp(np.cumsum(np.insert(detrended_rets, 0, 0)))

    # Scaliamo per il prezzo iniziale originale
    final_prices = reconstructed_path * prices[0]

    # Ricostruiamo la Serie Pandas con l'indice originale corretto
    return pd.Series(final_prices, index=clean_series.index, name=clean_series.name)


def shuffle_series(series: pd.Series) -> pd.Series:
    """
    Monte Carlo Permutation: Mescola i rendimenti giornalieri.
    Distrugge la correlazione seriale mantenendo la distribuzione statistica.
    """
    if series.empty: return series

    # Calcola rendimenti percentuali
    rets = series.pct_change().dropna().values

    # Mescola casualmente (Shuffle in-place)
    np.random.shuffle(rets)

    # Ricostruisce la serie prezzi
    initial_price = series.iloc[0]
    price_path = np.cumprod(1 + rets) * initial_price

    # Aggiungi il prezzo iniziale in testa
    full_path = np.insert(price_path, 0, initial_price)

    # Gestione lunghezza
    if len(full_path) > len(series):
        full_path = full_path[:len(series)]

    return pd.Series(full_path, index=series.index, name=series.name)


def run_detrended_analysis(tickers, start_date, initial_capital=1000):
    """Esegue il backtest su dati Detrended (Zero Mean) ricalcolando gli indicatori."""
    real_data = prepare_market_data(tickers)
    detrended_data = {}

    for t, df in real_data.items():
        if df.empty: continue

        df_d = df.copy()

        # 1. Detrend Close
        orig_close = df['Close']
        detr_close = detrend_series(orig_close)

        # 2. Allinea indici e scala OHLC
        common_index = df.index.intersection(detr_close.index)
        if common_index.empty: continue

        df = df.loc[common_index]
        detr_close = detr_close.loc[common_index]

        # Ratio per scalare Open, High, Low
        ratio = detr_close / df['Close']

        df_d = pd.DataFrame(index=common_index)
        df_d['Close'] = detr_close
        df_d['Open'] = df['Open'] * ratio
        df_d['High'] = df['High'] * ratio
        df_d['Low'] = df['Low'] * ratio
        df_d['Volume'] = df['Volume']  # Volume originale

        # 3. RICALCOLO CRITICO INDICATORI SU PREZZI DETRENDED
        df_d['ATR'] = calculate_atr_series(df_d)
        df_d['SMA200'] = df_d['Close'].rolling(200).mean()
        df_d['SMA50'] = df_d['Close'].rolling(50).mean()
        df_d['SMA20'] = df_d['Close'].rolling(20).mean()
        df_d['STD20'] = df_d['Close'].rolling(20).std()
        df_d['Vol_SMA20'] = df_d['Volume'].rolling(20).mean()
        df_d['ADX'] = calculate_adx_series(df_d)

        # RSI
        delta = df_d['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df_d['RSI'] = 100 - (100 / (1 + rs))

        detrended_data[t] = df_d

    if not detrended_data:
        return pd.DataFrame(), initial_capital, {}

    trades, end_cap = run_market_aware_backtest(
        tickers, start_date, initial_capital, preloaded_data=detrended_data
    )

    return trades, end_cap, detrended_data


def run_monte_carlo_permutation_test(tickers, start_date, n_simulations=50, initial_capital=1000):
    """Esegue N simulazioni Monte Carlo con dati 'shuffled'."""
    real_data = prepare_market_data(tickers)
    real_trades, real_final_cap = run_market_aware_backtest(
        tickers, start_date, initial_capital, preloaded_data=real_data
    )
    real_return = (real_final_cap - initial_capital) / initial_capital

    simulation_results = []

    for i in range(n_simulations):
        synthetic_data = {}
        for t, df in real_data.items():
            df_synth = df.copy()

            original_close = df['Close'].copy()
            shuffled_close = shuffle_series(original_close)

            # Gestione allineamento per sicurezza
            common_index = df.index.intersection(shuffled_close.index)
            df_slice = df.loc[common_index]
            shuffled_close = shuffled_close.loc[common_index]

            ratio = shuffled_close / df_slice['Close']

            df_synth = pd.DataFrame(index=common_index)
            df_synth['Close'] = shuffled_close
            df_synth['Open'] = df_slice['Open'] * ratio
            df_synth['High'] = df_slice['High'] * ratio
            df_synth['Low'] = df_slice['Low'] * ratio
            df_synth['Volume'] = df_slice['Volume']

            # Ricalcolo indicatori
            df_synth['ATR'] = calculate_atr_series(df_synth)
            df_synth['SMA200'] = df_synth['Close'].rolling(200).mean()
            df_synth['SMA50'] = df_synth['Close'].rolling(50).mean()
            df_synth['SMA20'] = df_synth['Close'].rolling(20).mean()
            df_synth['STD20'] = df_synth['Close'].rolling(20).std()
            df_synth['Vol_SMA20'] = df_synth['Volume'].rolling(20).mean()
            df_synth['ADX'] = calculate_adx_series(df_synth)

            # RSI
            delta = df_synth['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df_synth['RSI'] = 100 - (100 / (1 + rs))

            synthetic_data[t] = df_synth

        _, synth_cap = run_market_aware_backtest(
            tickers, start_date, initial_capital, preloaded_data=synthetic_data
        )
        synth_return = (synth_cap - initial_capital) / initial_capital
        simulation_results.append(synth_return)

    better_than_real = sum(1 for r in simulation_results if r > real_return)
    p_value = better_than_real / n_simulations if n_simulations > 0 else 1.0

    return {
        "real_return": real_return,
        "monte_carlo_returns": simulation_results,
        "p_value": p_value,
        "n_simulations": n_simulations
    }


def run_walk_forward_analysis(tickers, initial_capital=1000, train_months=12, test_months=3):
    """Walk-Forward Analysis (WFA) su finestre scorrevoli."""
    full_data = prepare_market_data(tickers, period="5y")
    if not full_data: return pd.DataFrame()

    sample = list(full_data.values())[0]
    start_dt = sample.index[0]
    end_dt = sample.index[-1]

    current_dt = start_dt + pd.DateOffset(months=train_months)
    results = []

    while current_dt < end_dt:
        oos_end = current_dt + pd.DateOffset(months=test_months)
        if oos_end > end_dt: oos_end = end_dt

        window_data = {}
        for t, df in full_data.items():
            mask = (df.index >= current_dt) & (df.index <= oos_end)
            if mask.any():
                window_data[t] = df.loc[mask]

        if window_data:
            trades, end_cap = run_market_aware_backtest(
                tickers, "2000-01-01", initial_capital, preloaded_data=window_data
            )

            pnl = end_cap - initial_capital
            results.append({
                "Window Start": current_dt.date(),
                "Window End": oos_end.date(),
                "PnL": pnl,
                "Trades": len(trades[trades['Action'].str.contains("SELL")]) if not trades.empty else 0
            })

        current_dt = oos_end

    return pd.DataFrame(results)