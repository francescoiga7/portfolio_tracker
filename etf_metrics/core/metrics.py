# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def _normalize_timezone(s: pd.Series) -> pd.Series:
    """Rimuove la timezone dall'indice (i dati del DB sono naive)."""
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s

def _as_price_series(s: pd.Series) -> Optional[pd.Series]:
    """Serie dei prezzi pulita (datetime naive, ordinata, valori > 0, >= 2
    punti); accetta anche un DataFrame con colonna 'Close'. None se inusabile."""
    if s is None: return None
    s = _normalize_timezone(s).sort_index()
    if isinstance(s, pd.DataFrame):
        values = s['Close'] if 'Close' in s.columns else None
    else:
        values = s
    if values is None: return None
    values = pd.to_numeric(values, errors='coerce').dropna()
    values = values[values > 0]
    return values if len(values) >= 2 else None

def calculate_atr_series(df: pd.DataFrame, window=14) -> pd.Series:
    """ATR (Average True Range) con smoothing EMA del True Range."""
    if not all(c in df.columns for c in ['High', 'Low', 'Close']):
        return pd.Series(0.0, index=df.index)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

def compute_metrics_from_series(s: pd.Series) -> Dict[str, Optional[float]]:
    """Metriche di base da una serie di prezzi: {'total', 'cagr', 'vol_ann',
    'mdd'} in % (None se non calcolabile)."""
    s_clean = _as_price_series(s)
    if s_clean is None: return {}
    total_return = (s_clean.iloc[-1] / s_clean.iloc[0]) - 1.0
    days = (s_clean.index[-1] - s_clean.index[0]).days
    years = days / 365.25 if days > 0 else len(s_clean) / 252.0
    cagr = ((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else None
    returns = s_clean.pct_change().dropna()
    vol_ann = returns.std(ddof=1) * np.sqrt(252) if len(returns) >= 2 else None
    wealth = s_clean / s_clean.iloc[0]
    mdd = (wealth / wealth.cummax() - 1.0).min()
    return {
        "total": total_return * 100.0,
        "cagr": cagr * 100.0 if cagr else None,
        "vol_ann": vol_ann * 100.0 if vol_ann else None,
        "mdd": mdd * 100.0 if mdd else None,
    }

def compute_sharpe_ratio(s: pd.Series, rf_annual_pct: float = 0.0) -> Optional[float]:
    """Sharpe annuizzato dai rendimenti daily (rf in % annuo)."""
    s_clean = _as_price_series(s)
    if s_clean is None or len(s_clean) < 3: return None
    returns = s_clean.pct_change().dropna()
    rf_daily = (rf_annual_pct / 100.0) / 252.0
    excess = returns - rf_daily
    std = excess.std(ddof=1)
    return float((excess.mean() / std) * np.sqrt(252)) if std > 0 else None


def compute_backtest_performance_metrics(df_trades: pd.DataFrame, initial_capital: float,
                                         rf_annual: float = 0.03, nav_daily=None):
    """
    Metriche di performance di un backtest (CAGR, Sharpe, Sortino, MaxDD,
    Profit Factor, Equity_Curve) dal log dei trade.

    Con `nav_daily` (Serie NAV mark-to-market giornaliera emessa dal motore,
    `diag['nav_daily']`) la curva equity e' quella vera: DD e volatilita'
    riflettono il prezzo del portafoglio a OGNI seduta e il CAGR parte
    dall'inizio della simulazione (periodo in cash incluso). Senza `nav_daily`
    si ricade nella ricostruzione storica (Total_Equity/Capital ffilled sui
    soli giorni di trade), che sottostima volatilita' e drawdown.
    """
    if df_trades is None or df_trades.empty:
        return None

    df = df_trades.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = df['Date'].min()
    end_date = df['Date'].max()

    if start_date == end_date:
        return None

    nav = None
    if nav_daily is not None:
        nav = _normalize_timezone(pd.Series(nav_daily)).dropna().astype(float)
        if len(nav) < 2 or nav.index[-1] < end_date:
            nav = None  # incompleta rispetto al log: fallback alla ricostruzione

    if nav is not None:
        equity_df = pd.DataFrame(index=nav.index)
        equity_df['Value'] = nav.to_numpy()
    else:
        all_dates = pd.date_range(start_date, end_date, freq='B')
        equity_df = pd.DataFrame(index=all_dates)

        val_col = 'Total_Equity' if 'Total_Equity' in df.columns else 'Capital'
        equity_df['Value'] = np.nan
        equity_df.iloc[0, 0] = initial_capital

        daily_val_update = df.groupby('Date')[val_col].last()
        equity_df.loc[daily_val_update.index, 'Value'] = daily_val_update
        equity_df['Value'] = equity_df['Value'].ffill()

    equity_df['Daily_Ret'] = equity_df['Value'].pct_change().fillna(0)

    final_cap = equity_df['Value'].iloc[-1]
    days = (equity_df.index[-1] - equity_df.index[0]).days
    years = days / 365.25
    cagr = ((final_cap / initial_capital) ** (1 / years)) - 1 if years > 0 else 0

    volatility = equity_df['Daily_Ret'].std() * np.sqrt(252)
    sharpe = (cagr - rf_annual) / volatility if volatility > 0 else 0

    negative_returns = equity_df[equity_df['Daily_Ret'] < 0]['Daily_Ret']
    downside_dev = negative_returns.std() * np.sqrt(252)
    sortino = (cagr - rf_annual) / downside_dev if downside_dev > 0 else 0

    cumulative_returns = (1 + equity_df['Daily_Ret']).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    sells = df[df['Action'] == 'SELL']
    gross_profit = sells[sells['PnL_Net'] > 0]['PnL_Net'].sum()
    gross_loss = abs(sells[sells['PnL_Net'] < 0]['PnL_Net'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": max_drawdown,
        "Profit_Factor": profit_factor,
        "Equity_Curve": equity_df['Value'],
    }