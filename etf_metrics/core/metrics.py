# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _normalize_timezone(s: pd.Series) -> pd.Series:
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s


def _as_price_series(s: pd.Series) -> Optional[pd.Series]:
    if s is None: return None
    s = _normalize_timezone(s).sort_index()
    if isinstance(s, pd.DataFrame):
        if 'Close' in s.columns:
            values = s['Close']
        else:
            return None
    else:
        values = s

    values = pd.to_numeric(values, errors='coerce').dropna()
    values = values[values > 0]

    if len(values) < 2: return None
    return values


def calculate_atr_series(df: pd.DataFrame, window=14) -> pd.Series:
    """Calcola ATR usando la media mobile esponenziale (Wilder's Smoothing)."""
    if 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns:
        return pd.Series(dtype=float)

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def calculate_adx_series(df: pd.DataFrame, window=14) -> pd.Series:
    """Calcola ADX usando EMA."""
    if 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns:
        return pd.Series(dtype=float)

    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = calculate_atr_series(df, window=1)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean().abs() / tr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx


def calculate_rsi_series(series: pd.Series, window=14) -> pd.Series:
    """Calcola RSI usando EMA (Standard TradingView)."""
    delta = series.diff()

    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_metrics_from_series(s: pd.Series) -> Dict[str, Optional[float]]:
    s_clean = _as_price_series(s)
    if s_clean is None: return {}

    total_return = (s_clean.iloc[-1] / s_clean.iloc[0]) - 1.0

    days = (s_clean.index[-1] - s_clean.index[0]).days
    years = days / 365.25 if days > 0 else len(s_clean) / 252.0

    cagr = ((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else None

    returns = s_clean.pct_change().dropna()
    vol_ann = returns.std(ddof=1) * np.sqrt(252) if len(returns) >= 2 else None

    wealth = s_clean / s_clean.iloc[0]
    roll_max = wealth.cummax()
    drawdowns = wealth / roll_max - 1.0
    mdd = drawdowns.min() if not drawdowns.empty else None

    return {
        "total": total_return * 100.0 if total_return is not None else None,
        "cagr": cagr * 100.0 if cagr is not None else None,
        "vol_ann": vol_ann * 100.0 if vol_ann is not None else None,
        "mdd": mdd * 100.0 if mdd is not None else None,
    }


def compute_sharpe_ratio(s: pd.Series, rf_annual_pct: float = 0.0) -> Optional[float]:
    s_clean = _as_price_series(s)
    if s_clean is None or len(s_clean) < 3: return None
    returns = s_clean.pct_change().dropna()

    rf_daily = (rf_annual_pct / 100.0) / 252.0
    excess_returns = returns - rf_daily

    mean_excess = excess_returns.mean()
    std_dev = excess_returns.std(ddof=1)

    if std_dev == 0 or np.isnan(std_dev): return None
    return float((mean_excess / std_dev) * np.sqrt(252))


def compute_sortino_ratio(s: pd.Series, rf_annual_pct: float = 0.0) -> Optional[float]:
    s_clean = _as_price_series(s)
    if s_clean is None: return None
    returns = s_clean.pct_change().dropna()

    rf_daily = (rf_annual_pct / 100.0) / 252.0
    excess_returns = returns - rf_daily

    downside = excess_returns[excess_returns < 0]
    downside_std = downside.std(ddof=1)

    if downside_std == 0 or np.isnan(downside_std): return None
    return float((excess_returns.mean() / downside_std) * np.sqrt(252))


def compute_omega_ratio(s: pd.Series, required_return_pct: float = 0.0) -> Optional[float]:
    s_clean = _as_price_series(s)
    if s_clean is None: return None
    returns = s_clean.pct_change().dropna()

    thresh = (required_return_pct / 100.0) / 252.0
    gains = (returns - thresh).where(returns > thresh, 0).sum()
    losses = (thresh - returns).where(returns < thresh, 0).sum()

    if losses == 0: return None
    return float(gains / losses)


def compute_var(s: pd.Series, confidence_level: float = 0.95) -> Optional[float]:
    s_clean = _as_price_series(s)
    if s_clean is None: return None
    returns = s_clean.pct_change().dropna()
    if returns.empty: return None

    return float(-returns.quantile(1 - confidence_level) * 100.0)