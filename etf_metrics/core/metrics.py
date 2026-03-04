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
        values = s['Close'] if 'Close' in s.columns else None
    else:
        values = s
    if values is None: return None
    values = pd.to_numeric(values, errors='coerce').dropna()
    values = values[values > 0]
    return values if len(values) >= 2 else None

def calculate_atr_series(df: pd.DataFrame, window=14) -> pd.Series:
    if not all(c in df.columns for c in ['High', 'Low', 'Close']):
        return pd.Series(0.0, index=df.index)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

def calculate_adx_series(df: pd.DataFrame, window=14) -> pd.Series:
    if not all(c in df.columns for c in ['High', 'Low', 'Close']):
        return pd.Series(0.0, index=df.index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr = calculate_atr_series(df, window=1)
    tr_smooth = tr.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/window, min_periods=window, adjust=False).mean() / tr_smooth.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/window, min_periods=window, adjust=False).mean() / tr_smooth.replace(0, np.nan))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1/window, min_periods=window, adjust=False).mean().fillna(0)

def calculate_rsi_series(series: pd.Series, window=14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

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
    mdd = (wealth / wealth.cummax() - 1.0).min()
    return {
        "total": total_return * 100.0,
        "cagr": cagr * 100.0 if cagr else None,
        "vol_ann": vol_ann * 100.0 if vol_ann else None,
        "mdd": mdd * 100.0 if mdd else None,
    }

def compute_sharpe_ratio(s: pd.Series, rf_annual_pct: float = 0.0) -> Optional[float]:
    s_clean = _as_price_series(s)
    if s_clean is None or len(s_clean) < 3: return None
    returns = s_clean.pct_change().dropna()
    rf_daily = (rf_annual_pct / 100.0) / 252.0
    excess = returns - rf_daily
    std = excess.std(ddof=1)
    return float((excess.mean() / std) * np.sqrt(252)) if std > 0 else None

def compute_sortino_ratio(s: pd.Series, rf_annual_pct: float = 0.0) -> Optional[float]:
    s_clean = _as_price_series(s)
    if s_clean is None: return None
    returns = s_clean.pct_change().dropna()
    rf_daily = (rf_annual_pct / 100.0) / 252.0
    excess = returns - rf_daily
    downside_std = excess[excess < 0].std(ddof=1)
    return float((excess.mean() / downside_std) * np.sqrt(252)) if downside_std > 0 else None

def compute_omega_ratio(s: pd.Series, required_return_pct: float = 0.0) -> Optional[float]:
    s_clean = _as_price_series(s)
    if s_clean is None: return None
    returns = s_clean.pct_change().dropna()
    thresh = (required_return_pct / 100.0) / 252.0
    gains = (returns - thresh).clip(lower=0).sum()
    losses = (thresh - returns).clip(lower=0).sum()
    return float(gains / losses) if losses > 0 else None

def compute_var(s: pd.Series, confidence_level: float = 0.95) -> Optional[float]:
    s_clean = _as_price_series(s)
    if s_clean is None: return None
    returns = s_clean.pct_change().dropna()
    return float(-returns.quantile(1 - confidence_level) * 100.0) if not returns.empty else None