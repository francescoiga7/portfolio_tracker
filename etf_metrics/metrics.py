# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.stats import norm

log = logging.getLogger(__name__)


class MetricsCalculator:
    VALIDATION_THRESHOLDS = {
        "min_daily_return": -0.5,
        "max_daily_return": 2.0,
        "min_annual_vol": 0.01,
        "max_annual_vol": 2.0,
        "min_price": 0.01,
        "max_sharpe": 10.0
    }

    @staticmethod
    def _normalize_timezone(s: pd.Series) -> pd.Series:
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s = s.copy()
            s.index = s.index.tz_localize(None)
        return s

    @staticmethod
    def _safe_division(numerator: float, denominator: float, default: Optional[float] = None,
                       operation_name: str = "division") -> Optional[float]:
        try:
            if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                return default
            if np.isnan(numerator) or np.isinf(numerator):
                return default
            result = numerator / denominator
            if np.isnan(result) or np.isinf(result):
                return default
            return float(result)
        except Exception:
            return default

    @staticmethod
    def _as_price_series(s: pd.Series, series_name: str = "unnamed") -> Optional[pd.Series]:
        if s is None: return None
        s = MetricsCalculator._normalize_timezone(s).sort_index()
        values = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        values = values[values > 0]
        cleaned = pd.Series(values.values, index=values.index).sort_index()
        if len(cleaned) < 2: return None
        return cleaned

    @staticmethod
    def _infer_periods_per_year(s: pd.Series) -> float:
        default = 252.0
        if len(s) < 2 or not isinstance(s.index, pd.DatetimeIndex):
            return default
        elapsed_days = (s.index[-1] - s.index[0]).days
        years = elapsed_days / 365.25
        returns = s.pct_change().dropna()
        if years > 0 and len(returns) > 0:
            return float(np.clip(len(returns) / years, 50.0, 520.0))
        return default

    @staticmethod
    def _clip_extreme_returns(returns: pd.Series) -> pd.Series:
        lo = MetricsCalculator.VALIDATION_THRESHOLDS["min_daily_return"]
        hi = MetricsCalculator.VALIDATION_THRESHOLDS["max_daily_return"]
        return returns.clip(lower=lo, upper=hi)

    @staticmethod
    def _validate_series(s: pd.Series, min_length: int = 2, series_name: str = "unnamed") -> bool:
        return s is not None and len(s) >= min_length

def compute_metrics_from_series(s: pd.Series) -> Dict[str, Optional[float]]:
    if not MetricsCalculator._validate_series(s):
        return {}
    s_clean = MetricsCalculator._as_price_series(s)
    if s_clean is None: return {}

    total_return = (s_clean.iloc[-1] / s_clean.iloc[0]) - 1.0

    years = (s_clean.index[-1] - s_clean.index[0]).days / 365.25 if isinstance(s_clean.index,
                                                                               pd.DatetimeIndex) else len(
        s_clean) / 252.0
    cagr = ((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else None

    returns = s_clean.pct_change().dropna()
    per_year = MetricsCalculator._infer_periods_per_year(s_clean)
    vol_ann = returns.std(ddof=1) * np.sqrt(per_year) if len(returns) >= 2 else None

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
    s_clean = MetricsCalculator._as_price_series(s)
    if s_clean is None or len(s_clean) < 3: return None
    returns = s_clean.pct_change().dropna()
    if len(returns) < 2: return None

    per_year = MetricsCalculator._infer_periods_per_year(s_clean)
    rf_per_period = (rf_annual_pct / 100.0) / per_year

    excess_returns = MetricsCalculator._clip_extreme_returns(returns) - rf_per_period
    mean_excess = excess_returns.mean()
    std_dev = excess_returns.std(ddof=1)

    sharpe = (mean_excess / std_dev) * np.sqrt(per_year) if std_dev > 0 else None
    return float(sharpe) if sharpe is not None and np.isfinite(sharpe) else None


def compute_sortino_ratio(s: pd.Series, rf_annual_pct: float = 0.0) -> Optional[float]:
    """Calcola il Sortino Ratio, che penalizza solo la volatilità negativa."""
    s_clean = MetricsCalculator._as_price_series(s)
    if s_clean is None or len(s_clean) < 3: return None
    returns = s_clean.pct_change().dropna()
    if len(returns) < 2: return None

    per_year = MetricsCalculator._infer_periods_per_year(s_clean)
    rf_per_period = (rf_annual_pct / 100.0) / per_year

    excess_returns = returns - rf_per_period
    mean_excess = excess_returns.mean()

    # Calcola la deviazione standard solo dei rendimenti negativi (downside deviation)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std(ddof=1)

    if downside_std == 0 or np.isnan(downside_std):
        return np.inf if mean_excess > 0 else 0.0

    sortino = (mean_excess / downside_std) * np.sqrt(per_year)
    return float(sortino) if sortino is not None and np.isfinite(sortino) else None


def compute_omega_ratio(s: pd.Series, required_return_pct: float = 0.0) -> Optional[float]:
    """Calcola l'Omega Ratio, che misura il rapporto tra guadagni e perdite ponderati."""
    s_clean = MetricsCalculator._as_price_series(s)
    if s_clean is None or len(s_clean) < 3: return None
    returns = s_clean.pct_change().dropna()
    if returns.empty: return None

    threshold = (required_return_pct / 100.0) / MetricsCalculator._infer_periods_per_year(s)

    gains = (returns - threshold).where(returns > threshold, 0).sum()
    losses = (threshold - returns).where(returns < threshold, 0).sum()

    return MetricsCalculator._safe_division(gains, losses)


def compute_var(s: pd.Series, confidence_level: float = 0.95, holding_period_days: int = 1) -> Optional[float]:
    """Calcola il Value at Risk (VaR) storico."""
    s_clean = MetricsCalculator._as_price_series(s)
    if s_clean is None: return None
    returns = s_clean.pct_change().dropna()
    if returns.empty: return None

    # Calcola il VaR per un giorno
    daily_var = -returns.quantile(1 - confidence_level)

    # Estrapola per il periodo di detenzione (es. 10 giorni)
    var_period = daily_var * np.sqrt(holding_period_days)

    return float(var_period * 100.0) if np.isfinite(var_period) else None


def compute_current_drawdown(s: pd.Series) -> Optional[float]:
    s_clean = MetricsCalculator._as_price_series(s)
    if s_clean is None: return None
    rel = s_clean / s_clean.cummax()
    return float((rel.iloc[-1] - 1.0) * 100.0)


def calculate_technical_indicators(series: pd.Series) -> Dict[str, float]:
    indicators = {'sma50': None, 'sma200': None, 'rsi': None}
    if series is None or len(series) < 200: return indicators
    indicators['sma50'] = series.rolling(window=50).mean().iloc[-1]
    indicators['sma200'] = series.rolling(window=200).mean().iloc[-1]
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    indicators['rsi'] = rsi.iloc[-1]
    return indicators


def get_trading_signal(series: pd.Series) -> Dict[str, str]:
    if series is None or len(series) < 200:
        return {"signal": "Dati Insufficienti", "reason": "Servono almeno 200 giorni di storico."}

    current_price = series.iloc[-1]
    indicators = calculate_technical_indicators(series)
    sma50, sma200, rsi = indicators.get('sma50'), indicators.get('sma200'), indicators.get('rsi')

    if any(v is None or np.isnan(v) for v in [sma50, sma200, rsi]):
        return {"signal": "Non Disponibile", "reason": "Impossibile calcolare indicatori."}

    is_uptrend = sma50 > sma200
    is_price_above_sma50 = current_price > sma50

    if is_uptrend and is_price_above_sma50 and rsi < 70:
        return {"signal": "Compra Ora",
                "reason": "Trend rialzista (SMA50>SMA200), prezzo sopra SMA50 e non in ipercomprato."}
    elif not is_uptrend and not is_price_above_sma50:
        return {"signal": "Vendi Ora", "reason": "Incrocio ribassista (SMA50<SMA200) e prezzo sotto SMA50."}

    reason = f"Trend: {'Positivo' if is_uptrend else 'Negativo'}. RSI: {'Ipercomprato' if rsi >= 70 else 'Ipervenduto' if rsi <= 30 else 'Neutrale'}."
    return {"signal": "Mantieni/Monitora", "reason": reason}