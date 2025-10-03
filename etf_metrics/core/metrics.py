# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

VALIDATION_THRESHOLDS = {
    "min_daily_return": -0.5,
    "max_daily_return": 2.0,
    "min_annual_vol": 0.01,
    "max_annual_vol": 2.0,
    "min_price": 0.01,
    "max_sharpe": 10.0
}


def _normalize_timezone(s: pd.Series) -> pd.Series:
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s


def _safe_division(numerator: float, denominator: float, default: Optional[float] = None) -> Optional[float]:
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


def _as_price_series(s: pd.Series) -> Optional[pd.Series]:
    if s is None: return None
    s = _normalize_timezone(s).sort_index()
    # Ensure volume is also numeric if present
    if isinstance(s, pd.DataFrame) and 'Volume' in s.columns:
        s['Volume'] = pd.to_numeric(s['Volume'], errors='coerce')
        s = s.dropna(subset=['Close'])  # Drop rows where Close is NaN

    values = pd.to_numeric(s if isinstance(s, pd.Series) else s['Close'], errors='coerce').replace([np.inf, -np.inf],
                                                                                                   np.nan).dropna()
    values = values[values > 0]

    if isinstance(s, pd.DataFrame):
        cleaned = s.loc[values.index].copy()
        cleaned['Close'] = values
    else:
        cleaned = pd.Series(values.values, index=values.index).sort_index()

    if len(cleaned) < 2: return None
    return cleaned


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


def _clip_extreme_returns(returns: pd.Series) -> pd.Series:
    lo = VALIDATION_THRESHOLDS["min_daily_return"]
    hi = VALIDATION_THRESHOLDS["max_daily_return"]
    return returns.clip(lower=lo, upper=hi)


def _validate_series(s: pd.Series, min_length: int = 2) -> bool:
    return s is not None and len(s) >= min_length


def compute_metrics_from_series(s: pd.Series) -> Dict[str, Optional[float]]:
    if not _validate_series(s):
        return {}
    s_clean = _as_price_series(s)
    if s_clean is None: return {}

    total_return = (s_clean.iloc[-1] / s_clean.iloc[0]) - 1.0

    years = (s_clean.index[-1] - s_clean.index[0]).days / 365.25 if isinstance(s_clean.index,
                                                                               pd.DatetimeIndex) else len(
        s_clean) / 252.0
    cagr = ((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else None

    returns = s_clean.pct_change().dropna()
    per_year = _infer_periods_per_year(s_clean)
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
    s_clean = _as_price_series(s)
    if s_clean is None or len(s_clean) < 3: return None
    returns = s_clean.pct_change().dropna()
    if len(returns) < 2: return None

    per_year = _infer_periods_per_year(s_clean)
    rf_per_period = (rf_annual_pct / 100.0) / per_year

    excess_returns = _clip_extreme_returns(returns) - rf_per_period
    mean_excess = excess_returns.mean()
    std_dev = excess_returns.std(ddof=1)

    sharpe = (mean_excess / std_dev) * np.sqrt(per_year) if std_dev > 0 else None
    return float(sharpe) if sharpe is not None and np.isfinite(sharpe) else None


def compute_sortino_ratio(s: pd.Series, rf_annual_pct: float = 0.0) -> Optional[float]:
    """Calcola il Sortino Ratio, che penalizza solo la volatilità negativa."""
    s_clean = _as_price_series(s)
    if s_clean is None or len(s_clean) < 3: return None
    returns = s_clean.pct_change().dropna()
    if len(returns) < 2: return None

    per_year = _infer_periods_per_year(s_clean)
    rf_per_period = (rf_annual_pct / 100.0) / per_year

    excess_returns = returns - rf_per_period
    mean_excess = excess_returns.mean()

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std(ddof=1)

    if downside_std == 0 or np.isnan(downside_std):
        return np.inf if mean_excess > 0 else 0.0

    sortino = (mean_excess / downside_std) * np.sqrt(per_year)
    return float(sortino) if sortino is not None and np.isfinite(sortino) else None


def compute_omega_ratio(s: pd.Series, required_return_pct: float = 0.0) -> Optional[float]:
    """Calcola l'Omega Ratio, che misura il rapporto tra guadagni e perdite ponderati."""
    s_clean = _as_price_series(s)
    if s_clean is None or len(s_clean) < 3: return None
    returns = s_clean.pct_change().dropna()
    if returns.empty: return None

    threshold = (required_return_pct / 100.0) / _infer_periods_per_year(s)

    gains = (returns - threshold).where(returns > threshold, 0).sum()
    losses = (threshold - returns).where(returns < threshold, 0).sum()

    return _safe_division(gains, losses)


def compute_var(s: pd.Series, confidence_level: float = 0.95, holding_period_days: int = 1) -> Optional[float]:
    """Calcola il Value at Risk (VaR) storico."""
    s_clean = _as_price_series(s)
    if s_clean is None: return None
    returns = s_clean.pct_change().dropna()
    if returns.empty: return None

    daily_var = -returns.quantile(1 - confidence_level)
    var_period = daily_var * np.sqrt(holding_period_days)

    return float(var_period * 100.0) if np.isfinite(var_period) else None


def compute_current_drawdown(s: pd.Series) -> Optional[float]:
    s_clean = _as_price_series(s)
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


def get_trend_signal(series: pd.Series) -> str:
    """
    Genera un segnale di trend basato su medie mobili a 50 e 200 giorni.
    """
    if series is None or len(series) < 200:
        return "Dati Insufficienti"

    sma50 = series.rolling(window=50).mean()
    sma200 = series.rolling(window=200).mean()

    last_price = series.iloc[-1]
    last_sma50 = sma50.iloc[-1]
    last_sma200 = sma200.iloc[-1]

    if pd.isna(last_sma50) or pd.isna(last_sma200):
        return "Dati Insufficienti"

    if last_sma50 > last_sma200:
        if last_price > last_sma50:
            return "Mantieni (Trend Forte)"
        else:
            return "Monitora (Trend in Indebolimento)"
    elif last_sma50 < last_sma200:
        return "Valuta Vendita (Death Cross)"
    else:
        return "Laterale"

def get_tactical_buy_signal_v2(series_df: pd.DataFrame) -> Tuple[Optional[Dict[str, str]], List[str]]:
    """
    Genera un segnale di acquisto tattico basato su momentum e volumi, con logging dettagliato.
    """
    ticker = series_df.attrs.get('ticker', 'N/A')
    logs = [f"⚪ **{ticker}**: Inizio analisi..."]

    if not isinstance(series_df, pd.DataFrame) or 'Close' not in series_df.columns or len(series_df) < 30:
        logs.append("❌ Dati storici insufficienti (meno di 30 giorni).")
        return None, logs

    # Indicatori
    price = series_df['Close'].iloc[-1]
    sma10 = series_df['Close'].rolling(window=10).mean().iloc[-1]
    volume = series_df['Volume'].iloc[-1]
    volume_sma20 = series_df['Volume'].rolling(window=20).mean().iloc[-1]
    roc_5 = (series_df['Close'].iloc[-1] / series_df['Close'].iloc[-6] - 1) * 100 if len(series_df) > 5 else 0

    # Condizioni
    price_above_sma = price > sma10
    volume_above_sma = volume > volume_sma20
    roc_positive = roc_5 > 0

    # Logging
    logs.append(f"  - Prezzo > SMA10: **{'Sì' if price_above_sma else 'No'}** (Prezzo: {price:.2f}, SMA10: {sma10:.2f})")
    logs.append(f"  - Volume > Media Volume 20gg: **{'Sì' if volume_above_sma else 'No'}** (Volume: {volume:,.0f}, Media: {volume_sma20:,.0f})")
    logs.append(f"  - ROC 5gg > 0: **{'Sì' if roc_positive else 'No'}** (ROC: {roc_5:.2f}%)")

    if price_above_sma and volume_above_sma and roc_positive:
        signal = {
            "signal": "Compra",
            "reason": f"Momentum positivo con Prezzo > SMA10, Volume > Media e ROC 5gg a +{roc_5:.2f}%."
        }
        logs.append(f"✅ **Segnale di ACQUISTO trovato!**")
        return signal, logs

    logs.append("❌ Nessuna condizione di acquisto soddisfatta.")
    return None, logs

def monitor_simulated_trade_v2(
        purchase_price: float,
        series: pd.Series,
        take_profit_pct: float,
        stop_loss_pct: float,
        commission: float,
        tax_rate: float
) -> Optional[Dict[str, str]]:
    """Monitora una posizione aperta e genera segnali di vendita con logica fiscale."""
    if series is None or len(series) < 2:
        return None

    current_price = series.iloc[-1]
    gross_pnl = current_price - purchase_price

    # 1. Take Profit
    if current_price >= purchase_price * (1 + take_profit_pct / 100):
        return {"signal": "Vendi (Take Profit)", "reason": f"Obiettivo di profitto (+{take_profit_pct}%) raggiunto."}

    # 2. Stop Loss
    if current_price <= purchase_price * (1 - stop_loss_pct / 100):
        return {"signal": "Vendi (Stop Loss)", "reason": f"Stop loss (-{stop_loss_pct}%) raggiunto."}

    # 3. Condizione Fiscale e Commissioni
    if gross_pnl > 0:
        taxable_amount = gross_pnl
        tax_paid = taxable_amount * (tax_rate / 100.0)
        net_pnl = gross_pnl - commission - tax_paid
        if net_pnl <= 0: # Se il profitto non copre tasse e commissioni, non è un segnale di vendita "ideale"
             pass # Potresti voler aggiungere una logica qui, ma per ora non genera segnale
    else: # Se in perdita, non ci sono tasse sulla plusvalenza
        net_pnl = gross_pnl - commission

    # Esempio di segnale di vendita basato sul PNL Netto (puoi adattarlo)
    # Questa è una logica di esempio, potresti volerla più complessa
    # Per ora, la vendita è guidata solo da TP e SL.

    # Potresti aggiungere altre condizioni qui, es. incrocio ribassista di medie mobili
    sma10 = series.rolling(window=10).mean().iloc[-1]
    if current_price < sma10:
         return {"signal": "Vendi (Segnale Tecnico)", "reason": "Il prezzo è sceso sotto la media mobile a 10 giorni."}

    return None