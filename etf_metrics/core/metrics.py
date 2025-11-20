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
    if isinstance(s, pd.DataFrame) and 'Volume' in s.columns:
        s['Volume'] = pd.to_numeric(s['Volume'], errors='coerce')
        s = s.dropna(subset=['Close'])

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
    """
    Genera segnali operativi inclusi scenari Short/Vendita.
    """
    if series is None or len(series) < 200:
        return {"signal": "Dati Insufficienti", "reason": "Servono almeno 200 giorni di storico."}

    current_price = series.iloc[-1]
    indicators = calculate_technical_indicators(series)
    sma50, sma200, rsi = indicators.get('sma50'), indicators.get('sma200'), indicators.get('rsi')

    if any(v is None or np.isnan(v) for v in [sma50, sma200, rsi]):
        return {"signal": "Non Disponibile", "reason": "Impossibile calcolare indicatori tecnici."}

    is_uptrend = sma50 > sma200
    is_price_above_sma50 = current_price > sma50
    is_price_above_sma200 = current_price > sma200

    if is_uptrend and is_price_above_sma50 and rsi < 70:
        return {
            "signal": "Compra (Trend Follow)",
            "reason": "Trend rialzista solido (Prezzo > SMA50 > SMA200) e RSI non in eccesso."
        }

    if is_uptrend and not is_price_above_sma50 and rsi < 40:
        return {
            "signal": "Compra (Dip)",
            "reason": "Il prezzo ha ritracciato in un trend rialzista (Oversold). Possibile rimbalzo."
        }

    if not is_uptrend and not is_price_above_sma200:
        return {
            "signal": "Vendi / Short (Trend Ribassista)",
            "reason": "Trend negativo confermato (SMA50 < SMA200) e prezzo sotto la media a 200 giorni."
        }

    if rsi > 75:
        return {
            "signal": "Vendi / Short (Ipercomprato)",
            "reason": f"RSI estremo ({rsi:.1f}). Probabilità statistica di correzione o pullback imminente."
        }

    if is_uptrend and current_price < sma50 and current_price > sma200:
        return {
            "signal": "Chiudi Long / Attenzione",
            "reason": "Il prezzo ha rotto al ribasso la SMA50. Il momentum rialzista si sta indebolendo."
        }

    trend_status = "Positivo" if is_uptrend else "Negativo"
    rsi_status = "Alto" if rsi >= 60 else "Basso" if rsi <= 40 else "Neutrale"

    return {
        "signal": "Mantieni / Laterale",
        "reason": f"Nessun segnale direzionale chiaro. Trend: {trend_status}. RSI: {rsi_status} ({rsi:.1f})."
    }

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


def get_monday_buy_signal(series_df: pd.DataFrame, vix_series: pd.Series) -> Tuple[Optional[Dict[str, str]], List[str]]:
    """
    Genera un segnale di acquisto di lunedì basato su una confluenza di fattori
    quantitativi ed economici.
    """
    ticker = series_df.attrs.get('ticker', 'N/A')
    logs = [f"⚪ **{ticker}**: Analisi per 'Monday Buy'..."]

    today = series_df.index[-1]
    if today.weekday() != 0:
       logs.append("❌ Non è lunedì.")
       return None, logs

    if not isinstance(series_df, pd.DataFrame) or 'Close' not in series_df.columns or len(series_df) < 51:
        logs.append("❌ Dati storici insufficienti (meno di 51 giorni).")
        return None, logs

    price = series_df['Close'].iloc[-1]
    sma50 = series_df['Close'].rolling(window=50).mean().iloc[-1]
    volume = series_df['Volume'].iloc[-1]
    volume_sma20 = series_df['Volume'].rolling(window=20).mean().iloc[-1]

    fridays = series_df.index[(series_df.index < today) & (series_df.index.weekday == 4)]
    if fridays.empty:
        logs.append("❌ Impossibile trovare il venerdì precedente.")
        return None, logs
    last_friday = fridays[-1]

    friday_return = (series_df.loc[last_friday, 'Close'] / series_df.loc[last_friday, 'Open'] - 1) * 100
    current_vix = vix_series.iloc[-1]

    is_risk_on = current_vix < 25
    is_uptrend = price > sma50
    had_friday_dip = friday_return < 0
    has_volume_confirmation = volume > volume_sma20

    logs.append(f"  - Regime di Rischio: **{'Favorevole' if is_risk_on else 'Avverso'}** (VIX: {current_vix:.2f})")
    logs.append(
        f"  - Trend di Medio Termine: **{'Rialzista' if is_uptrend else 'Ribassista'}** (Prezzo: {price:.2f}, SMA50: {sma50:.2f})")
    logs.append(
        f"  - 'Effetto Weekend': **{'Sì' if had_friday_dip else 'No'}** (Rendimento Venerdì: {friday_return:.2f}%)")
    logs.append(
        f"  - Conferma Volumi: **{'Sì' if has_volume_confirmation else 'No'}** (Volume: {volume:,.0f}, Media 20gg: {volume_sma20:,.0f})")

    if is_risk_on and is_uptrend and had_friday_dip and has_volume_confirmation:
        signal = {
            "signal": "Compra (Monday Buy)",
            "reason": f"Confluenza di segnali: trend rialzista, VIX basso, dip di venerdì e volumi in aumento."
        }
        logs.append(f"✅ **Segnale di ACQUISTO 'Monday Buy' trovato!**")
        return signal, logs

    logs.append("❌ Nessuna condizione di acquisto soddisfatta.")
    return None, logs


def monitor_weekly_trade(
        purchase_price: float,
        series: pd.Series,
        commission: float,
        tax_rate: float,
        stop_loss_pct: float
) -> Optional[Dict[str, str]]:
    """
    Monitora una posizione aperta con la strategia settimanale.
    Genera segnali di vendita il venerdì o per gestione del rischio.
    """
    if series is None or len(series) < 2:
        return None

    current_price = series.iloc[-1]
    today = series.index[-1]

    stop_loss_price = purchase_price * (1 - stop_loss_pct / 100)
    if current_price <= stop_loss_price:
        return {"signal": "Vendi (Stop Loss)", "reason": f"Stop loss (-{stop_loss_pct}%) raggiunto."}

    if len(series) > 10:
        sma10 = series.rolling(window=10).mean().iloc[-1]
        if current_price < sma10:
            return {"signal": "Vendi (Segnale Tecnico)",
                    "reason": "Il prezzo è sceso sotto la media mobile a 10 giorni."}

    if today.weekday() == 4:
        gross_pnl_per_share = current_price - purchase_price

        tax_per_share = max(0, gross_pnl_per_share) * (tax_rate / 100.0)

        net_pnl_per_share = gross_pnl_per_share - (commission * 2) - tax_per_share

        if net_pnl_per_share > 0:
            return {"signal": "Vendi (Fine Settimana)",
                    "reason": f"Obiettivo settimanale raggiunto con profitto netto."}
        else:
            return {"signal": "Mantieni (Non Profittevole)",
                    "reason": f"Venerdì, ma la vendita non copre costi e tasse."}

    return None

# Importa get_series se non è già presente nel file metrics.py
from etf_metrics.clients.yahoo_client import get_series
import pandas as pd
import numpy as np # Assicurati che numpy sia importato

# ... (altro codice in metrics.py) ...

def calculate_atr(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    """Calcola l'Average True Range (ATR)."""
    if df is None or not all(col in df.columns for col in ['High', 'Low', 'Close']) or len(df) < window + 1: # Aggiunto +1 per shift
        return None

    high_low = df['High'] - df['Low']
    high_close_prev = np.abs(df['High'] - df['Close'].shift())
    low_close_prev = np.abs(df['Low'] - df['Close'].shift())

    # Combina le tre componenti del True Range, ignorando NaN per il primo giorno
    tr_df = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev})
    tr = tr_df.max(axis=1)
    tr = tr.dropna() # Rimuovi il primo NaN risultante dallo shift

    if len(tr) < window:
        return None # Non abbastanza dati per la media mobile

    # Calcola l'ATR usando Simple Moving Average (SMA)
    atr = tr.rolling(window=window, min_periods=window).mean().iloc[-1]
    # Alternativa: Exponential Moving Average (EMA)
    # atr = tr.ewm(span=window, adjust=False).mean().iloc[-1]

    return atr if pd.notna(atr) and np.isfinite(atr) else None

def get_dynamic_atr_multiplier(vix_value: Optional[float]) -> float:
    """Determina il moltiplicatore ATR basato sul VIX."""
    if vix_value is None or pd.isna(vix_value):
        return 3.0 # Default se VIX non disponibile
    elif vix_value < 15:
        return 2.5 # Mercato calmo, stop più stretto
    elif vix_value < 25:
        return 3.0 # Volatilità normale
    elif vix_value < 35:
        return 3.5 # Volatilità elevata
    else:
        return 4.0 # Volatilità molto alta, stop più largo

def get_satellite_signal(series_df: pd.DataFrame, vix_series: pd.Series, atr_window: int = 20) -> Dict[str, str]:
    """Genera segnale per asset satellite usando trailing stop ATR dinamico basato su VIX."""
    if series_df is None or len(series_df) < atr_window + 1 or 'Close' not in series_df.columns:
         return {"signal": "Dati Insufficienti", "reason": f"Servono almeno {atr_window+1} giorni di storico OHLC."}

    current_price = series_df['Close'].iloc[-1]
    atr = calculate_atr(series_df, window=atr_window)

    if atr is None:
         return {"signal": "Dati Insufficienti", "reason": f"Impossibile calcolare ATR({atr_window})."}

    # Determina moltiplicatore dinamico
    current_vix = vix_series.iloc[-1] if vix_series is not None and not vix_series.empty else None
    atr_multiplier = get_dynamic_atr_multiplier(current_vix)
    vix_info = f"(VIX: {current_vix:.2f}, Multiplier: {atr_multiplier:.1f})" if current_vix is not None else "(VIX N/D, Multiplier: Default)"

    # Calcola lo stop level
    lookback_period_high = atr_window * 2 # Periodo per trovare il massimo recente
    if len(series_df) < lookback_period_high:
        recent_high = series_df['High'].max() # Usa il massimo disponibile se lo storico è breve
    else:
        recent_high = series_df['High'].rolling(window=lookback_period_high).max().iloc[-1]

    if pd.isna(recent_high):
         return {"signal": "Dati Insufficienti", "reason": "Impossibile determinare il massimo recente."}

    stop_level = recent_high - atr_multiplier * atr

    if current_price < stop_level:
        return {
            "signal": f"Vendi (ATR Dinamico)",
            "reason": f"Prezzo sotto trailing stop ({stop_level:.2f}). {vix_info}"
            #"reason": f"Prezzo sotto trailing stop ({stop_level:.2f}) basato su Max recente={recent_high:.2f}, ATR({atr_window})={atr:.2f}. {vix_info}"
        }
    else:
         return {
             "signal": "Mantieni (Trend Forte)",
             "reason": f"Prezzo sopra trailing stop ({stop_level:.2f}). {vix_info}"
         }