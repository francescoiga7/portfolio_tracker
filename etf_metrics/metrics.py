# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from datetime import timedelta

log = logging.getLogger(__name__)


class MetricsCalculator:
    VALIDATION_THRESHOLDS = {
        "min_daily_return": -0.5,  # -50% max perdita giornaliera "plausibile"
        "max_daily_return": 2.0,  # +200% max guadagno giornaliero "plausibile"
        "min_annual_vol": 0.01,  # 1% volatilità minima annua (per flaggare dati sospetti)
        "max_annual_vol": 2.0,  # 200% volatilità massima annua (per flaggare dati sospetti)
        "min_price": 0.01,  # Prezzo minimo realistico
        "max_sharpe": 10.0  # Sharpe ratio massimo realistico
    }

    # -----------------------------
    # Helpers interni
    # -----------------------------
    @staticmethod
    def _normalize_timezone(s: pd.Series) -> pd.Series:
        """Rende tz-naive l'indice datetime (se presente) per evitare conflitti."""
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s = s.copy()
            s.index = s.index.tz_localize(None)
        return s

    @staticmethod
    def _safe_division(numerator: float, denominator: float, default: Optional[float] = None,
                       operation_name: str = "division") -> Optional[float]:
        """Divisione sicura con gestione errori e logging."""
        try:
            if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                log.debug(f"{operation_name}: denominatore non valido ({denominator})")
                return default
            if np.isnan(numerator) or np.isinf(numerator):
                log.debug(f"{operation_name}: numeratore non valido ({numerator})")
                return default
            result = numerator / denominator
            if np.isnan(result) or np.isinf(result):
                log.debug(f"{operation_name}: risultato non valido ({result})")
                return default
            return float(result)
        except Exception as e:
            log.error(f"{operation_name}: errore nella divisione: {e}")
            return default

    @staticmethod
    def _as_price_series(s: pd.Series, series_name: str = "unnamed") -> Optional[pd.Series]:
        """
        Prepara la serie prezzi per i calcoli: coercizione numerica, rimozione inf/NaN,
        rimozione di valori non-positivi, ordinamento e normalizzazione timezone.
        Non riassegna frequenze né filla buchi: lascia il dato com'è (più trasparente).
        """
        if s is None:
            log.warning(f"Serie {series_name}: None")
            return None

        # Copia, normalizza tz, ordina indice
        s = MetricsCalculator._normalize_timezone(s)
        try:
            s = s.sort_index()
        except Exception:
            # Se l'indice non è ordinabile (es. misto tipi), fallback a reset
            s = s.copy()

        # Coercizione numerica e pulizia
        values = pd.to_numeric(s, errors='coerce')
        n_before = len(values)
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        n_after_dropna = len(values)

        # Rimuove valori non-positivi (prezzi devono essere > 0)
        non_pos = values[values <= 0]
        if not non_pos.empty:
            log.warning(f"Serie {series_name}: rimossi {len(non_pos)} valori non-positivi")
            values = values[values > 0]

        # Riallinea con indice originale mantenendo l'ordine
        cleaned = pd.Series(values.values, index=values.index)
        cleaned = cleaned.sort_index()

        if len(cleaned) < 2:
            log.warning(f"Serie {series_name}: dati insufficienti dopo pulizia "
                        f"({len(cleaned)} < 2; rimossi NaN/non-positivi: {n_before - n_after_dropna + len(non_pos)})")
            return None

        # Check e avvisi sulla continuità temporale solo se DatetimeIndex
        if isinstance(cleaned.index, pd.DatetimeIndex) and len(cleaned) > 1:
            diffs = cleaned.index.to_series().diff().dropna()
            if not diffs.empty and diffs.max() > timedelta(days=30):
                log.info(f"Serie {series_name}: gap temporali significativi rilevati (max gap: {diffs.max()})")

        return cleaned

    @staticmethod
    def _infer_periods_per_year(s: pd.Series) -> float:
        """
        Stima robusta dei periodi per anno per annualizzazione:
        - Se DatetimeIndex: usa (#returns / anni effettivi).
        - Altrimenti: fallback a 252.
        Cap limit [50, 520] per evitare estremi.
        """
        default = 252.0
        if len(s) < 2:
            return default

        if isinstance(s.index, pd.DatetimeIndex):
            elapsed_days = (s.index[-1] - s.index[0]).days
            years = elapsed_days / 365.25 if elapsed_days > 0 else 0.0
            returns = s.pct_change().dropna()
            if years > 0 and len(returns) > 0:
                per_year = len(returns) / years
                # Limiti ragionevoli (settimanale ~52, daily ~252, intraday ignorato)
                return float(np.clip(per_year, 50.0, 520.0))
        return default

    @staticmethod
    def _clip_extreme_returns(returns: pd.Series) -> pd.Series:
        """Clip dei rendimenti a soglie plausibili per robustezza (non modifica i prezzi)."""
        lo = MetricsCalculator.VALIDATION_THRESHOLDS["min_daily_return"]
        hi = MetricsCalculator.VALIDATION_THRESHOLDS["max_daily_return"]
        clipped = returns.clip(lower=lo, upper=hi)
        n_clipped = (clipped != returns).sum()
        if n_clipped > 0:
            log.info(f"Rendimenti estremi clip: {n_clipped} osservazioni fuori [{lo}, {hi}]")
        return clipped

    @staticmethod
    def _validate_series(s: pd.Series, min_length: int = 2, series_name: str = "unnamed") -> bool:
        """Valida input base (pre-pulizia) per evitare crash precoci."""
        if s is None or len(s) < min_length:
            log.warning(
                f"Serie {series_name}: lunghezza insufficiente ({len(s) if s is not None else 0} < {min_length})")
            return False
        return True

    @staticmethod
    def _detect_outliers(s: pd.Series, method: str = "iqr") -> List[pd.Timestamp]:
        """Rileva outlier (per diagnosi). Ritorna gli indici corrispondenti."""
        if s.empty:
            return []
        # Lavora sui valori (non sui rendimenti)
        x = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            return []
        if method == "iqr":
            Q1 = x.quantile(0.25)
            Q3 = x.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return x[(x < lower_bound) | (x > upper_bound)].index.tolist()
        elif method == "zscore":
            std = x.std(ddof=1)
            if std == 0 or np.isnan(std):
                return []
            z_scores = np.abs((x - x.mean()) / std)
            return x[z_scores > 3].index.tolist()
        return []


def compute_metrics_from_series(s: pd.Series) -> Dict[str, Optional[float]]:
    """
    Calcola metriche di base da una serie di prezzi:
    - total (%), cagr (%), vol_ann (%), mdd (%)
    Robusto a NaN/inf, indice non-datetime, outlier nei rendimenti, frequenze non-daily.
    """
    if not MetricsCalculator._validate_series(s):
        return dict(total=None, cagr=None, vol_ann=None, mdd=None)

    try:
        # Pulizia e normalizzazione
        s_clean = MetricsCalculator._as_price_series(s, series_name=getattr(s, "name", "unnamed"))
        if s_clean is None or len(s_clean) < 2:
            return dict(total=None, cagr=None, vol_ann=None, mdd=None)

        # Rendimento totale (%)
        total_return = MetricsCalculator._safe_division(
            (s_clean.iloc[-1] - s_clean.iloc[0]), s_clean.iloc[0], default=None, operation_name="total_return"
        )
        total = total_return * 100.0 if total_return is not None else None

        # Anni effettivi per CAGR
        if isinstance(s_clean.index, pd.DatetimeIndex):
            delta_days = (s_clean.index[-1] - s_clean.index[0]).days
            years = delta_days / 365.25 if delta_days > 0 else 0.0
        else:
            # Stima anni con 252 come fallback
            years = (len(s_clean) - 1) / 252.0

        cagr = None
        if years > 0 and total_return is not None:
            cagr_factor = (1.0 + total_return) ** (1.0 / years) - 1.0
            cagr = cagr_factor * 100.0 if not np.isnan(cagr_factor) else None

        # Volatilità annualizzata (%) con clipping dei rendimenti estremi
        returns = s_clean.pct_change().dropna()
        vol_ann = None
        if len(returns) >= 2:
            per_year = MetricsCalculator._infer_periods_per_year(s_clean)
            returns_clipped = MetricsCalculator._clip_extreme_returns(returns)
            std = float(returns_clipped.std(ddof=1))
            if std > 0 and not np.isnan(std):
                vol_ann = std * np.sqrt(per_year) * 100.0
                # Flag range plausibile (warning non bloccante)
                vol_min = MetricsCalculator.VALIDATION_THRESHOLDS["min_annual_vol"] * 100.0
                vol_max = MetricsCalculator.VALIDATION_THRESHOLDS["max_annual_vol"] * 100.0
                if (vol_ann < vol_min) or (vol_ann > vol_max):
                    log.info(f"Volatilità annualizzata fuori range plausibile: {vol_ann:.2f}%")

        # Maximum Drawdown (%) sul wealth relativo
        wealth = s_clean / s_clean.iloc[0]
        roll_max = wealth.cummax()
        drawdowns = wealth / roll_max - 1.0
        mdd = float(drawdowns.min()) * 100.0 if not drawdowns.empty else None
        return dict(total=total, cagr=cagr, vol_ann=vol_ann, mdd=mdd)

    except Exception as e:
        log.error(f"Errore nel calcolo delle metriche: {e}", exc_info=True)
        return dict(total=None, cagr=None, vol_ann=None, mdd=None)


def compute_sharpe_ratio(s: pd.Series, rf_annual_pct: float = 0.0) -> Optional[float]:
    """
    Sharpe ratio (ann.): usa rendimenti aritmetici, clipping outlier e annualizzazione
    coerente con la frequenza stimata. rf_annual_pct in percentuale (es. 3.0 = 3%)
    """
    if not MetricsCalculator._validate_series(s):
        return None

    try:
        s_clean = MetricsCalculator._as_price_series(s, series_name=getattr(s, "name", "unnamed"))
        if s_clean is None or len(s_clean) < 3:
            return None

        returns = s_clean.pct_change().dropna()
        if len(returns) < 2:
            return None

        per_year = MetricsCalculator._infer_periods_per_year(s_clean)
        rf_per_period = (rf_annual_pct / 100.0) / per_year if per_year > 0 else 0.0

        excess = MetricsCalculator._clip_extreme_returns(returns) - rf_per_period
        std = float(excess.std(ddof=1))
        mean = float(excess.mean())

        if std == 0 or np.isnan(std):
            return None

        sharpe = (mean / std) * np.sqrt(per_year)
        if np.isnan(sharpe) or np.isinf(sharpe):
            return None

        if abs(sharpe) > MetricsCalculator.VALIDATION_THRESHOLDS["max_sharpe"]:
            log.info(f"Sharpe fuori range plausibile: {sharpe:.2f}")

        return float(sharpe)

    except Exception as e:
        log.error(f"Errore nel calcolo dello Sharpe Ratio: {e}", exc_info=True)
        return None


def compute_current_drawdown(s: pd.Series) -> Optional[float]:
    """
    Drawdown corrente (%) rispetto al massimo storico precedente.
    """
    if not MetricsCalculator._validate_series(s):
        return None

    try:
        s_clean = MetricsCalculator._as_price_series(s, series_name=getattr(s, "name", "unnamed"))
        if s_clean is None or len(s_clean) < 2:
            return None

        rel = s_clean / s_clean.cummax()
        current_dd = rel.iloc[-1] - 1.0
        return float(current_dd * 100.0)

    except Exception as e:
        log.error(f"Errore nel calcolo del drawdown corrente: {e}", exc_info=True)
        return None


# --- NUOVE FUNZIONI INTEGRATE ---
def calculate_technical_indicators(series: pd.Series) -> Dict[str, float]:
    """Calcola SMA50, SMA200 e RSI a 14 periodi."""
    indicators = {'sma50': None, 'sma200': None, 'rsi': None}
    if series is None or len(series) < 200:
        return indicators

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
    """Genera un segnale di trading basato su SMA e RSI."""
    if series is None or len(series) < 200:
        return {"signal": "Dati Insufficienti",
                "reason": "Servono almeno 200 giorni di storico per calcolare gli indicatori."}

    current_price = series.iloc[-1]
    indicators = calculate_technical_indicators(series)
    sma50 = indicators.get('sma50')
    sma200 = indicators.get('sma200')
    rsi = indicators.get('rsi')

    if any(v is None or np.isnan(v) for v in [sma50, sma200, rsi]):
        return {"signal": "Non Disponibile", "reason": "Impossibile calcolare gli indicatori richiesti."}

    is_uptrend = sma50 > sma200
    is_price_above_sma50 = current_price > sma50

    if is_uptrend and is_price_above_sma50 and rsi < 70:
        return {"signal": "Compra Ora",
                "reason": "Trend rialzista (SMA50 > SMA200), prezzo sopra la media mobile a 50 giorni e non in ipercomprato (RSI < 70)."}
    elif not is_uptrend and not is_price_above_sma50:
        return {"signal": "Vendi Ora",
                "reason": "Incrocio ribassista (SMA50 < SMA200) e prezzo sotto la media mobile a 50 giorni."}

    reason_parts = []
    if is_uptrend:
        reason_parts.append("Il trend di fondo è positivo (SMA50 > SMA200).")
    else:
        reason_parts.append("Il trend di fondo è negativo (SMA50 < SMA200).")

    if rsi >= 70:
        reason_parts.append("L'asset è in zona di ipercomprato (RSI >= 70), suggerendo cautela.")
    elif rsi <= 30:
        reason_parts.append("L'asset è in zona di ipervenduto (RSI <= 30), possibile segnale di rimbalzo.")
    else:
        reason_parts.append("L'RSI è in zona neutrale (30 < RSI < 70).")

    return {"signal": "Mantieni/Monitora", "reason": " ".join(reason_parts)}