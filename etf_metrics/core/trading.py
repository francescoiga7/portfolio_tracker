# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, Optional
from etf_metrics.core.metrics import calculate_atr_series
from etf_metrics.clients.yahoo_client import get_series


def _get_latest_vix() -> Optional[float]:
    """
    Recupera l'ultimo valore di chiusura del VIX (^VIX).
    Restituisce None se il download fallisce.
    """
    try:
        # Scarica ultimi 5 giorni per sicurezza
        vix_series = get_series("^VIX", period="5d")
        if vix_series is not None and not vix_series.empty:
            return float(vix_series.iloc[-1])
        return None
    except Exception:
        return None


def _get_dynamic_atr_multiplier(vix_value: Optional[float]) -> float:
    """
    Determina il moltiplicatore ATR basato sul livello del VIX.
    - VIX Basso (<15): Mercato calmo -> Stop più stretto (2.5)
    - VIX Normale (15-25): Standard -> Stop medio (3.0)
    - VIX Alto (25-35): Alta volatilità -> Stop largo (3.5)
    - VIX Estremo (>35): Panico -> Stop molto largo (4.0)
    """
    if vix_value is None or pd.isna(vix_value):
        return 3.0  # Default conservativo

    if vix_value < 15:
        return 2.5
    elif vix_value < 25:
        return 3.0
    elif vix_value < 35:
        return 3.5
    else:
        return 4.0


def calculate_full_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce il DataFrame con gli indicatori per la strategia Core (V28/Golden Mean)
    e per il calcolo dei rischi (ATR).
    """
    df = df.copy()
    if 'Close' not in df.columns: return df

    # SMA 130 per trend di lungo periodo
    df['SMA_130'] = df['Close'].rolling(130).mean()

    # Momentum a 6 mesi (circa 126 giorni di trading)
    df['Momentum_6M'] = df['Close'].pct_change(126)

    # ATR 14 periodi per gestione volatilità
    df['ATR'] = calculate_atr_series(df, window=14)

    return df


def analyze_ticker(ticker: str, df: pd.DataFrame) -> Dict:
    """
    Analisi per la strategia Core (V28 / Golden Mean).
    Restituisce segnali ENTRY/EXIT basati su SMA130 e Momentum.
    """
    if df is None or len(df) < 135:
        return {"signal": "Dati Insufficienti", "confidence": 0, "reason": "Storico < 135gg"}

    df = calculate_full_indicators(df)
    last = df.iloc[-1]

    price = last['Close']
    sma130 = last['SMA_130']
    mom_6m = last.get('Momentum_6M', 0)
    atr = last.get('ATR', 0)

    signal = "NEUTRAL"
    ui_signal = "ATTENDI"
    score = 0
    reason = "Dati non calcolabili"

    if pd.notna(sma130) and pd.notna(mom_6m):
        dist_sma = (price / sma130) - 1.0

        if price < sma130:
            signal = "EXIT"
            ui_signal = "VENDI (Trend Break)"
            reason = f"Prezzo sotto SMA 130 ({dist_sma:.1%}). Trend ribassista."
            score = -100

        elif mom_6m > 0:
            signal = "ENTRY"
            ui_signal = "COMPRA (Golden Mean)"
            reason = f"Trend Sano (> SMA130) e Momentum Positivo."
            score = mom_6m * 100

        else:
            signal = "HOLD_WEAK"
            ui_signal = "MONITORA (Mom. Neg)"
            reason = f"Sopra SMA 130 ma Momentum negativo ({mom_6m:.1%}). Nessun trigger."
            score = mom_6m * 100

    return {
        "ticker": ticker,
        "signal": ui_signal,
        "raw_signal": signal,
        "confidence": score,
        "reason": reason,
        "price": price,
        "indicators": {
            "SMA_130": sma130,
            "Momentum_6M": mom_6m,
            "ATR": atr
        }
    }


def check_satellite_status(df: pd.DataFrame, purchase_date: pd.Timestamp, current_shares: float) -> Dict:
    """
    Gestione posizioni SATELLITE nel Portfolio Tracker.

    Strategia: Trailing Stop su ATR Dinamico (VIX-Adjusted).
    1. Calcola l'ATR dello strumento.
    2. Recupera il VIX attuale per decidere il moltiplicatore (da 2.5x a 4.0x).
    3. Fissa lo Stop Loss sottraendo (ATR * Moltiplicatore) dal Massimo (High) raggiunto dall'acquisto.
    """
    if df is None or df.empty:
        return {"action": "HOLD", "reason": "Dati non disponibili"}

    # 1. Calcolo/Verifica ATR
    if 'ATR' not in df.columns:
        df['ATR'] = calculate_atr_series(df, window=14)

    # 2. Normalizzazione Timezone e Date
    df_clean = df.copy()
    if df_clean.index.tz is not None:
        df_clean.index = df_clean.index.tz_localize(None)

    if purchase_date.tzinfo is not None:
        purchase_date = purchase_date.tz_localize(None)

    # 3. Filtraggio periodo di detenzione
    # Consideriamo solo i prezzi dalla data di acquisto a oggi per trovare il "Massimo Locale"
    mask = df_clean.index >= purchase_date
    df_since_buy = df_clean.loc[mask]

    if df_since_buy.empty:
        return {"action": "HOLD", "reason": "Data acquisto futura o dati mancanti"}

    # 4. Recupero VIX e Moltiplicatore Dinamico
    current_vix = _get_latest_vix()
    atr_multiplier = _get_dynamic_atr_multiplier(current_vix)

    vix_str = f"{current_vix:.2f}" if current_vix else "N/D"

    # 5. Calcolo Trailing Stop
    # Highest High since purchase (Massimo intraday raggiunto durante il possesso)
    highest_high = df_since_buy['High'].max()

    current_close = df_clean['Close'].iloc[-1]
    current_atr = df_clean['ATR'].iloc[-1]

    if pd.isna(current_atr) or current_atr <= 0:
        return {"action": "HOLD", "reason": "ATR non disponibile"}

    # Livello di Stop Dinamico
    stop_price = highest_high - (current_atr * atr_multiplier)

    # 6. Generazione Segnale
    if current_close < stop_price:
        return {
            "action": "SELL",
            "reason": f"Stop ATR rotto. Prezzo {current_close:.2f} < {stop_price:.2f} (Max: {highest_high:.2f}, ATR: {current_atr:.2f}x{atr_multiplier}, VIX: {vix_str})",
            "stop_price": stop_price
        }

    return {
        "action": "HOLD",
        "reason": f"In Trend. Stop @ {stop_price:.2f} (VIX: {vix_str} -> Molt: {atr_multiplier}x)",
        "stop_price": stop_price
    }