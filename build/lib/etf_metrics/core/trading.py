# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, Optional
from etf_metrics.core.metrics import calculate_atr_series, calculate_rsi_series, calculate_adx_series
from etf_metrics.clients.yahoo_client import get_series


def _get_latest_vix() -> Optional[float]:
    try:
        vix_series = get_series("^VIX", period="5d")
        return float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else None
    except Exception:
        return None


def _get_dynamic_atr_multiplier(vix_value: Optional[float]) -> float:
    """
    Restituisce un moltiplicatore ATR più ampio per gestire la volatilità
    degli ETF settoriali (Satellite), evitando uscite premature.
    """
    if vix_value is None or pd.isna(vix_value):
        return 3.5  # Default aumentato (era 3.0)

    # Scala aumentata di 0.5 - 1.0 punti rispetto alla versione precedente
    if vix_value < 15: return 3.0  # Bassa volatilità: stop stretto ma non soffocante (era 2.5)
    if vix_value < 25: return 3.5  # Volatilità normale: spazio di manovra (era 3.0)
    if vix_value < 35: return 4.5  # Volatilità alta: serve aria (era 3.5)
    return 6.0  # Panico (VIX > 35): stop molto largo per evitare whipsaw (era 4.0)


def calculate_full_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Close' not in df.columns: return df

    # Calcoli robusti con gestione errori
    try:
        df['SMA_130'] = df['Close'].rolling(130).mean()
        df['Momentum_6M'] = df['Close'].pct_change(126)
        df['ATR'] = calculate_atr_series(df, window=14)
        df['RSI'] = calculate_rsi_series(df['Close'], window=14)
        df['ADX'] = calculate_adx_series(df, window=14)
    except Exception:
        pass  # Se fallisce qualche indicatore, proseguiamo

    return df


def analyze_ticker(ticker: str, df: pd.DataFrame) -> Dict:
    # 1. Check Dati Minimi
    if df is None or len(df) < 135:
        return {
            "ticker": ticker,
            "signal": "Dati Insufficienti",
            "raw_signal": "NEUTRAL",
            "confidence": 0,
            "reason": "Storico < 135gg",
            "stop_loss": 0,
            "price": 0,
            "indicators": {}
        }

    # 2. Calcolo Indicatori
    df = calculate_full_indicators(df)
    last = df.iloc[-1]

    # 3. Estrazione Valori (FIX BUG: Definizione variabili rsi/adx)
    price = last['Close']
    sma130 = last.get('SMA_130', np.nan)
    mom_6m = last.get('Momentum_6M', 0)
    atr = last.get('ATR', 0)
    rsi = last.get('RSI', 50)  # Default neutro se manca
    adx = last.get('ADX', 0)

    # 4. Calcolo Stop Loss Dinamico
    vix = _get_latest_vix()
    mult = _get_dynamic_atr_multiplier(vix)
    stop_loss = price - (atr * mult) if atr > 0 else 0

    # 5. Logica Segnale V28 (Golden Mean)
    ui_signal, signal, score, reason = "ATTENDI", "NEUTRAL", 0, "Analisi in corso"

    if pd.notna(sma130) and pd.notna(mom_6m):
        if price < sma130:
            signal, ui_signal, score = "EXIT", "VENDI (Trend Break)", -100
            reason = f"Prezzo sotto SMA 130. Trend ribassista."
        elif mom_6m > 0:
            signal, ui_signal, score = "ENTRY", "COMPRA (Golden Mean)", int(min(mom_6m * 100, 100))
            reason = f"Trend sopra SMA130 e Momentum positivo."
        else:
            signal, ui_signal, score = "HOLD_WEAK", "MONITORA (Mom. Neg)", int(mom_6m * 100)
            reason = f"Sopra SMA 130 ma Momentum negativo."

    # Aggiustamento score in base a RSI (Overbought/Oversold)
    if pd.notna(rsi):
        if rsi > 70: reason += " (Attenzione: RSI Ipercomprato)"
        if rsi < 30: reason += " (Attenzione: RSI Ipervenduto)"

    return {
        "ticker": ticker,
        "signal": ui_signal,
        "raw_signal": signal,
        "confidence": score,
        "reason": reason,
        "price": price,
        "stop_loss": stop_loss,
        "indicators": {
            "SMA_130": sma130,
            "Momentum_6M": mom_6m,
            "ATR": atr,
            "RSI": rsi,  # Ora la variabile è definita
            "ADX": adx  # Ora la variabile è definita
        }
    }


def check_satellite_status(df: pd.DataFrame, purchase_date: pd.Timestamp, current_shares: float) -> Dict:
    """
    Verifica lo stato di una posizione satellite usando un Trailing Stop basato su ATR e Close.
    """
    if df is None or df.empty: return {"action": "HOLD", "reason": "Dati non disponibili"}

    df_clean = df.copy()
    if df_clean.index.tz is not None: df_clean.index = df_clean.index.tz_localize(None)

    p_date = purchase_date.tz_localize(None) if purchase_date.tzinfo else purchase_date
    df_since = df_clean[df_clean.index >= p_date]

    if df_since.empty: return {"action": "HOLD", "reason": "Data acquisto futura"}

    # Calcolo ATR corrente
    atr_series = calculate_atr_series(df_clean)
    if atr_series is None or atr_series.empty:
        return {"action": "HOLD", "reason": "ATR non calcolabile"}

    atr_val = atr_series.iloc[-1]

    # 1. Recupero Moltiplicatore Dinamico (più largo per i satelliti)
    mult = _get_dynamic_atr_multiplier(_get_latest_vix())

    # 2. MODIFICA CRITICA: Uso il 'Close' massimo invece del 'High' massimo.
    highest_close_since = df_since['Close'].max()

    # Calcolo Stop Price (Chandelier Exit su Close)
    stop_price = highest_close_since - (atr_val * mult)

    current_price = df_clean['Close'].iloc[-1]

    # Calcolo distanza percentuale dallo stop per info
    dist_pct = ((current_price - stop_price) / current_price) * 100

    if current_price < stop_price:
        return {
            "action": "SELL",
            "stop_price": stop_price,
            "reason": f"Prezzo {current_price:.2f} < Stop Dinamico {stop_price:.2f} (MaxClose: {highest_close_since:.2f})"
        }

    return {
        "action": "HOLD",
        "stop_price": stop_price,
        "reason": f"Trend OK. Stop @ {stop_price:.2f} ({dist_pct:.1f}% di margine)"
    }