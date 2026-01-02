# -*- coding: utf-8 -*-
import pandas as pd
from typing import Dict
from etf_metrics.core.metrics import calculate_atr_series
from etf_metrics.clients.yahoo_client import get_series

vix_series = get_series("^VIX", period="5d")
current_vix = vix_series.iloc[-1] if vix_series is not None else None

def calculate_full_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce il DataFrame con gli indicatori V28 (Golden Mean).
    """
    df = df.copy()
    if 'Close' not in df.columns: return df

    df['SMA_130'] = df['Close'].rolling(130).mean()
    df['Momentum_6M'] = df['Close'].pct_change(126)
    df['ATR'] = calculate_atr_series(df, window=14)  # Assicuriamoci che l'ATR sia calcolato

    return df


def analyze_ticker(ticker: str, df: pd.DataFrame) -> Dict:
    """
    Analisi allineata alla Strategia V28 (Golden Mean).
    Restituisce segnali basati su SMA130 e Momentum.
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
    Logica di VENDITA per il Portfolio Tracker (Satellite).
    Implementa un Trailing Stop basato su ATR(14).
    Stop Price = Highest Close (dall'acquisto) - 3 * ATR.
    """
    if df is None or df.empty:
        return {"action": "HOLD", "reason": "No Data"}

    # 1. Calcolo Indicatori necessari (ATR)
    # Se l'ATR non è già nel dataframe, lo calcoliamo
    if 'ATR' not in df.columns:
        df['ATR'] = calculate_atr_series(df, window=14)

    # 2. Preparazione dati e gestione Timezone
    df_clean = df.copy()
    if df_clean.index.tz is not None:
        df_clean.index = df_clean.index.tz_localize(None)

    if purchase_date.tzinfo is not None:
        purchase_date = purchase_date.tz_localize(None)

    # 3. Filtraggio periodo di detenzione (dall'acquisto a oggi)
    mask = df_clean.index >= purchase_date
    df_since_buy = df_clean.loc[mask]

    if df_since_buy.empty:
        # Potrebbe accadere se la data di acquisto è errata o futura
        return {"action": "HOLD", "reason": "Data acq. non valida/futura"}

    # 4. Calcolo Trailing Stop
    # Troviamo il prezzo di chiusura più alto raggiunto DURANTE il possesso
    highest_close_since_buy = df_since_buy['Close'].max()

    # Valori attuali
    current_atr = df_clean['ATR'].iloc[-1]
    current_close = df_clean['Close'].iloc[-1]

    base_multiplier = 3.0
    if current_vix:
        vix_factor = current_vix / 20.0
        vix_factor = max(0.5, min(vix_factor, 1.6))

        final_multiplier = base_multiplier * vix_factor
    else:
        final_multiplier = base_multiplier

    stop_price = highest_close_since_buy - (current_atr * final_multiplier)
    if pd.isna(current_atr) or current_atr <= 0:
        return {"action": "HOLD", "reason": "ATR non disponibile"}

    # 5. Generazione Segnale
    if current_close < stop_price:
        return {
            "action": "SELL",
            "reason": f"Trailing Stop ATR rotto. Prezzo ({current_close:.2f}) < Stop ({stop_price:.2f}) [Max: {highest_close_since_buy:.2f}, ATR: {current_atr:.2f}]",
            "stop_price": stop_price
        }

    return {
        "action": "HOLD",
        "reason": f"In Trend. Stop ATR dinamico a {stop_price:.2f} (Max dall'acq: {highest_close_since_buy:.2f})",
        "stop_price": stop_price
    }