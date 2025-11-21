# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola indicatori avanzati per lo swing trading."""
    df = df.copy()

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['BB_Std'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()

    return df


def analyze_ticker(ticker: str, df: pd.DataFrame) -> Optional[Dict]:
    """Analizza un ticker e genera un segnale di trading."""
    if df is None or len(df) < 50:
        return None

    df = calculate_advanced_indicators(df)
    last = df.iloc[-1]

    signal_type = "NEUTRAL"
    confidence = 0
    reasons = []

    is_uptrend = last['Close'] > last['SMA_50']
    atr = last['ATR'] if pd.notna(last['ATR']) else (last['Close'] * 0.02)
    price = last['Close']

    vol_sma = last.get('Vol_SMA_20', 0)
    vol_rel = (last['Volume'] / vol_sma) if vol_sma > 0 else 0

    if last['Close'] > last['BB_Upper'] and is_uptrend:
        signal_type = "LONG_BREAKOUT"
        confidence += 2
        reasons.append("Rottura Banda Superiore Bollinger")
        if last['MACD'] > last['MACD_Signal']:
            confidence += 1
            reasons.append("MACD Positivo")
        if vol_rel > 1.5:
            confidence += 1
            reasons.append(f"Volume Esplosivo ({vol_rel:.1f}x media)")

    elif is_uptrend and last['RSI'] < 40:
        signal_type = "LONG_DIP"
        confidence += 2
        reasons.append("Ritracciamento in Trend Rialzista")

    elif last['Close'] < last['BB_Lower'] and not is_uptrend:
        signal_type = "SHORT_BREAKDOWN"
        confidence += 2
        reasons.append("Rottura Banda Inferiore Bollinger")

    if "LONG" in signal_type:
        stop_loss = price - (atr * 3.0)
        take_profit = price + (atr * 4.0)
    elif "SHORT" in signal_type:
        stop_loss = price + (atr * 3.0)
        take_profit = price - (atr * 4.0)
    else:
        stop_loss = last['BB_Lower']
        take_profit = last['BB_Upper']

    if confidence < 2:
        signal_type = "NEUTRAL"

    return {
        "ticker": ticker,
        "signal": signal_type,
        "confidence": confidence,
        "price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasons": reasons,
        "indicators": {
            "RSI": last['RSI'],
            "ATR": atr,
            "BB_Width": last['BB_Width'],
            "Vol_Rel": vol_rel
        }
    }