# -*- coding: utf-8 -*-
import pandas as pd
from typing import Dict
from etf_metrics.core.metrics import calculate_atr_series, calculate_adx_series, calculate_rsi_series


def calculate_full_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Arricchisce il DataFrame con tutti gli indicatori necessari."""
    df = df.copy()
    if 'Close' not in df.columns: return df

    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    df['STD_20'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)

    df['ATR'] = calculate_atr_series(df)
    df['ADX'] = calculate_adx_series(df)
    df['RSI'] = calculate_rsi_series(df['Close'])

    df['Vol_SMA_20'] = df['Volume'].rolling(20).mean()

    return df


def analyze_ticker(ticker: str, df: pd.DataFrame) -> Dict:
    """
    Analisi per 'Single ETF' o 'Trading Scanner'.
    Restituisce segnali: COMPRA (Breakout/Dip), VENDI (Breakdown), MANTIENI.
    """
    if df is None or len(df) < 200:
        return {"signal": "Dati Insufficienti", "confidence": 0, "reason": "Storico < 200gg"}

    df = calculate_full_indicators(df)
    last = df.iloc[-1]

    price = last['Close']
    atr = last.get('ATR', price * 0.01)
    adx = last.get('ADX', 0)
    rsi = last.get('RSI', 50)

    is_uptrend = last['Close'] > last['SMA_200']

    signal = "NEUTRAL"
    score = 0
    reason = "Fase Laterale"

    if price > last['BB_Upper'] and is_uptrend:
        signal = "LONG_BREAKOUT"
        score = adx + 20
        reason = "Rottura Bollinger Band Superiore in Trend Rialzista"

    elif is_uptrend and rsi < 40 and price > last['SMA_200']:
        signal = "LONG_DIP"
        score = (100 - rsi)
        reason = f"Trend Rialzista con RSI scarico ({rsi:.1f})"

    elif price < last['SMA_200']:
        signal = "SHORT_BEAR"
        score = 50
        reason = "Prezzo sotto la Media a 200 periodi (Trend Ribassista)"

    elif is_uptrend:
        signal = "MANTIENI"
        score = 0
        reason = "Trend positivo, nessun segnale di ingresso specifico."

    stop_loss = price - (atr * 3) if "LONG" in signal else price * 0.9

    ui_signal = "MANTIENI"
    if "LONG" in signal: ui_signal = "COMPRA"
    if "SHORT" in signal: ui_signal = "VENDI"

    return {
        "ticker": ticker,
        "signal": ui_signal,
        "raw_signal": signal,
        "confidence": min(100, int(score)),
        "reason": reason,
        "price": price,
        "stop_loss": stop_loss,
        "indicators": {"RSI": rsi, "ADX": adx, "ATR": atr}
    }


def check_satellite_status(df: pd.DataFrame, purchase_date: pd.Timestamp, current_shares: float) -> Dict:
    """
    Logica di VENDITA per il Portfolio Tracker (Strategia Satellite).
    Replica la logica di automated_backtest: Trailing Stop basato su ATR dai massimi.
    """
    if df is None or df.empty:
        return {"action": "HOLD", "reason": "No Data"}

    df_holding = df[df.index >= purchase_date].copy()

    if len(df_holding) < 2:
        return {"action": "HOLD", "reason": "Posizione troppo recente"}

    full_atr = calculate_atr_series(df)
    current_atr = full_atr.iloc[-1]

    highest_price_since_buy = df_holding['High'].max()

    stop_price = highest_price_since_buy - (current_atr * 3.0)
    current_price = df['Close'].iloc[-1]

    if current_price < stop_price:
        return {
            "action": "SELL",
            "reason": f"Trailing Stop raggiunto ({stop_price:.2f}). Max recente: {highest_price_since_buy:.2f}",
            "stop_price": stop_price
        }

    return {
        "action": "HOLD",
        "reason": f"Trend Satellite intatto. Stop attuale: {stop_price:.2f}",
        "stop_price": stop_price
    }