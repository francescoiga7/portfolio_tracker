# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_atr_series(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window).mean()


def calculate_adx_series(df, window=14):
    """Calcola ADX per misurare la forza del trend."""
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = calculate_atr_series(df, window=1)
    atr = tr.rolling(window).mean().replace(0, np.nan)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window).mean().abs() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window).mean()
    return adx


def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Medie Mobili
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['BB_Std'] * 2)
    # Bollinger Band Width (Compressione)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volume e Volatilità
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['ATR'] = calculate_atr_series(df)
    df['ADX'] = calculate_adx_series(df)

    return df


def analyze_ticker(ticker: str, df: pd.DataFrame) -> Optional[Dict]:
    """
    Genera segnale di trading con 'Confidence Score' (Ranking 0-100).
    """
    if df is None or len(df) < 200: return None

    df = calculate_advanced_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]  # Usato per confermare incroci

    # Dati base
    price = last['Close']
    atr = last['ATR'] if pd.notna(last['ATR']) else price * 0.02
    adx = last.get('ADX', 0)
    vol_rel = (last['Volume'] / last['Vol_SMA_20']) if last['Vol_SMA_20'] > 0 else 1.0
    rsi = last['RSI']

    # Trend Filters
    is_uptrend_strong = (price > last['SMA_50']) and (last['SMA_50'] > last['SMA_200'])
    is_uptrend_weak = (price > last['SMA_200'])

    signal_type = "NEUTRAL"
    score = 0.0  # Score continuo per il Ranking
    reasons = []

    # --- LOGICA SEGNALI E SCORING ---

    # 1. LONG BREAKOUT (Esplosione Volatilità)
    # Prezzo rompe la banda superiore + Volumi + Trend
    if (price > last['BB_Upper']) and is_uptrend_weak:
        # Filtro base: deve esserci un minimo di spinta
        if vol_rel > 1.2:
            signal_type = "LONG_BREAKOUT"

            # === ALGORITMO RANKING (BREAKOUT) ===
            # Base Score: ADX (Forza del trend in atto)
            score = adx

            # Bonus Volume: Più è alto, più il breakout è genuino.
            # Esempio: Vol 3x aggiunge 30 punti. Vol 1.5x aggiunge 5 punti.
            # Formula: (Vol_Rel - 1) * 15
            vol_bonus = (vol_rel - 1.0) * 15
            score += vol_bonus

            # Bonus Compressione: Se veniamo da una fase di compressione (BB Width basso), il movimento è più potente
            # Non lo calcoliamo qui per brevità, ma potremmo.

            # Malus Estensione: Se siamo già troppo estesi (es. RSI > 75), riduciamo lo score (rischio pullback)
            if rsi > 75:
                score -= 10
                reasons.append("⚠️ RSI esteso")

            reasons.append(f"Volumi: {vol_rel:.1f}x media")
            reasons.append(f"Forza Trend (ADX): {adx:.0f}")

    # 2. LONG DIP (Ritracciamento)
    # Trend rialzista forte ma RSI basso (Ipervenduto temporaneo)
    elif is_uptrend_strong and (rsi < 45):
        signal_type = "LONG_DIP"

        # === ALGORITMO RANKING (DIP) ===
        # Base Score: Quanto è 'scontato' il prezzo? (Più basso RSI, meglio è)
        # Esempio: RSI 30 -> Score 70. RSI 40 -> Score 60.
        score = 100 - rsi

        # Bonus Trend: Se il trend di fondo è fortissimo (ADX alto), il dip è un regalo
        score += (adx / 2)

        # Bonus Supporto: Se siamo vicini alla SMA50, aumenta lo score
        dist_sma50_pct = ((price - last['SMA_50']) / last['SMA_50']) * 100
        if -2 < dist_sma50_pct < 2:  # Siamo sul supporto dinamico
            score += 15
            reasons.append("Supporto SMA50 Testato")

        reasons.append(f"RSI Scarico: {rsi:.1f}")

    # 3. SHORT BREAKDOWN
    elif (price < last['BB_Lower']) and (price < last['SMA_200']):
        signal_type = "SHORT_BREAKDOWN"
        score = adx + ((vol_rel - 1.0) * 15)  # Simmetrico al Long
        reasons.append("Rottura Supporti")

    # --- GESTIONE STOP/TARGET ---
    stop_loss = price
    take_profit = price

    if "LONG" in signal_type:
        if "BREAKOUT" in signal_type:
            stop_loss = price - (atr * 1.5)  # Stop stretto
            take_profit = price + (atr * 3.0)
        else:  # DIP
            stop_loss = price - (atr * 3.0)  # Stop largo (spazio per volatilità)
            take_profit = price + (atr * 4.0)
    elif "SHORT" in signal_type:
        stop_loss = price + (atr * 2.0)
        take_profit = price - (atr * 3.0)

    # Normalizzazione Score (0 - 100)
    # Alcuni score potrebbero superare 100 con volumi folli, lo capiamo a 99
    if score > 99: score = 99
    if score < 0: score = 0

    # Se segnale NEUTRAL, score è 0
    if signal_type == "NEUTRAL":
        score = 0

    return {
        "ticker": ticker,
        "signal": signal_type,
        "confidence": round(score, 1),  # Ora è un valore float (es. 85.5) perfetto per il ranking
        "price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasons": reasons,
        "indicators": {
            "RSI": rsi,
            "ATR": atr,
            "BB_Width": last['BB_Width'],
            "Vol_Rel": vol_rel,
            "ADX": adx
        }
    }