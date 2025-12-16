# -*- coding: utf-8 -*-
import pandas as pd
from typing import Dict
from etf_metrics.core.metrics import calculate_atr_series


def calculate_full_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce il DataFrame con gli indicatori V28 (Golden Mean).
    """
    df = df.copy()
    if 'Close' not in df.columns: return df

    # --- INDICATORI STRATEGIA V28 ---
    # 1. Trend Filter: SMA 130
    df['SMA_130'] = df['Close'].rolling(130).mean()

    # 2. Ranking Factor: Momentum 6 Mesi (126 giorni)
    df['Momentum_6M'] = df['Close'].pct_change(126)

    # Indicatori ausiliari per UI (opzionali per la strategia, utili per contesto)
    df['ATR'] = calculate_atr_series(df)  # Utile per volatilità corrente

    return df


def analyze_ticker(ticker: str, df: pd.DataFrame) -> Dict:
    """
    Analisi allineata alla Strategia V28 (Golden Mean).
    Restituisce segnali basati su SMA130 e Momentum.
    """
    # Servono almeno 130 giorni + buffer
    if df is None or len(df) < 135:
        return {"signal": "Dati Insufficienti", "confidence": 0, "reason": "Storico < 135gg"}

    df = calculate_full_indicators(df)
    last = df.iloc[-1]

    price = last['Close']
    sma130 = last['SMA_130']
    mom_6m = last.get('Momentum_6M', 0)
    atr = last.get('ATR', 0)

    # Default
    signal = "NEUTRAL"
    ui_signal = "ATTENDI"
    score = 0
    reason = "Dati non calcolabili"

    if pd.notna(sma130) and pd.notna(mom_6m):

        # Distanza dalla SMA in %
        dist_sma = (price / sma130) - 1.0

        # LOGICA V28
        if price < sma130:
            # Uscita immediata (o divieto di ingresso)
            signal = "EXIT"
            ui_signal = "VENDI (Trend Break)"
            reason = f"Prezzo sotto SMA 130 ({dist_sma:.1%}). Trend ribassista."
            score = -100  # Priorità bassa / negativa

        elif mom_6m > 0:
            # Condizione di Ingresso Soddisfatta (Trend OK + Momentum OK)
            signal = "ENTRY"
            ui_signal = "COMPRA (Golden Mean)"
            reason = f"Trend Sano (> SMA130) e Momentum Positivo."
            # Lo score è il Momentum stesso (per il ranking)
            score = mom_6m * 100

        else:
            # Prezzo sopra SMA ma Momentum negativo (Fase di stallo/ritracciamento)
            signal = "HOLD_WEAK"
            ui_signal = "MONITORA (Mom. Neg)"
            reason = f"Sopra SMA 130 ma Momentum negativo ({mom_6m:.1%}). Nessun trigger."
            score = mom_6m * 100  # Score basso/negativo

    return {
        "ticker": ticker,
        "signal": ui_signal,
        "raw_signal": signal,
        "confidence": score,  # Usato per ordinare per Momentum
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
    Logica di VENDITA per il Portfolio Tracker.
    Per coerenza con la V28, usiamo la SMA 130 anche qui come exit principale,
    oppure manteniamo il Trailing Stop ATR come 'paracadute' aggiuntivo.
    """
    if df is None or df.empty:
        return {"action": "HOLD", "reason": "No Data"}

    # Calcolo SMA 130 corrente
    sma130 = df['Close'].rolling(130).mean().iloc[-1]
    current_price = df['Close'].iloc[-1]

    # 1. EXIT PRIMARIA: SMA 130
    if pd.notna(sma130) and current_price < sma130:
        return {
            "action": "SELL",
            "reason": f"Trend Break: Prezzo ({current_price:.2f}) sotto SMA 130 ({sma130:.2f})",
            "stop_price": sma130
        }

    return {
        "action": "HOLD",
        "reason": f"Trend Intatto (> SMA 130).",
        "stop_price": sma130 if pd.notna(sma130) else 0
    }