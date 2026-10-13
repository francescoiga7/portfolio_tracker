# -*- coding: utf-8 -*-
"""Logica di trading live: stop Chandelier per posizioni satellite.

`atr_multiplier_from_vol` e' la scala UNICA condivisa tra logica live
(check_satellite_status) e backtest (strategia GoldenChandelier):
quello che si valida in backtest e' esattamente quello che il portafoglio
reale esegue.
"""
import pandas as pd
from typing import Dict, Optional
from etf_metrics.core.metrics import calculate_atr_series
from etf_metrics.clients.yahoo_client import get_series


def _get_latest_vix() -> Optional[float]:
    """Ultimo valore del VIX da Yahoo (None se offline)."""
    try:
        vix_series = get_series("^VIX", period="5d")
        return float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else None
    except Exception:
        return None


def atr_multiplier_from_vol(vol_pct: Optional[float]) -> float:
    """Moltiplicatore ATR del trailing stop in funzione della volatilita'
    annualizzata del mercato (VIX in punti, o proxy = std annua dei rendimenti
    del benchmark in %).

    Un satellite ha bisogno di stop piu' larghi quando il mercato e' nervoso,
    per evitare uscite premature su noise (whipsaw):
      vol < 15 -> 3.0 | < 25 -> 3.5 | < 35 -> 4.5 | panico -> 6.0 | None -> 3.5
    """
    if vol_pct is None or pd.isna(vol_pct):
        return 3.5  # Default senza dato di volatilità
    if vol_pct < 15: return 3.0  # Bassa volatilità: stop stretto ma non soffocante
    if vol_pct < 25: return 3.5  # Volatilità normale: spazio di manovra
    if vol_pct < 35: return 4.5  # Volatilità alta: serve aria
    return 6.0  # Panico (VIX > 35): stop molto largo per evitare whipsaw


def check_satellite_status(df: pd.DataFrame, purchase_date: pd.Timestamp, current_shares: float,
                           market_vol_pct: Optional[float] = None) -> Dict:
    """Stato di una posizione satellite con Trailing Stop Chandelier su Close.

    market_vol_pct: volatilita' annualizzata del mercato in % (proxy del VIX).
    Se None prova a scaricare il VIX da Yahoo. Passando la proxy si replica
    ESATTAMENTE la strategia GoldenChandelier del backtest (stessa scala).
    Ritorna {'action': 'HOLD'|'SELL', 'reason', 'stop_price'}.
    """
    if df is None or df.empty: return {"action": "HOLD", "reason": "Dati non disponibili"}

    df_clean = df.copy()
    if df_clean.index.tz is not None: df_clean.index = df_clean.index.tz_localize(None)

    p_date = purchase_date.tz_localize(None) if purchase_date.tzinfo else purchase_date
    df_since = df_clean[df_clean.index >= p_date]

    if df_since.empty: return {"action": "HOLD", "reason": "Data acquisto futura"}

    atr_series = calculate_atr_series(df_clean)
    if atr_series is None or atr_series.empty:
        return {"action": "HOLD", "reason": "ATR non calcolabile"}

    atr_val = atr_series.iloc[-1]

    # Moltiplicatore dinamico: proxy della volatilita' se fornita, altrimenti VIX da Yahoo
    vol = market_vol_pct if market_vol_pct is not None else _get_latest_vix()
    mult = atr_multiplier_from_vol(vol)

    # Chandelier Exit sul massimo dei CLOSE dall'acquisto (non High:
    # fedele alla logica live e alla strategia GoldenChandelier)
    highest_close_since = df_since['Close'].max()

    stop_price = highest_close_since - (atr_val * mult)

    current_price = df_clean['Close'].iloc[-1]

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