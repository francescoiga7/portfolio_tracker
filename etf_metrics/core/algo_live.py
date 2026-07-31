# -*- coding: utf-8 -*-
"""Logica del portafoglio algoritmico live: stato, update giornaliero, metriche.

Nessuna dipendenza da Streamlit: la UI si occupa solo di rendering e progress.
"""
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.automated_backtest import (
    calculate_advanced_metrics_vectorized,
    calculate_ai_smart_score,
    _assess_market_regime,
    CONFIG,
)
from etf_metrics.shared.config import ALGO_STATE_FILE

logger = logging.getLogger(__name__)


def load_algo_state(filename: str = ALGO_STATE_FILE) -> Dict:
    """Carica lo stato persistito del portafoglio algoritmico."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Impossibile leggere %s: %s", filename, e)
    return {"cash": 10000.0, "positions": {}, "trade_log": [], "last_update": None}


def save_algo_state(state: Dict, filename: str = ALGO_STATE_FILE) -> None:
    """Salva lo stato del portafoglio algoritmico su file."""
    try:
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4, default=str)
    except Exception as e:
        logger.warning("Errore salvataggio %s: %s", filename, e)


def get_latest_market_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Scarica dati 2y per i ticker richiesti (più SPY) e calcola gli indicatori."""
    data = {}
    all_tickers = list(set(tickers + ['SPY']))
    for t in all_tickers:
        df = get_series(t, period="2y", as_dataframe=True)
        if df is not None and not df.empty:
            df = calculate_advanced_metrics_vectorized(df)
            data[t] = df
    return data


def run_daily_update(state: Dict, tickers_list: List[str], allow_fractional: bool) -> Tuple[Dict, List[str]]:
    """
    Esegue l'analisi giornaliera supportando azioni frazionate.
    Ritorna (nuovo_stato, messaggi).
    """
    market_data = get_latest_market_data(tickers_list)
    if 'SPY' not in market_data:
        return state, ["⚠️ Dati SPY mancanti."]

    today = datetime.now().date()
    today_str = str(today)

    # 1. ANALISI MACRO
    spy_df = market_data['SPY']
    price_matrix = pd.DataFrame({t: d['Close'] for t, d in market_data.items()})
    last_date = spy_df.index[-1]

    regime = _assess_market_regime(price_matrix, last_date, market_data)
    is_bull = (regime == "BULL")
    is_crash = (regime in ["DANGER", "VOLATILE"])

    messages = [f"📅 Data Dati: {last_date.date()}", f"🌍 Regime: **{regime}**"]
    new_log_entries = []

    # 2. GESTIONE USCITE (SELL)
    for ticker, pos in list(state['positions'].items()):
        if ticker not in market_data:
            continue
        row = market_data[ticker].iloc[-1]
        curr_price = row['Close']
        atr = row['ATR']

        reason_sell = None
        sell_portion = 0.0
        roi = (curr_price / pos['entry_price']) - 1

        if curr_price < pos['stop_loss']:
            reason_sell = "STOP LOSS"
            sell_portion = 1.0
        elif not pos.get('tp1_taken', False) and roi >= CONFIG['TP1_PCT']:
            reason_sell = "TP1 (Lock Profit)"
            sell_portion = 0.33
        elif is_crash and roi < 0.05:
            reason_sell = "MACRO RISK"
            sell_portion = 1.0
        else:
            # Time Stop Check
            try:
                entry_dt = datetime.strptime(pos['entry_date'], "%Y-%m-%d").date()
            except Exception:
                entry_dt = datetime.strptime(pos['entry_date'], "%Y-%m-%d %H:%M:%S").date()
            if (today - entry_dt).days >= CONFIG['TIME_STOP_DAYS'] and roi < 0.01:
                reason_sell = "TIME STOP"
                sell_portion = 1.0

        # Trailing Stop Update
        if sell_portion < 1.0:
            if pos.get('tp1_taken', False) or roi > 0.03:
                new_stop = curr_price - (atr * CONFIG['STOP_LOSS_ATR_MULT'])
                if new_stop > pos['stop_loss']:
                    state['positions'][ticker]['stop_loss'] = new_stop
                    messages.append(f"🛡️ {ticker}: Stop alzato a {new_stop:.2f}")

        # Execute Sell
        if reason_sell:
            qty_sell = pos['qty'] * sell_portion
            if not allow_fractional:
                qty_sell = int(qty_sell)
                if qty_sell == 0 and sell_portion > 0:
                    qty_sell = pos['qty']

            if qty_sell > 0:
                cash_in = (qty_sell * curr_price) - CONFIG['COMMISSION']
                state['cash'] += cash_in

                if sell_portion >= 0.99 or (pos['qty'] - qty_sell) < (0.01 if allow_fractional else 1):
                    del state['positions'][ticker]
                else:
                    state['positions'][ticker]['qty'] -= qty_sell
                    state['positions'][ticker]['tp1_taken'] = True
                    state['positions'][ticker]['stop_loss'] = max(pos['stop_loss'], pos['entry_price'] * 1.01)

                log_entry = {
                    "Date": today_str, "Ticker": ticker, "Action": "SELL",
                    "Price": curr_price, "Reason": reason_sell,
                    "PnL_Net": (curr_price - pos['entry_price']) * qty_sell
                }
                state['trade_log'].insert(0, log_entry)
                new_log_entries.append(f"🔴 VENDITA: {ticker} ({reason_sell})")

    # 3. GESTIONE INGRESSI (BUY)
    if is_bull:
        free_slots = CONFIG['MAX_POSITIONS'] - len(state['positions'])
        if free_slots > 0 and state['cash'] > 10:
            candidates = []
            for t in tickers_list:
                if t == 'SPY' or t in state['positions'] or t not in market_data:
                    continue
                row = market_data[t].iloc[-1]
                score, reason = calculate_ai_smart_score(row)
                if score >= 60:
                    candidates.append({'t': t, 'score': score, 'row': row, 'reason': reason})

            candidates.sort(key=lambda x: x['score'], reverse=True)

            for cand in candidates:
                if free_slots <= 0 or state['cash'] < 10:
                    break

                t = cand['t']
                row = cand['row']
                price = row['Close']

                # Sizing Dinamico
                alloc = (state['cash'] / free_slots) * 0.99

                if allow_fractional:
                    qty = alloc / price
                    qty = round(qty, 4)
                else:
                    qty = int(alloc / price)

                cost = (qty * price) + CONFIG['COMMISSION']

                if qty > 0 and cost <= state['cash']:
                    state['cash'] -= cost
                    initial_stop = price - (row['ATR'] * CONFIG['STOP_LOSS_ATR_MULT'])

                    state['positions'][t] = {
                        'qty': qty, 'entry_price': price, 'entry_date': today_str,
                        'stop_loss': initial_stop, 'tp1_taken': False
                    }

                    log_entry = {
                        "Date": today_str, "Ticker": t, "Action": "BUY",
                        "Price": price, "Reason": cand['reason'], "PnL_Net": 0
                    }
                    state['trade_log'].insert(0, log_entry)
                    new_log_entries.append(f"🟢 ACQUISTO: {t} (Q: {qty})")
                    free_slots -= 1
        elif free_slots == 0:
            messages.append("ℹ️ Portafoglio pieno.")
    else:
        messages.append("⛔ Mercato non Bull: Acquisti bloccati.")

    state['last_update'] = today_str
    return state, messages + new_log_entries
