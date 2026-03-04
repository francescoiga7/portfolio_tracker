# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.automated_backtest import (
    calculate_advanced_metrics_vectorized,
    calculate_ai_smart_score,
    _assess_market_regime,
    CONFIG
)

ALGO_FILE = "live_algo_portfolio.json"


# --- GESTIONE STATO ---
def load_algo_state():
    if os.path.exists(ALGO_FILE):
        with open(ALGO_FILE, 'r') as f:
            return json.load(f)
    return {
        "cash": 10000.0,  # Valore default, verrà sovrascritto dalla UI
        "positions": {},
        "trade_log": [],
        "last_update": None
    }


def save_algo_state(state):
    with open(ALGO_FILE, 'w') as f:
        json.dump(state, f, indent=4, default=str)


def get_latest_market_data(tickers):
    data = {}
    all_tickers = list(set(tickers + ['SPY']))
    with st.spinner(f"Scaricamento dati aggiornati per {len(all_tickers)} asset..."):
        for t in all_tickers:
            df = get_series(t, period="2y", as_dataframe=True)
            if df is not None and not df.empty:
                df = calculate_advanced_metrics_vectorized(df)
                data[t] = df
    return data


def run_daily_update(state, tickers_list, allow_fractional):
    """
    Esegue l'analisi giornaliera supportando azioni frazionate.
    """
    market_data = get_latest_market_data(tickers_list)
    if 'SPY' not in market_data:
        st.error("Dati SPY mancanti.")
        return state, []

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
        if ticker not in market_data: continue
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
            except:
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
            # Gestione arrotondamento vendita
            if not allow_fractional:
                qty_sell = int(qty_sell)
                if qty_sell == 0 and sell_portion > 0: qty_sell = pos['qty']  # Vendi tutto se troppo poco

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
        # RIMOSSO LIMITE HARD DI 2000$. Ora basta avere cash > 0
        if free_slots > 0 and state['cash'] > 10:
            candidates = []
            for t in tickers_list:
                if t == 'SPY' or t in state['positions'] or t not in market_data: continue
                row = market_data[t].iloc[-1]
                score, reason = calculate_ai_smart_score(row)
                if score >= 60:
                    candidates.append({'t': t, 'score': score, 'row': row, 'reason': reason})

            candidates.sort(key=lambda x: x['score'], reverse=True)

            for cand in candidates:
                if free_slots <= 0 or state['cash'] < 10: break

                t = cand['t']
                row = cand['row']
                price = row['Close']

                # Sizing Dinamico
                alloc = (state['cash'] / free_slots) * 0.99  # Usa 99% del cash disponibile per slot

                # LOGICA FRAZIONATA VS INTERA
                if allow_fractional:
                    qty = alloc / price
                    # Arrotonda a 4 decimali per pulizia
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


def render_algo_live_ui():
    st.title("💲Portafoglio Live")

    state = load_algo_state()

    # --- SIDEBAR: SETUP CAPITALE E OPZIONI ---
    st.sidebar.header("⚙️ Configurazione")

    # 1. Modifica Capitale Dinamica
    current_cash = state.get('cash', 10000.0)
    new_cash = st.sidebar.number_input(
        "💰 Capitale Liquido ($)",
        min_value=0.0,
        value=float(current_cash),
        step=100.0,
        help="Modifica manualmente la liquidità disponibile."
    )

    # Aggiorna lo stato se l'utente cambia il numero
    if new_cash != current_cash:
        state['cash'] = new_cash
        save_algo_state(state)
        st.rerun()

    # 2. Flag Frazionato
    allow_fractional = st.sidebar.checkbox(
        "Ammetti Azioni Frazionate",
        value=True,
        help="Se attivo, compra es. 0.5 azioni. Se disattivo, solo numeri interi (es. NVDA a 1000$ richiede almeno 1000$)."
    )

    # 3. Tickers
    default_tickers = "NVDA\nAMD\nTSLA\nAAPL\nMSFT\nAMZN\nGOOGL\nMETA\nNFLX\nPLTR\nCOIN"
    with st.sidebar.expander("Lista Ticker", expanded=False):
        tickers_input = st.text_area("Ticker", default_tickers, height=150)
    tickers_list = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

    # Reset Button
    if st.sidebar.button("🗑️ Reset Totale"):
        state = {"cash": 10000.0, "positions": {}, "trade_log": [], "last_update": None}
        save_algo_state(state)
        st.rerun()

    # --- DASHBOARD ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Liquidità (Cash)", f"${state['cash']:.2f}")
    col2.metric("Posizioni Aperte", len(state['positions']))
    col3.metric("Ultimo Aggiornamento", state['last_update'] if state['last_update'] else "Mai")

    st.divider()

    # --- ACTION ---
    if st.button("🚀 ESEGUI ANALISI GIORNALIERA", type="primary", width="stretch"):
        with st.status("Elaborazione Strategia...", expanded=True) as status:
            st.write("📥 Analisi Mercato e Regime...")
            new_state, msgs = run_daily_update(state, tickers_list, allow_fractional)
            save_algo_state(new_state)

            for m in msgs:
                if "VENDITA" in m or "ACQUISTO" in m:
                    st.write(f"**{m}**")
                else:
                    st.write(m)
            status.update(label="Fatto!", state="complete")
            st.session_state['algo_updated'] = True

    if st.session_state.get('algo_updated'):
        st.session_state['algo_updated'] = False
        st.rerun()

    # --- TABELLE ---
    st.subheader("💼 Posizioni")
    if state['positions']:
        pos_data = []
        for t, data in state['positions'].items():
            pos_data.append({
                "Ticker": t,
                "Qty": f"{data['qty']:.4f}" if allow_fractional else f"{data['qty']:.0f}",
                "Entry": f"${data['entry_price']:.2f}",
                "Stop": f"${data['stop_loss']:.2f}",
                "Date": data['entry_date']
            })
        st.dataframe(pd.DataFrame(pos_data), width="stretch")
    else:
        st.info("Portafoglio Flat (100% Cash)")

    st.subheader("📜 Log")
    if state['trade_log']:
        st.dataframe(pd.DataFrame(state['trade_log']), width="stretch")