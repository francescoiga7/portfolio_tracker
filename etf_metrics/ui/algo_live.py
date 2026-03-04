# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.automated_backtest import (
    calculate_advanced_metrics_vectorized,
    calculate_stock_score,
    calculate_crypto_score,
    _assess_market_regime,
    get_crypto_cycle_phase,
    is_crypto_asset,
    CONFIG
)

ALGO_FILE = "live_algo_portfolio.json"


# --- GESTIONE STATO ---
def load_algo_state():
    if os.path.exists(ALGO_FILE):
        with open(ALGO_FILE, 'r') as f:
            return json.load(f)
    return {
        "cash": 10000.0,
        "positions": {},
        "trade_log": [],
        "last_update": None
    }


def save_algo_state(state):
    with open(ALGO_FILE, 'w') as f:
        json.dump(state, f, indent=4, default=str)


def get_latest_market_data(tickers):
    data = {}
    # Aggiungiamo sempre BTC e SPY come benchmark
    all_tickers = list(set(tickers + ['SPY', 'BTC-USD']))

    with st.spinner(f"Scaricamento dati aggiornati per {len(all_tickers)} asset..."):
        # 1. Scaricamento Grezzo
        raw_data = {}
        for t in all_tickers:
            df = get_series(t, period="2y", as_dataframe=True)
            if df is not None and not df.empty:
                raw_data[t] = df

        # 2. Calcolo Metriche Avanzate con Benchmark corretti
        spy_df = raw_data.get('SPY')
        btc_df = raw_data.get('BTC-USD')

        for t, df in raw_data.items():
            # Determina benchmark appropriato per RS_Score
            is_crypto = is_crypto_asset(t)
            bench = btc_df if is_crypto else spy_df

            # Calcola indicatori (RSI, ADX, ATR, RS_Score)
            # Passiamo il benchmark così l'RS_Score è reale e non 50.0 fisso
            processed_df = calculate_advanced_metrics_vectorized(df, bench)
            data[t] = processed_df

    return data


def run_daily_update(state, tickers_list, allow_fractional):
    """
    Esegue l'analisi giornaliera in modalità IBRIDA (Stock + Crypto).
    """
    market_data = get_latest_market_data(tickers_list)

    # Controlli Benchmark
    if 'SPY' not in market_data:
        return state, ["❌ Errore critico: Dati SPY non disponibili."]

    has_btc = 'BTC-USD' in market_data

    today = datetime.now().date()
    today_str = str(today)

    # 1. ANALISI MACRO (Doppio Binario)
    # Usiamo l'ultima data disponibile di SPY come riferimento temporale
    last_ref_date = market_data['SPY'].index[-1]

    # A. Regime Stock
    regime_stock = _assess_market_regime(last_ref_date, market_data, 'SPY', is_crypto_context=False)
    is_bull_stock = (regime_stock == "BULL")

    # B. Regime Crypto
    regime_crypto = "UNKNOWN"
    cycle_phase = "ACCUMULATION"
    is_bull_crypto = False

    if has_btc:
        regime_crypto = _assess_market_regime(last_ref_date, market_data, 'BTC-USD', is_crypto_context=True)
        cycle_phase = get_crypto_cycle_phase(last_ref_date)
        is_bull_crypto = (regime_crypto == "BULL")

    messages = [
        f"📅 Dati aggiornati al: {last_ref_date.date()}",
        f"🏢 Regime Stocks: **{regime_stock}**",
        f"🪙 Regime Crypto: **{regime_crypto}** ({cycle_phase})"
    ]
    new_log_entries = []

    # 2. GESTIONE USCITE (SELL)
    for ticker, pos in list(state['positions'].items()):
        if ticker not in market_data: continue

        row = market_data[ticker].iloc[-1]
        curr_price = row['Close']
        atr = row['ATR']

        is_c = is_crypto_asset(ticker)

        reason_sell = None
        sell_portion = 0.0
        roi = (curr_price / pos['entry_price']) - 1

        # A. STOP LOSS (Universale)
        if curr_price < pos['stop_loss']:
            reason_sell = "STOP LOSS"
            sell_portion = 1.0

        # B. TAKE PROFIT (Ibrido)
        elif is_c and cycle_phase == "BULL_RUN":
            # Moonbag Crypto: Vendi 25% a +100%
            if not pos.get('moonbag_secured', False) and roi >= 1.0:
                reason_sell = "TP 100% (Moonbag)"
                sell_portion = 0.25
                # Nota: l'aggiornamento dello stop loss avviene dopo
        else:
            # Stock o Crypto Bear: TP Fisso
            if not pos.get('tp1_taken', False) and roi >= CONFIG['TP1_PCT']:
                reason_sell = "TP1 (Lock Profit)"
                sell_portion = 0.33

        # C. MACRO PANIC (Ibrido)
        if not reason_sell:
            if is_c:
                # Se è Inverno Crypto e stiamo perdendo, tagliare
                if regime_crypto == "BEAR" and roi < -0.10:
                    reason_sell = "MACRO RISK (Crypto Winter)"
                    sell_portion = 1.0
            else:
                # Se Stock Market crash
                if regime_stock in ["DANGER", "VOLATILE"] and roi < 0.05:
                    reason_sell = "MACRO RISK (Stock)"
                    sell_portion = 1.0

        # D. TIME STOP
        # Disattivato per Crypto in Bull Run
        if not reason_sell:
            try:
                entry_dt = datetime.strptime(str(pos['entry_date']), "%Y-%m-%d").date()
            except:
                entry_dt = datetime.now().date()

            days_held = (today - entry_dt).days

            # Crypto Time Stop (solo se non Bull Run)
            if is_c:
                if cycle_phase != "BULL_RUN" and days_held >= 60 and roi < 0.01:
                    reason_sell = "TIME STOP (Crypto)"
                    sell_portion = 1.0
            # Stock Time Stop (sempre attivo)
            else:
                if days_held >= CONFIG['TIME_STOP_DAYS'] and roi < 0.01:
                    reason_sell = "TIME STOP (Stock)"
                    sell_portion = 1.0

        # E. TRAILING STOP UPDATE
        # Si attiva solo se la posizione non viene chiusa completamente
        if sell_portion < 1.0:
            if pos.get('tp1_taken', False) or pos.get('moonbag_secured', False) or roi > 0.05:
                # Moltiplicatore ATR in base all'asset
                mult = 4.0 if (is_c and cycle_phase == "BULL_RUN") else CONFIG['STOP_LOSS_ATR_MULT']
                new_stop = curr_price - (atr * mult)

                if new_stop > pos['stop_loss']:
                    state['positions'][ticker]['stop_loss'] = new_stop
                    messages.append(f"🛡️ {ticker}: Stop alzato a {new_stop:.2f}")

        # ESECUZIONE VENDITA
        if reason_sell:
            qty_sell = pos['qty'] * sell_portion
            if not allow_fractional:
                qty_sell = int(qty_sell)
                # Se la porzione è piccola ma intera fa 0, forza vendita totale se portion > 0.5
                if qty_sell == 0 and sell_portion > 0.5: qty_sell = pos['qty']

            if qty_sell > 0:
                cash_in = (qty_sell * curr_price) - CONFIG['COMMISSION']
                state['cash'] += cash_in

                if sell_portion >= 0.99 or (pos['qty'] - qty_sell) < (0.001 if allow_fractional else 1):
                    del state['positions'][ticker]
                else:
                    state['positions'][ticker]['qty'] -= qty_sell
                    if "TP" in reason_sell:
                        state['positions'][ticker]['tp1_taken'] = True
                        if "Moonbag" in reason_sell:
                            state['positions'][ticker]['moonbag_secured'] = True
                            # Alza stop a breakeven aggressivo
                            state['positions'][ticker]['stop_loss'] = max(pos['stop_loss'], pos['entry_price'] * 1.10)
                        else:
                            # Alza stop a breakeven
                            state['positions'][ticker]['stop_loss'] = max(pos['stop_loss'], pos['entry_price'] * 1.01)

                log_entry = {
                    "Date": today_str, "Ticker": ticker, "Action": "SELL",
                    "Price": curr_price, "Reason": reason_sell,
                    "PnL_Net": (curr_price - pos['entry_price']) * qty_sell  # Approx PnL
                }
                state['trade_log'].insert(0, log_entry)
                new_log_entries.append(f"🔴 VENDITA: {ticker} ({reason_sell})")

    # 3. GESTIONE INGRESSI (BUY)
    free_slots = CONFIG['MAX_POSITIONS'] - len(state['positions'])

    if free_slots > 0 and state['cash'] > 50:
        candidates = []
        for t in tickers_list:
            # Skip se già in portafoglio, se dati mancanti o se benchmark
            if t in ['SPY', 'BTC-USD'] or t in state['positions'] or t not in market_data: continue

            row = market_data[t].iloc[-1]
            is_c = is_crypto_asset(t)

            # FILTRO REGIME:
            # Compra stock solo se Stock Bull
            # Compra crypto solo se Crypto Bull (o Recovery)
            if is_c:
                if cycle_phase == "BEAR_WINTER": continue  # Mai comprare in inverno profondo
                if not is_bull_crypto: continue
            else:
                if not is_bull_stock: continue

            # SCORING SPECIFICO
            if is_c:
                score, reason = calculate_crypto_score(row, cycle_phase)
                min_req = 55 if cycle_phase == "BULL_RUN" else 75
            else:
                score, reason = calculate_stock_score(row)
                min_req = 60

            if score >= min_req:
                candidates.append({'t': t, 'score': score, 'row': row, 'reason': reason, 'is_crypto': is_c})

        # Ordina per score decrescente
        candidates.sort(key=lambda x: x['score'], reverse=True)

        for cand in candidates:
            if free_slots <= 0 or state['cash'] < 50: break

            t = cand['t']
            row = cand['row']
            price = row['Close']
            is_c = cand['is_crypto']

            # Sizing (Equal Weight sugli slot rimanenti)
            alloc = (state['cash'] / free_slots) * 0.98

            if allow_fractional:
                qty = alloc / price
                qty = round(qty, 6)  # Precisione crypto
            else:
                qty = int(alloc / price)

            cost = (qty * price) + CONFIG['COMMISSION']

            if qty > 0 and cost <= state['cash']:
                state['cash'] -= cost

                # Stop Loss Iniziale (Dinamico per asset)
                mult = 4.0 if (is_c and cycle_phase == "BULL_RUN") else CONFIG['STOP_LOSS_ATR_MULT']
                initial_stop = price - (row['ATR'] * mult)

                state['positions'][t] = {
                    'qty': qty, 'entry_price': price, 'entry_date': today_str,
                    'stop_loss': initial_stop, 'tp1_taken': False, 'moonbag_secured': False
                }

                log_entry = {
                    "Date": today_str, "Ticker": t, "Action": "BUY",
                    "Price": price, "Reason": cand['reason'], "PnL_Net": 0
                }
                state['trade_log'].insert(0, log_entry)
                new_log_entries.append(f"🟢 ACQUISTO: {t} (Score: {cand['score']})")
                free_slots -= 1
    elif free_slots == 0:
        messages.append("ℹ️ Portafoglio pieno.")

    state['last_update'] = today_str
    return state, messages + new_log_entries


def render_algo_live_ui():
    st.title("💲Portafoglio Live (Hybrid)")

    state = load_algo_state()

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Configurazione")

    # Modifica Capitale
    current_cash = state.get('cash', 10000.0)
    new_cash = st.sidebar.number_input(
        "💰 Capitale Liquido ($)",
        min_value=0.0,
        value=float(current_cash),
        step=100.0
    )
    if new_cash != current_cash:
        state['cash'] = new_cash
        save_algo_state(state)
        st.rerun()

    allow_fractional = st.sidebar.checkbox("Ammetti Frazionate", value=True)

    default_tickers = "BTC-USD\nETH-USD\nSOL-USD\nNVDA\nTSLA\nCOIN\nMSTR"
    with st.sidebar.expander("Lista Ticker", expanded=False):
        tickers_input = st.text_area("Ticker", default_tickers, height=150)
    tickers_list = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

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
        with st.status("Elaborazione Strategia Ibrida...", expanded=True) as status:
            st.write("📥 Scaricamento Dati e Analisi Macro...")
            try:
                new_state, msgs = run_daily_update(state, tickers_list, allow_fractional)
                save_algo_state(new_state)

                for m in msgs:
                    if "VENDITA" in m:
                        st.error(m)
                    elif "ACQUISTO" in m:
                        st.success(m)
                    else:
                        st.info(m)

                status.update(label="Analisi Completata!", state="complete")
                st.session_state['algo_updated'] = True
            except Exception as e:
                st.error(f"Errore durante l'esecuzione: {e}")
                status.update(label="Errore", state="error")

    if st.session_state.get('algo_updated'):
        st.session_state['algo_updated'] = False
        st.rerun()

    # --- TABELLE ---
    st.subheader("💼 Posizioni Attuali")
    if state['positions']:
        pos_data = []
        for t, data in state['positions'].items():
            pos_data.append({
                "Ticker": t,
                "Qty": f"{data['qty']:.4f}",
                "Entry": f"${data['entry_price']:.2f}",
                "Stop Loss": f"${data['stop_loss']:.2f}",
                "Date": data['entry_date']
            })
        st.dataframe(pd.DataFrame(pos_data), width="stretch")
    else:
        st.info("Nessuna posizione aperta.")

    st.subheader("📜 Storico Operazioni")
    if state['trade_log']:
        st.dataframe(pd.DataFrame(state['trade_log']), width="stretch")