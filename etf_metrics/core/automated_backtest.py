# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import logging
from datetime import timedelta
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.data_manager import MarketDataManager

logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE "AI ENHANCED STRATEGY" ---
CONFIG = {
    'MAX_POSITIONS': 3,
    'REBALANCE_DAYS': 5,
    'ENABLE_MONTHLY_LIMIT': True,
    'MAX_BUYS_PER_MONTH': 4,
    'COMMISSION': 2.0,
    'SPY_TICKER': 'SPY',
    'MACRO_TREND_FILTER': True,
    'BEAR_VOLATILITY_THRESHOLD': 3.0,
    'MIN_RS_SCORE': 80,
    'MIN_ADX': 25,
    'MIN_PROX_HIGH': 0.85,
    'STOP_LOSS_ATR_MULT': 2.5, #aumentare a 3 aumenta volatilità e rendimento
    'TIME_STOP_DAYS': 21,
    'TP1_PCT': 0.12, #0.08, #migliore 0.12
    'MAX_CORRELATION': 0.65
}


def calculate_advanced_metrics_vectorized(df, spy_df=None):
    if 'Close' not in df.columns: return df
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df['Vol_20'] = df['Close'].pct_change().rolling(20).std() * 100

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr14 = ranges.max(axis=1).rolling(14).sum()
    plus_di14 = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di14 = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    df['ADX'] = dx.rolling(14).mean()

    df['High_52w'] = df['Close'].rolling(252).max()
    df['Prox_High'] = df['Close'] / df['High_52w']

    df['RS_Score'] = 0.0
    if spy_df is not None:
        common = df.index.intersection(spy_df.index)
        if len(common) > 65:
            stock_ret = df['Close'].pct_change(63)
            spy_ret = spy_df['Close'].pct_change(63)
            rel_perf = (stock_ret.loc[common] - spy_ret.loc[common]) * 100
            df.loc[common, 'RS_Score'] = rel_perf

    return df


def _assess_market_regime(price_matrix, current_date, market_data):
    spy = CONFIG['SPY_TICKER']
    if spy not in market_data or current_date not in market_data[spy].index:
        start = current_date - timedelta(days=200)
        proxy = price_matrix.loc[start:current_date].mean(axis=1)
        if len(proxy) < 200: return "NEUTRAL"
        return "BULL" if proxy.iloc[-1] > proxy.mean() else "BEAR"

    row = market_data[spy].loc[current_date]
    if row['Close'] < row['SMA200']: return "BEAR"
    atr_pct = (row['ATR'] / row['Close']) * 100
    if atr_pct > CONFIG['BEAR_VOLATILITY_THRESHOLD']: return "VOLATILE"
    if row['RSI'] < 35: return "DANGER"
    return "BULL"


def check_correlation_strict(candidate, portfolio, price_matrix, current_date):
    if not portfolio: return True, None
    if candidate not in price_matrix.columns: return True, None
    start = current_date - timedelta(days=60)
    hist = price_matrix.loc[start:current_date]
    if len(hist) < 30: return True, None
    cand_series = hist[candidate]
    for p in portfolio:
        if p not in hist.columns: continue
        corr = cand_series.corr(hist[p])
        if corr > CONFIG['MAX_CORRELATION']: return False, p
    return True, None


def calculate_ai_smart_score(row):
    if row['Close'] < row['SMA50']: return 0, "Below SMA50"
    if row['Prox_High'] < CONFIG['MIN_PROX_HIGH']: return 0, "Too far from Highs"
    if row['ADX'] < CONFIG['MIN_ADX']: return 0, "Weak Trend"
    score = 50
    score += (row['RS_Score'] * 2)
    score += (row['ADX'] / 2)
    if row['Prox_High'] > 0.95: score += 10
    if row['Vol_20'] > 5.0: score -= 15
    if row['RSI'] > 85: score -= 25
    return max(0, score), f"AI Score: {score:.0f} | ProxHigh: {row['Prox_High']:.2f}"


def prepare_market_data(tickers, period="5y"):
    db_manager = MarketDataManager()
    all_tickers = list(set(tickers + [CONFIG['SPY_TICKER']]))
    missing = db_manager.get_tickers_needing_update(all_tickers)
    if missing:
        new_data = {t: get_series(t, period=period, as_dataframe=True) for t in missing}
        new_data = {k: v for k, v in new_data.items() if v is not None and not v.empty}
        if new_data: db_manager.save_bulk_data(new_data)

    loaded = db_manager.load_data(all_tickers)
    spy_df = loaded.get(CONFIG['SPY_TICKER'])
    if spy_df is not None and not spy_df.empty:
        spy_df = calculate_advanced_metrics_vectorized(spy_df, None)
        loaded[CONFIG['SPY_TICKER']] = spy_df

    processed = {}
    for t, df in loaded.items():
        if not df.empty and len(df) > 200:
            processed[t] = calculate_advanced_metrics_vectorized(df, spy_df)
    return processed


def run_market_aware_backtest(tickers: list, start_date="2015-01-01", initial_capital=10000,
                              preloaded_data=None, tax_rate=26.0, allow_fractional=True):
    if preloaded_data:
        market_data = preloaded_data
    else:
        market_data = prepare_market_data(tickers)

    if not market_data: return pd.DataFrame(), initial_capital

    ref = CONFIG['SPY_TICKER'] if CONFIG['SPY_TICKER'] in market_data else list(market_data.keys())[0]
    sim_dates = market_data[ref].index[market_data[ref].index >= pd.to_datetime(start_date)]
    price_matrix = pd.DataFrame({t: d['Close'] for t, d in market_data.items()}).ffill()

    cash = float(initial_capital)
    positions = {}
    trade_log = []
    tax_credit = 0.0
    current_sim_month = -1
    buys_this_month = 0

    try:
        progress = st.progress(0)
    except:
        progress = None

    # --- HELPER PER CALCOLO NAV (FIX DRAWDOWN) ---
    def _get_current_nav(curr_cash, curr_positions, curr_date):
        equity = curr_cash
        for p_ticker, p_data in curr_positions.items():
            # Cerca prezzo corrente, fallback su entry price
            try:
                curr_p = market_data[p_ticker].loc[curr_date]['Close']
            except:
                curr_p = p_data['entry_price']
            equity += p_data['qty'] * curr_p
        return equity

    # ---------------------------------------------

    for i, current_date in enumerate(sim_dates):
        if progress and i % 50 == 0: progress.progress((i + 1) / len(sim_dates))

        if current_date.month != current_sim_month:
            current_sim_month = current_date.month
            buys_this_month = 0

        regime = _assess_market_regime(price_matrix, current_date, market_data)
        is_bull = (regime == "BULL")
        is_crash = (regime == "DANGER" or regime == "VOLATILE")

        tokens_to_sell = []

        # 1. GESTIONE USCITE
        for t, pos in positions.items():
            if t not in market_data or current_date not in market_data[t].index: continue
            row = market_data[t].loc[current_date]
            curr_price = row['Close']
            low = row['Low']
            atr = row['ATR']

            if low < pos['stop_loss']:
                exit_price = max(row['Open'], pos['stop_loss'])
                tokens_to_sell.append((t, exit_price, "STOP LOSS", 1.0))
                continue

            roi = (curr_price / pos['entry_price']) - 1
            if not pos.get('tp1_taken', False) and roi >= CONFIG['TP1_PCT']:
                tokens_to_sell.append((t, curr_price, "TP1 (Lock)", 0.25))
                #tokens_to_sell.append((t, curr_price, "TP1 (Lock)", 0.33))
                positions[t]['stop_loss'] = pos['entry_price'] * 1.01
                positions[t]['tp1_taken'] = True
                continue

            if pos.get('tp1_taken', False) or roi > 0.03:
                new_stop = curr_price - (atr * CONFIG['STOP_LOSS_ATR_MULT'])
                if new_stop > positions[t]['stop_loss']:
                    positions[t]['stop_loss'] = new_stop

            if is_crash and roi < 0.05:
                tokens_to_sell.append((t, curr_price, "MACRO RISK", 1.0))
                continue

            days_held = (current_date - pos['entry_date']).days
            if days_held >= CONFIG['TIME_STOP_DAYS'] and roi < 0.01:
                tokens_to_sell.append((t, curr_price, "TIME STOP", 1.0))

        # ESECUZIONE VENDITE
        for t, price, reason, portion in tokens_to_sell:
            pos = positions[t]
            qty_sell = pos['qty'] * portion
            if not allow_fractional:
                qty_sell = int(qty_sell)
                if qty_sell == 0 and portion > 0.9: qty_sell = pos['qty']

            if qty_sell <= 0: continue

            net = (qty_sell * price) - CONFIG['COMMISSION']
            cost_portion = pos['cost_basis'] * (qty_sell / pos['qty'])
            gain = net - cost_portion

            tax = 0.0
            if gain > 0:
                taxable = max(0, gain - tax_credit)
                tax = taxable * (tax_rate / 100.0)
                tax_credit = max(0, tax_credit - gain)
            else:
                tax_credit += abs(gain)

            cash += (net - tax)

            # Aggiorna posizione
            if portion >= 0.99 or (pos['qty'] - qty_sell) < (0.001 if allow_fractional else 1):
                del positions[t]
            else:
                positions[t]['qty'] -= qty_sell
                positions[t]['cost_basis'] -= cost_portion

            # LOGGING: Usiamo Total_Equity invece di Cash
            current_nav = _get_current_nav(cash, positions, current_date)
            trade_log.append({
                "Date": current_date.date(), "Ticker": t, "Action": "SELL",
                "Price": price, "Reason": reason,
                "PnL_Net": gain - tax,
                "Capital": cash,  # Liquidità residua
                "Total_Equity": current_nav,  # Valore Reale Portafoglio
                "Qty": qty_sell
            })

        # 2. GESTIONE INGRESSI
        buy_allowed = is_bull and (not CONFIG['ENABLE_MONTHLY_LIMIT'] or buys_this_month < CONFIG['MAX_BUYS_PER_MONTH'])

        if buy_allowed and i % CONFIG['REBALANCE_DAYS'] == 0:
            free_slots = CONFIG['MAX_POSITIONS'] - len(positions)

            if free_slots > 0 and cash > 50:
                candidates = []
                for t in tickers:
                    if t == CONFIG['SPY_TICKER'] or t in positions: continue
                    if t not in market_data or current_date not in market_data[t].index: continue
                    row = market_data[t].loc[current_date]
                    score, reason = calculate_ai_smart_score(row)
                    if score >= 60:
                        candidates.append({'t': t, 'score': score, 'row': row, 'reason': reason})

                candidates.sort(key=lambda x: x['score'], reverse=True)

                for cand in candidates:
                    if free_slots <= 0 or cash < 50: break
                    if CONFIG['ENABLE_MONTHLY_LIMIT'] and buys_this_month >= CONFIG['MAX_BUYS_PER_MONTH']: break

                    t = cand['t']
                    row = cand['row']

                    is_safe_corr, conflict = check_correlation_strict(t, list(positions.keys()), price_matrix,
                                                                      current_date)
                    if not is_safe_corr: continue

                    risk_factor = 1.0
                    alloc_per_slot = (cash / free_slots) * risk_factor * 0.98

                    if allow_fractional:
                        qty = alloc_per_slot / row['Close']
                    else:
                        qty = int(alloc_per_slot / row['Close'])

                    cost = (qty * row['Close']) + CONFIG['COMMISSION']

                    if qty > 0 and cost <= cash:
                        cash -= cost
                        initial_stop = row['Close'] - (row['ATR'] * CONFIG['STOP_LOSS_ATR_MULT'])

                        positions[t] = {
                            'qty': qty, 'cost_basis': cost, 'entry_price': row['Close'],
                            'entry_date': current_date, 'stop_loss': initial_stop, 'tp1_taken': False
                        }

                        # LOGGING: Total_Equity
                        current_nav = _get_current_nav(cash, positions, current_date)
                        trade_log.append({
                            "Date": current_date.date(), "Ticker": t, "Action": "BUY",
                            "Price": row['Close'], "Reason": cand['reason'],
                            "PnL_Net": 0.0,
                            "Capital": cash,
                            "Total_Equity": current_nav,
                            "Qty": qty
                        })
                        free_slots -= 1
                        buys_this_month += 1

    if progress: progress.empty()

    # Chiusura Finale (Mark to Market)
    final_nav = cash
    for t, pos in positions.items():
        try:
            p = market_data[t].iloc[-1]['Close']
        except:
            p = pos['entry_price']
        val = pos['qty'] * p
        final_nav += val

        # Log Finale
        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": t, "Action": "HOLD (End)",
            "Price": p, "Reason": "Portfolio Value",
            "PnL_Net": val - pos['cost_basis'],
            "Capital": cash,
            "Total_Equity": final_nav,
            "Qty": pos['qty']
        })

    return pd.DataFrame(trade_log), final_nav