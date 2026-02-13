# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import logging
import yfinance as yf
from datetime import timedelta
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.data_manager import MarketDataManager

logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE "SNIPER PATIENCE" V13.0 (Monthly Cap) ---
CONFIG = {
    'MAX_POSITIONS': 3,  # Full Capital (33% a titolo)
    'REBALANCE_DAYS': 5,  # Controllo SETTIMANALE

    # 1. MONTHLY BUY LIMIT (Richiesta Utente)
    'ENABLE_MONTHLY_LIMIT': True,
    'MAX_BUYS_PER_MONTH': 3,  # MAX 3 acquisti al mese. Evita overtrading.

    # 2. MACRO FILTER
    'SPY_TICKER': 'SPY',
    'MACRO_TREND_FILTER': True,  # Compra solo se SPY > SMA200

    # 3. FILTRI DI INGRESSO (Relative Strength)
    'MIN_RS_SCORE': 0,
    'MIN_ADX': 20,
    'RSI_MIN': 50,
    'RSI_MAX': 80,

    # 4. GESTIONE RISCHIO (Swing Trading)
    'STOP_LOSS_ATR_MULT': 3.0,  # Stop AMPIO (3 ATR).
    'TIME_STOP_DAYS': 15,

    # 5. PROFIT TAKING (Let Winners Run)
    'TP1_PCT': 0.05,  # A +5% vendi 33%

    # 6. DIVERSIFICAZIONE
    'MAX_CORRELATION': 0.60
}


def calculate_advanced_indicators(df, spy_df=None):
    if 'Close' not in df.columns: return df

    # Trend
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Volatilità (ATR 14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()

    # Momentum
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ADX
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr14 = ranges.max(axis=1).rolling(14).sum()
    plus_di14 = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di14 = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    df['ADX'] = dx.rolling(14).mean()

    # RS Score vs SPY
    df['RS_Score'] = 0.0
    if spy_df is not None:
        common = df.index.intersection(spy_df.index)
        if len(common) > 65:
            stock_ret = df['Close'].pct_change(63)
            spy_ret = spy_df['Close'].pct_change(63)
            df.loc[common, 'RS_Score'] = (stock_ret.loc[common] - spy_ret.loc[common]) * 100

    return df


def _calculate_backtest_regime(price_matrix, current_date, market_data):
    spy = CONFIG['SPY_TICKER']
    if spy not in market_data:
        start = current_date - timedelta(days=200)
        proxy = price_matrix.loc[start:current_date].mean(axis=1)
        if len(proxy) < 200: return "NEUTRAL"
        return "BULL" if proxy.iloc[-1] > proxy.mean() else "BEAR"

    df_spy = market_data[spy]
    if current_date not in df_spy.index: return "NEUTRAL"
    row = df_spy.loc[current_date]

    if row['Close'] < row['SMA200']: return "BEAR"
    return "BULL"


def check_correlation_strict(candidate, portfolio, price_matrix, current_date):
    if not portfolio: return True, None
    if candidate not in price_matrix.columns: return True, None
    start = current_date - timedelta(days=90)
    hist = price_matrix.loc[start:current_date]
    if len(hist) < 20: return True, None

    cand_series = hist[candidate]
    for p in portfolio:
        if p not in hist.columns: continue
        corr = cand_series.corr(hist[p])
        if corr > CONFIG['MAX_CORRELATION']:
            return False, p
    return True, None


def calculate_smart_score(row):
    if row['Close'] < row['SMA200']: return 0, "Downtrend"
    if row['RS_Score'] < CONFIG['MIN_RS_SCORE']: return 0, "Weak"
    if row['ADX'] < CONFIG['MIN_ADX']: return 0, "No Trend"

    score = 50 + (row['RS_Score'] * 2) + (row['ADX'] / 2)
    if row['RSI'] > 85: score -= 20

    return max(0, score), f"Score: {score:.0f} | RS: {row['RS_Score']:.1f}%"


def prepare_market_data(tickers, period="10y"):
    db_manager = MarketDataManager()
    all_tickers = list(set(tickers + [CONFIG['SPY_TICKER']]))
    missing = db_manager.get_tickers_needing_update(all_tickers)
    if missing:
        new_data = {t: get_series(t, period=period, as_dataframe=True) for t in missing}
        new_data = {k: v for k, v in new_data.items() if v is not None and not v.empty}
        if new_data: db_manager.save_bulk_data(new_data)

    loaded = db_manager.load_data(all_tickers)
    spy_df = loaded.get(CONFIG['SPY_TICKER'])

    processed = {}
    for t, df in loaded.items():
        if not df.empty and len(df) > 250:
            processed[t] = calculate_advanced_indicators(df, spy_df)
    return processed


def run_market_aware_backtest(tickers: list, start_date="2015-01-01", initial_capital=10000,
                              preloaded_data=None, commission=2.0, tax_rate=26.0):
    if preloaded_data:
        market_data = preloaded_data
    else:
        market_data = prepare_market_data(tickers)
    if not market_data: return pd.DataFrame(), initial_capital

    ref = CONFIG['SPY_TICKER'] if CONFIG['SPY_TICKER'] in market_data else list(market_data.keys())[0]
    sim_dates = market_data[ref].index[market_data[ref].index >= pd.to_datetime(start_date)]
    price_matrix = pd.DataFrame({t: d['Close'] for t, d in market_data.items()}).fillna(method='ffill')

    cash = initial_capital
    positions = {}
    trade_log = []
    tax_credit = 0.0

    # Gestione Limite Mensile
    current_sim_month = -1
    buys_this_month = 0

    try:
        progress = st.progress(0)
    except:
        progress = None

    for i, current_date in enumerate(sim_dates):
        if progress and i % 50 == 0: progress.progress((i + 1) / len(sim_dates))

        # Reset contatore mensile
        if current_date.month != current_sim_month:
            current_sim_month = current_date.month
            buys_this_month = 0

        regime = _calculate_backtest_regime(price_matrix, current_date, market_data)
        is_bull = (regime == "BULL")

        tokens_to_sell = []

        # --- GESTIONE POSIZIONI ---
        for t, pos in positions.items():
            if t not in market_data or current_date not in market_data[t].index: continue

            row = market_data[t].loc[current_date]
            curr_price = row['Close']

            # 1. STOP LOSS (3 ATR)
            if row['Low'] < pos['stop_loss']:
                exit_price = row['Open'] if row['Open'] < pos['stop_loss'] else pos['stop_loss']
                tokens_to_sell.append((t, exit_price, "STOP LOSS", 1.0))
                continue

            # 2. TAKE PROFIT PARZIALE (1/3 a +5%)
            roi = (curr_price / pos['entry_price']) - 1
            if not pos.get('tp1_taken', False) and roi >= CONFIG['TP1_PCT']:
                tokens_to_sell.append((t, curr_price, "TP1 (+5%)", 0.33))
                positions[t]['stop_loss'] = pos['entry_price'] * 1.01  # Breakeven
                positions[t]['tp1_taken'] = True
                continue

            # 3. TIME STOP (15 giorni)
            days_held = (current_date - pos['entry_date']).days
            if days_held >= CONFIG['TIME_STOP_DAYS'] and roi < 0.0:
                tokens_to_sell.append((t, curr_price, "TIME STOP", 1.0))
                continue

            # 4. TRAILING STOP
            # Se TP1 è preso, seguiamo con trailing stop
            if pos.get('tp1_taken', False):
                # Usiamo EMA20 o 3 ATR come trailing
                # Semplifichiamo usando il prezzo - 3 ATR
                new_stop = curr_price - (row['ATR'] * 3.0)
                if new_stop > positions[t]['stop_loss']:
                    positions[t]['stop_loss'] = new_stop

            # 5. MACRO EXIT
            if not is_bull and roi < 0.05:
                tokens_to_sell.append((t, curr_price, "MACRO BEAR", 1.0))

        # ESECUZIONE VENDITE
        for t, price, reason, portion in tokens_to_sell:
            pos = positions[t]
            qty_sell = pos['qty'] * portion
            net = (qty_sell * price) - commission
            cost_portion = pos['cost_basis'] * portion
            gain = net - cost_portion

            tax = 0.0
            if gain > 0:
                taxable = max(0, gain - tax_credit)
                tax = taxable * (tax_rate / 100.0)
                tax_credit = max(0, tax_credit - gain)
            else:
                tax_credit += abs(gain)

            cash += (net - tax)
            trade_log.append({
                "Date": current_date.date(), "Ticker": t, "Action": "SELL",
                "Price": price, "Reason": reason,
                "PnL_Net": gain - tax, "Capital": cash
            })
            if portion >= 0.99:
                del positions[t]
            else:
                positions[t]['qty'] -= qty_sell
                positions[t]['cost_basis'] -= cost_portion

        # --- NUOVI INGRESSI ---
        # Check Monthly Limit
        can_buy = is_bull
        if CONFIG['ENABLE_MONTHLY_LIMIT'] and buys_this_month >= CONFIG['MAX_BUYS_PER_MONTH']:
            can_buy = False

        if can_buy and i % CONFIG['REBALANCE_DAYS'] == 0:
            free_slots = CONFIG['MAX_POSITIONS'] - len(positions)

            if free_slots > 0 and cash > 1000:
                candidates = []
                for t in tickers:
                    if t == CONFIG['SPY_TICKER'] or t in positions: continue
                    if t not in market_data or current_date not in market_data[t].index: continue

                    row = market_data[t].loc[current_date]
                    score, reason = calculate_smart_score(row)

                    if score >= 50:
                        candidates.append({'t': t, 'score': score, 'row': row, 'reason': reason})

                candidates.sort(key=lambda x: x['score'], reverse=True)

                for cand in candidates:
                    # Doppio check limite nel loop (se compriamo più di 1 asset oggi)
                    if CONFIG['ENABLE_MONTHLY_LIMIT'] and buys_this_month >= CONFIG['MAX_BUYS_PER_MONTH']: break
                    if free_slots <= 0 or cash < 500: break

                    t = cand['t']
                    row = cand['row']

                    is_safe_corr, conflict = check_correlation_strict(t, list(positions.keys()), price_matrix,
                                                                      current_date)
                    if not is_safe_corr: continue

                    # FULL ALLOCATION (Aggressiva)
                    alloc_per_slot = cash / free_slots
                    invest_amt = alloc_per_slot * 0.99

                    qty = invest_amt / row['Close']
                    cost = (qty * row['Close']) + commission

                    if cost <= cash:
                        cash -= cost
                        # Stop Loss Iniziale (3 ATR)
                        initial_stop = row['Close'] - (row['ATR'] * CONFIG['STOP_LOSS_ATR_MULT'])

                        positions[t] = {
                            'qty': qty, 'cost_basis': cost,
                            'entry_price': row['Close'],
                            'entry_date': current_date,
                            'stop_loss': initial_stop,
                            'tp1_taken': False
                        }

                        trade_log.append({
                            "Date": current_date.date(), "Ticker": t, "Action": "BUY",
                            "Price": row['Close'], "Reason": f"{cand['reason']} | M:{buys_this_month + 1}",
                            "PnL_Net": 0.0, "Capital": cash
                        })
                        free_slots -= 1
                        buys_this_month += 1

    if progress: progress.empty()

    for t, pos in positions.items():
        try:
            p = market_data[t].iloc[-1]['Close']
        except:
            p = pos['entry_price']
        net = (pos['qty'] * p) - commission
        gain = net - pos['cost_basis']
        tax = gain * (tax_rate / 100) if gain > 0 else 0
        cash += (net - tax)
        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": t, "Action": "END",
            "Price": p, "Reason": "Backtest End",
            "PnL_Net": gain - tax, "Capital": cash
        })

    return pd.DataFrame(trade_log), cash