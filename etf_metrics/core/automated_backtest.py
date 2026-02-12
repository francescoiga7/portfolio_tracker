# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.data_manager import MarketDataManager

# --- CONFIGURAZIONE V28 ---
MAX_POSITIONS = 4
REBALANCE_DAYS = 20


def calculate_indicators(df):
    if 'Close' in df.columns:
        df['SMA130'] = df['Close'].rolling(130).mean()
        df['Momentum'] = df['Close'].pct_change(126)
    return df


def prepare_market_data(tickers, period="10y"):
    db_manager = MarketDataManager()

    missing = db_manager.get_tickers_needing_update(tickers)
    if missing:
        new_data = {}
        for t in missing:
            df = get_series(t, period=period, as_dataframe=True)
            if df is not None and not df.empty:
                new_data[t] = df
        if new_data:
            db_manager.save_bulk_data(new_data)

    loaded = db_manager.load_data(tickers)

    processed = {}
    for t, df in loaded.items():
        if not df.empty:
            processed[t] = calculate_indicators(df)

    return processed


def run_market_aware_backtest(tickers: list, start_date="2015-01-01", initial_capital=1000,
                              preloaded_data=None, commission=2.0, tax_rate=26.0):
    if preloaded_data:
        market_data = preloaded_data
    else:
        market_data = prepare_market_data(tickers)

    if not market_data: return pd.DataFrame(), initial_capital

    sample_ticker = list(market_data.keys())[0]
    full_idx = market_data[sample_ticker].index
    sim_dates = full_idx[full_idx >= pd.to_datetime(start_date)]

    if len(sim_dates) == 0: return pd.DataFrame(), initial_capital

    combined_closes = pd.DataFrame({t: d['Close'] for t, d in market_data.items()}).reindex(sim_dates)
    combined_smas = pd.DataFrame({t: d['SMA130'] for t, d in market_data.items()}).reindex(sim_dates)
    combined_moms = pd.DataFrame({t: d['Momentum'] for t, d in market_data.items()}).reindex(sim_dates)

    sell_sig_arr = (combined_closes < combined_smas).fillna(False).values

    dates_arr = sim_dates
    closes_arr = combined_closes.values
    moms_arr = combined_moms.values
    tickers_list = combined_closes.columns.tolist()
    t_map = {t: i for i, t in enumerate(tickers_list)}

    cash = initial_capital
    positions = {}
    trade_log = []
    tax_credit = 0.0
    days_counter = 0

    try:
        progress = st.progress(0)
    except:
        progress = None

    num_days = len(dates_arr)

    for i in range(num_days):
        if progress and i % 50 == 0: progress.progress((i + 1) / num_days)

        current_date = dates_arr[i]
        days_counter += 1

        for t in list(positions.keys()):
            idx = t_map.get(t)
            if idx is None: continue

            price = closes_arr[i, idx]
            if np.isnan(price): continue

            if sell_sig_arr[i, idx]:
                pos = positions[t]
                gross = pos['qty'] * price
                net = gross - commission
                gain = net - pos['cost_basis']

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
                    "Price": price, "Reason": "Trend Break",
                    "PnL_Net": (net - tax) - pos['cost_basis'],
                    "PnL_Pct": 0,
                    "Capital": cash
                })
                del positions[t]

        if days_counter >= REBALANCE_DAYS:
            days_counter = 0

            row_moms = moms_arr[i, :]
            row_sells = sell_sig_arr[i, :]
            valid_mask = (row_moms > 0) & (~row_sells) & (~np.isnan(row_moms)) & (~np.isnan(closes_arr[i, :]))

            cands = []
            for idx in np.where(valid_mask)[0]:
                cands.append({'t': tickers_list[idx], 'sc': row_moms[idx], 'p': closes_arr[i, idx]})

            cands.sort(key=lambda x: x['sc'], reverse=True)
            top_picks = cands[:MAX_POSITIONS]
            top_tkrs = set(c['t'] for c in top_picks)

            for t in list(positions.keys()):
                if t not in top_tkrs:
                    idx = t_map[t]
                    price = closes_arr[i, idx]
                    if np.isnan(price): continue

                    pos = positions[t]
                    gross = pos['qty'] * price
                    net = gross - commission
                    gain = net - pos['cost_basis']

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
                        "Price": price, "Reason": "Rotation",
                        "PnL_Net": (net - tax) - pos['cost_basis'],
                        "PnL_Pct": 0,
                        "Capital": cash
                    })
                    del positions[t]

            free = MAX_POSITIONS - len(positions)
            if free > 0 and cash > 50:
                budget = cash / free
                for c in top_picks:
                    if c['t'] in positions: continue
                    if free <= 0: break
                    if budget < 50: continue

                    qty = budget / c['p']
                    cost = (qty * c['p']) + commission
                    if cost <= cash:
                        cash -= cost
                        free -= 1
                        positions[c['t']] = {'qty': qty, 'cost_basis': cost, 'entry_price': c['p']}
                        trade_log.append({
                            "Date": current_date.date(), "Ticker": c['t'], "Action": "BUY",
                            "Price": c['p'], "Reason": f"Mom: {c['sc']:.2%}",
                            "PnL_Net": 0.0,
                            "PnL_Pct": 0.0,
                            "Capital": cash
                        })

    if progress: progress.empty()

    final_idx = len(sim_dates) - 1
    for t, pos in positions.items():
        idx = t_map.get(t)
        p = closes_arr[final_idx, idx] if idx is not None else pos['entry_price']
        if np.isnan(p): p = pos['entry_price']

        gross = pos['qty'] * p
        net = gross - commission
        gain = net - pos['cost_basis']
        tax = gain * (tax_rate / 100) if gain > 0 else 0
        final_cash = net - tax
        cash += final_cash

        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": t, "Action": "END",
            "Price": p,
            "PnL_Net": final_cash - pos['cost_basis'],
            "PnL_Pct": 0.0,
            "Capital": cash
        })

    return pd.DataFrame(trade_log), cash