# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from etf_metrics.clients.yahoo_client import get_series

# --- CONFIGURAZIONE V28 (THE GOLDEN MEAN) ---
MAX_POSITIONS = 4
REBALANCE_DAYS = 20


def calculate_indicators(df):
    """
    Calcola indicatori per V28.
    SMA130: Il compromesso perfetto tra trend lungo (200) e breve (50).
    Momentum: 6 mesi.
    """
    df['SMA130'] = df['Close'].rolling(130).mean()
    df['Momentum'] = df['Close'].pct_change(126)

    return df


def prepare_market_data(tickers, period="10y"):
    market_data = {}
    for t in tickers:
        df = get_series(t, period=period, as_dataframe=True)
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            market_data[t] = df
    return market_data


def run_market_aware_backtest(tickers: list, start_date="2015-01-01", initial_capital=1000,
                              preloaded_data=None, commission=2.0, tax_rate=26.0):

    if preloaded_data:
        market_data = preloaded_data
    else:
        market_data = prepare_market_data(tickers)

    if not market_data: return pd.DataFrame(), initial_capital

    sample_ticker = list(market_data.keys())[0]
    sim_dates = market_data[sample_ticker].index[market_data[sample_ticker].index >= pd.to_datetime(start_date)]

    cash = initial_capital
    positions = {}
    trade_log = []
    tax_credit = 0.0

    try:
        progress_bar = st.progress(0)
    except:
        progress_bar = None

    days_counter = 0

    for i, current_date in enumerate(sim_dates):
        if progress_bar and i % 20 == 0:
            progress_bar.progress((i + 1) / len(sim_dates))

        days_counter += 1

        portfolio_equity = cash
        tickers_to_sell = []

        for t, pos in positions.items():
            if current_date in market_data[t].index:
                daily = market_data[t].loc[current_date]
                curr_close = daily['Close']
                sma130 = daily['SMA130']

                portfolio_equity += (pos['qty'] * curr_close)

                if pd.notna(sma130) and curr_close < sma130:
                    tickers_to_sell.append((t, curr_close, "📉 TREND BREAK (< SMA130)"))

            else:
                portfolio_equity += (pos['qty'] * pos.get('entry_price', 0))

        for t, price, reason in tickers_to_sell:
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
            portfolio_equity -= (pos['qty'] * price)

            trade_log.append({
                "Date": current_date.date(), "Ticker": t, "Action": "SELL",
                "Price": price, "Qty": pos['qty'], "Reason": reason,
                "Comm": commission, "Tax": tax,
                "PnL_Net": (net - tax) - pos['cost_basis'],
                "PnL_Pct": 0, "Capital": cash
            })
            del positions[t]

        if days_counter >= REBALANCE_DAYS:
            days_counter = 0

            candidates = []
            for t, df in market_data.items():
                if current_date not in df.index: continue
                daily = df.loc[current_date]

                if pd.isna(daily['SMA130']) or pd.isna(daily['Momentum']): continue

                if daily['Close'] < daily['SMA130']: continue

                if daily['Momentum'] <= 0: continue

                candidates.append({
                    'ticker': t, 'score': daily['Momentum'], 'close': daily['Close']
                })

            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_picks = candidates[:MAX_POSITIONS]
            top_tickers = [c['ticker'] for c in top_picks]

            tickers_out = []
            for t in list(positions.keys()):
                if t not in top_tickers:
                    if current_date in market_data[t].index:
                        curr_price = market_data[t].loc[current_date]['Close']
                        tickers_out.append((t, curr_price, "🔄 ROTATION EXIT"))

            for t, price, reason in tickers_out:
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
                portfolio_equity -= (pos['qty'] * price)

                trade_log.append({
                    "Date": current_date.date(), "Ticker": t, "Action": "SELL",
                    "Price": price, "Qty": pos['qty'], "Reason": reason,
                    "Comm": commission, "Tax": tax,
                    "PnL_Net": (net - tax) - pos['cost_basis'],
                    "PnL_Pct": 0, "Capital": cash
                })
                del positions[t]

            free_slots = MAX_POSITIONS - len(positions)

            if free_slots > 0 and cash > 50:
                budget_per_slot = cash / free_slots

                for cand in top_picks:
                    if cand['ticker'] in positions: continue
                    if free_slots <= 0: break

                    if budget_per_slot < 50: continue

                    qty = round(budget_per_slot / cand['close'], 4)
                    cost = (qty * cand['close']) + commission

                    if cost <= cash:
                        cash -= cost
                        free_slots -= 1

                        positions[cand['ticker']] = {
                            'qty': qty,
                            'entry_price': cand['close'],
                            'cost_basis': cost
                        }

                        portfolio_equity += (qty * cand['close'])

                        trade_log.append({
                            "Date": current_date.date(), "Ticker": cand['ticker'], "Action": "BUY",
                            "Price": cand['close'], "Qty": qty,
                            "Reason": f"GOLD ENTRY (Mom: {cand['score']:.2%})",
                            "Comm": commission, "Tax": 0.0, "PnL_Net": 0, "PnL_Pct": 0,
                            "Capital": cash
                        })

    if progress_bar: progress_bar.empty()

    for t, pos in positions.items():
        if t not in market_data: continue
        final_price = market_data[t]['Close'].iloc[-1]
        gross = pos['qty'] * final_price
        net = gross - commission
        gain = net - pos['cost_basis']
        tax = gain * (tax_rate / 100) if gain > 0 else 0
        final_cash = net - tax
        cash += final_cash

        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": t, "Action": "END",
            "Price": final_price, "Qty": pos['qty'], "Reason": "Chiusura Simulazione",
            "Comm": commission, "Tax": tax,
            "PnL_Net": final_cash - pos['cost_basis'], "PnL_Pct": 0,
            "Capital": cash
        })

    return pd.DataFrame(trade_log), cash