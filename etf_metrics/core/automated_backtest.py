# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from etf_metrics.clients.yahoo_client import get_series


def calculate_atr_series(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window).mean()


def calculate_adx_series(df, window=14):
    """Calcola l'indicatore ADX per misurare la forza del trend."""
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = calculate_atr_series(df, window=1)
    # Evitiamo divisioni per zero
    atr = tr.rolling(window).mean().replace(0, np.nan)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window).mean().abs() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window).mean()
    return adx


def calculate_position_size(capital, current_portfolio_value, vix_value, base_alloc_pct=0.20):
    """Money Management VIX-Adjusted."""
    if pd.isna(vix_value) or vix_value <= 0: vix_value = 20.0
    # Se il capitale è piccolo (es. < 2000), forziamo size più grandi per evitare ordini da 5 euro
    if current_portfolio_value < 2000:
        adjusted_pct = 1.0  # Max 3 posizioni
    else:
        adjusted_pct = min(0.35, base_alloc_pct * (20.0 / vix_value))

    target_amount = current_portfolio_value * adjusted_pct
    return min(capital, target_amount)


def prepare_market_data(tickers, period="5y"):
    """Scarica i dati e calcola gli indicatori."""
    market_data = {}
    for t in tickers:
        df = get_series(t, period=period, as_dataframe=True)
        if df is not None and not df.empty:
            df['ATR'] = calculate_atr_series(df)
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['STD20'] = df['Close'].rolling(20).std()
            df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
            df['ADX'] = calculate_adx_series(df)

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            market_data[t] = df
    return market_data


def run_market_aware_backtest(tickers: list, start_date="2021-01-01", initial_capital=10000, use_tp_only=False,
                              preloaded_data=None):
    """
    Versione 5.1 (Confidence Filter):
    - Filtra segnali con Confidence Score < 75.
    - Allinea il calcolo dello score con trading.py.
    """

    # 1. SELEZIONE FONTE DATI
    if preloaded_data:
        market_data = preloaded_data
    else:
        market_data = prepare_market_data(tickers)

    if not market_data: return pd.DataFrame(), initial_capital

    # Gestione dati ausiliari
    try:
        vix_df = get_series("^VIX", period="5y", as_dataframe=True)
        vix_series = vix_df['Close'] if vix_df is not None else pd.Series(dtype=float)

        sp500_df = get_series("^GSPC", period="5y", as_dataframe=True)
        if sp500_df is not None:
            sp500_df['SMA200'] = sp500_df['Close'].rolling(200).mean()
    except:
        vix_series = pd.Series(dtype=float)
        sp500_df = None

    sample_ticker = list(market_data.keys())[0]
    sim_dates = market_data[sample_ticker].index[market_data[sample_ticker].index >= pd.to_datetime(start_date)]

    cash = initial_capital
    positions = {}
    trade_log = []

    try:
        progress_bar = st.progress(0)
    except:
        progress_bar = None

    # 2. LOOP GIORNALIERO
    for i, current_date in enumerate(sim_dates):
        if progress_bar and i % 5 == 0:
            progress_bar.progress((i + 1) / len(sim_dates))

        try:
            curr_vix = vix_series.loc[:current_date].iloc[-1]
        except:
            curr_vix = 20.0

        market_is_bullish = True
        if sp500_df is not None and current_date in sp500_df.index:
            sp_today = sp500_df.loc[current_date]
            if sp_today['Close'] < sp_today['SMA200']:
                market_is_bullish = False

        # Calcolo valore portafoglio
        portfolio_value = cash
        for t, p in positions.items():
            if current_date in market_data[t].index:
                portfolio_value += p['qty'] * market_data[t].loc[current_date]['Close']
            else:
                portfolio_value += p['qty'] * p['entry_price']

        # --- FASE A: GESTIONE POSIZIONI APERTE (VENDITE) ---
        tickers_with_data = [t for t in positions.keys() if current_date in market_data[t].index]

        for ticker in tickers_with_data:
            pos = positions[ticker]
            df = market_data[ticker]
            daily = df.loc[current_date]
            curr_close = daily['Close']
            curr_high = daily['High']
            curr_atr = daily['ATR']

            if pd.isna(curr_atr): continue

            should_sell = False
            sell_price = curr_close
            sell_reason = ""

            if use_tp_only:
                if curr_high >= pos['take_profit']:
                    should_sell = True
                    sell_price = pos['take_profit']
                    sell_reason = f"🎯 TARGET PROFIT ({pos['take_profit']:.2f})"
                elif curr_close < (pos['entry_price'] * 0.85):  # Hard stop
                    should_sell = True
                    sell_price = curr_close
                    sell_reason = "🛑 HARD STOP (Safety)"
            else:
                if curr_high > pos['highest_price']:
                    positions[ticker]['highest_price'] = curr_high

                mult = pos.get('sl_mult', 3.0)
                potential_stop = positions[ticker]['highest_price'] - (curr_atr * mult)

                if potential_stop > pos['trailing_stop']:
                    positions[ticker]['trailing_stop'] = potential_stop

                if curr_close < pos['trailing_stop']:
                    should_sell = True
                    sell_price = curr_close
                    sell_reason = f"📉 TRAILING STOP (Mult {mult})"

            if should_sell:
                qty = pos['qty']
                revenue = qty * sell_price
                pnl = revenue - (qty * pos['entry_price'])
                pnl_pct = (pnl / (qty * pos['entry_price'])) * 100

                cash += revenue
                trade_log.append({
                    "Date": current_date.date(), "Ticker": ticker, "Action": "SELL (100%)",
                    "Price": sell_price, "Qty": qty,
                    "Reason": sell_reason,
                    "PnL_Eur": pnl, "PnL_Pct": pnl_pct, "Capital": cash
                })
                del positions[ticker]

        # --- FASE B: RACCOLTA CANDIDATI (ACQUISTI) ---
        daily_candidates = []

        for ticker, df in market_data.items():
            if ticker in positions: continue
            if current_date not in df.index: continue

            daily = df.loc[current_date]
            curr_close = daily['Close']

            # Filtri Macro
            if curr_vix > 35: continue
            if not market_is_bullish: continue
            if pd.isna(daily['SMA200']): continue
            if curr_close < daily['SMA200']: continue

            upper_band = daily['SMA20'] + (2 * daily['STD20'])
            vol_sma = daily['Vol_SMA20']
            vol_rel = (daily['Volume'] / vol_sma) if (pd.notna(vol_sma) and vol_sma > 0) else 1.0
            adx_val = daily['ADX'] if pd.notna(daily['ADX']) else 0

            # 1. Breakout
            is_breakout = (curr_close > upper_band) and \
                          (curr_close > daily['SMA50']) and \
                          (vol_rel > 2.0) and \
                          (adx_val > 30)

            # 2. Dip
            is_dip = (curr_close > daily['SMA50']) and (daily['RSI'] < 35)

            if is_breakout or is_dip:
                # === CALCOLO SCORE CONFIDENCE (0-100) ===
                score = 0.0

                if is_breakout:
                    # Formula Breakout: ADX + Bonus Volumi - Malus RSI
                    score = adx_val
                    score += (vol_rel - 1.0) * 15
                    if daily['RSI'] > 75: score -= 10

                elif is_dip:
                    # Formula Dip: 100-RSI + Bonus Trend + Bonus Supporto
                    score = 100 - daily['RSI']
                    score += (adx_val / 2)

                    # Bonus Supporto SMA50
                    dist_sma50 = abs(curr_close - daily['SMA50']) / daily['SMA50']
                    if dist_sma50 < 0.02:  # Entro il 2% dalla SMA50
                        score += 15

                # === FILTRO RICHIESTO: Confidence >= 75 ===
                if score >= 75:
                    daily_candidates.append({
                        'ticker': ticker,
                        'price': curr_close,
                        'atr': daily['ATR'],
                        'high': daily['High'],
                        'type': "BREAKOUT" if is_breakout else "DIP",
                        'score': score,
                        'adx': adx_val,
                        'vol_rel': vol_rel
                    })

        # --- FASE C: ESECUZIONE ---
        daily_candidates.sort(key=lambda x: x['score'], reverse=True)

        for cand in daily_candidates:
            alloc = calculate_position_size(cash, portfolio_value, curr_vix)
            if cash < 200: break

            qty = int(alloc / cand['price'])

            if qty >= 1:
                cost = qty * cand['price']
                if cost > cash: continue

                cash -= cost

                curr_atr = cand['atr'] if pd.notna(cand['atr']) else (cand['price'] * 0.02)

                if cand['type'] == 'BREAKOUT':
                    sl_mult = 1.5
                    tp_mult = 3.0
                else:
                    sl_mult = 3.0
                    tp_mult = 4.0

                initial_stop = cand['price'] - (curr_atr * sl_mult)
                take_profit_target = cand['price'] + (curr_atr * tp_mult)

                positions[cand['ticker']] = {
                    'qty': qty,
                    'entry_price': cand['price'],
                    'trailing_stop': initial_stop,
                    'highest_price': cand['high'],
                    'take_profit': take_profit_target,
                    'sl_mult': sl_mult
                }

                reason_str = f"Score: {cand['score']:.0f} (ADX:{cand['adx']:.0f}, Vol:{cand['vol_rel']:.1f}x)"
                if use_tp_only:
                    reason_str += f" | TP: {take_profit_target:.2f}"

                trade_log.append({
                    "Date": current_date.date(), "Ticker": cand['ticker'],
                    "Action": f"BUY ({cand['type']}) 🚀",
                    "Price": cand['price'], "Qty": qty,
                    "Reason": reason_str,
                    "PnL_Eur": 0, "PnL_Pct": 0, "Capital": cash
                })

    if progress_bar: progress_bar.empty()

    for ticker, pos in positions.items():
        last_price = market_data[ticker]['Close'].iloc[-1]
        val = pos['qty'] * last_price
        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": ticker, "Action": "HELD (End)",
            "Price": last_price, "Qty": pos['qty'],
            "Reason": "End of Simulation",
            "PnL_Eur": val - (pos['qty'] * pos['entry_price']),
            "PnL_Pct": ((last_price / pos['entry_price']) - 1) * 100, "Capital": cash + val
        })
        cash += val

    return pd.DataFrame(trade_log), cash