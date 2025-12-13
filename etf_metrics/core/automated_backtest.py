# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
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
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = calculate_atr_series(df, window=1)
    atr = tr.rolling(window).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window).mean().abs() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window).mean()


def calculate_position_size(capital, current_portfolio_value, vix_value, base_alloc_pct=0.20):
    if pd.isna(vix_value) or vix_value <= 0: vix_value = 20.0

    # OTTIMIZZAZIONE PER CONTI PICCOLI VS GRANDI
    if current_portfolio_value < 2500:
        adjusted_pct = 0.95
    else:
        adjusted_pct = min(0.35, base_alloc_pct * (20.0 / vix_value))

    target_amount = current_portfolio_value * adjusted_pct
    return min(capital, target_amount)


def prepare_market_data(tickers, period="10y"):
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


def run_market_aware_backtest(tickers: list, start_date="2019-01-01", initial_capital=1000,
                              preloaded_data=None, commission=1.0, tax_rate=26.0):
    """
    Backtest fiscale realistico con strategia TRAILING STOP.
    """

    # 1. SELEZIONE FONTE DATI
    if preloaded_data:
        market_data = preloaded_data
    else:
        market_data = prepare_market_data(tickers)

    if not market_data: return pd.DataFrame(), initial_capital

    # Gestione dati ausiliari (VIX e SP500)
    try:
        vix_df = get_series("^VIX", period="10y", as_dataframe=True)
        vix_series = vix_df['Close'] if vix_df is not None else pd.Series(dtype=float)
        sp500_df = get_series("^GSPC", period="10y", as_dataframe=True)
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
    tax_credit = 0.0

    try:
        progress_bar = st.progress(0)
    except:
        progress_bar = None

    # 2. LOOP GIORNALIERO
    for i, current_date in enumerate(sim_dates):
        if progress_bar and i % 10 == 0:
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

        # Calcolo valore portafoglio (Mark to Market)
        portfolio_value = cash
        for t, p in positions.items():
            if current_date in market_data[t].index:
                curr_price = market_data[t].loc[current_date]['Close']
                portfolio_value += (p['qty'] * curr_price)

        # --- FASE A: GESTIONE VENDITE (SELL) ---
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

            # Logica di uscita: SOLO TRAILING STOP
            if curr_high > pos['highest_price']:
                positions[ticker]['highest_price'] = curr_high

            mult = pos.get('sl_mult', 3.0)
            potential_stop = positions[ticker]['highest_price'] - (curr_atr * mult)

            # Il trailing stop può solo salire
            if potential_stop > pos['trailing_stop']:
                positions[ticker]['trailing_stop'] = potential_stop

            if curr_close < pos['trailing_stop']:
                should_sell = True
                sell_price = curr_close
                sell_reason = f"📉 TRAILING STOP"

            # Hard stop emergenza
            if curr_close < (pos['entry_price'] * 0.85):
                should_sell = True
                sell_price = curr_close
                sell_reason = "🛑 HARD STOP"

            if should_sell:
                qty = pos['qty']

                # === CALCOLO FISCALE E COMMISSIONI ===
                gross_revenue = qty * sell_price
                net_revenue = gross_revenue - commission
                total_entry_cost = pos['total_cost_basis']
                capital_gain = net_revenue - total_entry_cost

                tax_amount = 0.0
                if capital_gain > 0:
                    taxable_gain = max(0, capital_gain - tax_credit)
                    tax_credit = max(0, tax_credit - capital_gain)
                    tax_amount = taxable_gain * (tax_rate / 100.0)
                else:
                    tax_credit += abs(capital_gain)

                final_cash_in = net_revenue - tax_amount
                cash += final_cash_in

                realized_pnl_eur = final_cash_in - total_entry_cost
                realized_pnl_pct = (realized_pnl_eur / total_entry_cost) * 100

                trade_log.append({
                    "Date": current_date.date(), "Ticker": ticker, "Action": "SELL",
                    "Price": sell_price, "Qty": qty,
                    "Reason": sell_reason,
                    "Comm": commission, "Tax": tax_amount,
                    "PnL_Net": realized_pnl_eur, "PnL_Pct": realized_pnl_pct,
                    "Capital": cash
                })
                del positions[ticker]

        # --- FASE B: GESTIONE ACQUISTI (BUY) ---
        daily_candidates = []

        for ticker, df in market_data.items():
            if ticker in positions: continue
            if current_date not in df.index: continue

            daily = df.loc[current_date]
            curr_close = daily['Close']

            # Filtri Macro
            if curr_vix > 35: continue
            if not market_is_bullish: continue
            if pd.isna(daily['SMA200']) or curr_close < daily['SMA200']: continue

            # Setup Breakout
            upper_band = daily['SMA20'] + (2 * daily['STD20'])
            vol_rel = (daily['Volume'] / daily['Vol_SMA20']) if (
                        pd.notna(daily['Vol_SMA20']) and daily['Vol_SMA20'] > 0) else 1.0
            adx_val = daily['ADX'] if pd.notna(daily['ADX']) else 0

            is_breakout = (curr_close > upper_band) and (vol_rel > 1.5) and (adx_val > 25)
            is_dip = (curr_close > daily['SMA50']) and (daily['RSI'] < 35)

            if is_breakout or is_dip:
                score = 0
                if is_breakout:
                    score = adx_val + ((vol_rel - 1.0) * 10)
                    if daily['RSI'] > 75: score -= 10
                elif is_dip:
                    score = (100 - daily['RSI']) + (adx_val / 2)

                if score >= 70:
                    daily_candidates.append({
                        'ticker': ticker, 'price': curr_close, 'atr': daily['ATR'],
                        'high': daily['High'], 'type': "BREAKOUT" if is_breakout else "DIP",
                        'score': score
                    })

        # --- FASE C: ESECUZIONE ORDINI ---
        daily_candidates.sort(key=lambda x: x['score'], reverse=True)

        for cand in daily_candidates:
            alloc_eur = calculate_position_size(cash, portfolio_value, curr_vix)
            max_buy_eur = alloc_eur - (commission * 2)

            if max_buy_eur < (cand['price']): continue

            qty = int(max_buy_eur / cand['price'])
            if qty < 1: continue

            cost_shares = qty * cand['price']
            total_cost = cost_shares + commission

            if total_cost > cash: continue

            cash -= total_cost

            curr_atr = cand['atr'] if pd.notna(cand['atr']) else (cand['price'] * 0.02)
            sl_mult = 2.0 if cand['type'] == 'BREAKOUT' else 3.0

            initial_stop = cand['price'] - (curr_atr * sl_mult)

            positions[cand['ticker']] = {
                'qty': qty,
                'entry_price': cand['price'],
                'total_cost_basis': total_cost,
                'trailing_stop': initial_stop,
                'highest_price': cand['high'],
                'sl_mult': sl_mult
            }

            trade_log.append({
                "Date": current_date.date(), "Ticker": cand['ticker'],
                "Action": f"BUY",
                "Price": cand['price'], "Qty": qty,
                "Reason": f"Score: {cand['score']:.0f}",
                "Comm": commission, "Tax": 0.0,
                "PnL_Net": 0, "PnL_Pct": 0,
                "Capital": cash
            })

    if progress_bar: progress_bar.empty()

    # Chiusura forzata posizioni finali
    for ticker, pos in positions.items():
        last_price = market_data[ticker]['Close'].iloc[-1]
        gross_rev = pos['qty'] * last_price
        net_rev = gross_rev - commission
        cap_gain = net_rev - pos['total_cost_basis']
        tax = (cap_gain * (tax_rate / 100)) if cap_gain > 0 else 0
        net_in = net_rev - tax
        realized_pnl = net_in - pos['total_cost_basis']
        cash += net_in

        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": ticker, "Action": "END",
            "Price": last_price, "Qty": pos['qty'],
            "Reason": "Chiusura Simulazione",
            "Comm": commission, "Tax": tax,
            "PnL_Net": realized_pnl, "PnL_Pct": 0, "Capital": cash
        })

    return pd.DataFrame(trade_log), cash