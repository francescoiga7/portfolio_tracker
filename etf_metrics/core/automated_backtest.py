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

    atr = tr.rolling(window).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window).mean().abs() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window).mean()
    return adx


def calculate_position_size(capital, current_portfolio_value, vix_value, base_alloc_pct=0.20):
    """Money Management VIX-Adjusted."""
    if pd.isna(vix_value) or vix_value <= 0: vix_value = 20.0
    adjusted_pct = min(0.35, base_alloc_pct * (20.0 / vix_value))
    target_amount = current_portfolio_value * adjusted_pct
    return min(capital, target_amount)


def run_market_aware_backtest(tickers: list, start_date="2021-01-01", initial_capital=10000):
    """
    Versione 4.0: "Market Guard" & "Smart Trend"
    1. Market Filter: Niente nuovi Long se S&P500 < SMA200.
    2. Trend Filter: Breakout validi solo se ADX > 20.
    3. Trailing Progressivo: Lo stop si stringe man mano che il profitto aumenta.
    """
    market_data = {}

    for t in tickers:
        df = get_series(t, period="5y", as_dataframe=True)
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

    if not market_data: return pd.DataFrame(), 0.0

    vix_df = get_series("^VIX", period="5y", as_dataframe=True)
    vix_series = vix_df['Close'] if vix_df is not None else pd.Series()

    sp500_df = get_series("^GSPC", period="5y", as_dataframe=True)
    if sp500_df is not None:
        sp500_df['SMA200'] = sp500_df['Close'].rolling(200).mean()

    sample_ticker = list(market_data.keys())[0]
    sim_dates = market_data[sample_ticker].index[market_data[sample_ticker].index >= pd.to_datetime(start_date)]

    cash = initial_capital
    positions = {}
    trade_log = []
    progress_bar = st.progress(0)

    for i, current_date in enumerate(sim_dates):
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

        portfolio_value = cash
        for t, p in positions.items():
            if current_date in market_data[t].index:
                portfolio_value += p['qty'] * market_data[t].loc[current_date]['Close']
            else:
                portfolio_value += p['qty'] * p['entry_price']

        for ticker, df in market_data.items():
            if current_date not in df.index: continue

            daily = df.loc[current_date]
            curr_close = daily['Close']
            curr_high = daily['High']
            curr_atr = daily['ATR']

            if pd.isna(curr_atr): continue

            if ticker in positions:
                pos = positions[ticker]

                current_roi_pct = ((curr_high / pos['entry_price']) - 1) * 100

                dynamic_mult = max(1.5, 3.0 - (current_roi_pct / 20.0))

                if curr_high > pos['highest_price']:
                    positions[ticker]['highest_price'] = curr_high

                potential_stop = positions[ticker]['highest_price'] - (curr_atr * dynamic_mult)

                if potential_stop > pos['trailing_stop']:
                    positions[ticker]['trailing_stop'] = potential_stop

                if curr_close < pos['trailing_stop']:
                    qty = pos['qty']
                    revenue = qty * curr_close
                    pnl = revenue - (qty * pos['entry_price'])
                    pnl_pct = (pnl / (qty * pos['entry_price'])) * 100

                    cash += revenue
                    trade_log.append({
                        "Date": current_date.date(), "Ticker": ticker, "Action": "SELL (100%)",
                        "Price": curr_close, "Qty": qty,
                        "Reason": f"TRAILING STOP (Mult {dynamic_mult:.1f})",
                        "PnL_Eur": pnl, "PnL_Pct": pnl_pct, "Capital": cash
                    })
                    del positions[ticker]
                    continue

            else:
                if curr_vix > 40: continue

                if not market_is_bullish: continue

                if pd.isna(daily['SMA200']): continue
                if curr_close < daily['SMA200']: continue

                upper_band = daily['SMA20'] + (2 * daily['STD20'])
                vol_ok = daily['Volume'] > (daily['Vol_SMA20'] * 1.2)
                adx_ok = daily['ADX'] > 20

                is_breakout = (curr_close > upper_band) and (curr_close > daily['SMA50']) and vol_ok and adx_ok

                is_dip = (curr_close > daily['SMA50']) and (daily['RSI'] < 35)

                if is_breakout or is_dip:
                    alloc = calculate_position_size(cash, portfolio_value, curr_vix)
                    qty = int(alloc / curr_close)

                    if qty >= 1:
                        cost = qty * curr_close
                        cash -= cost

                        initial_stop = curr_close - (curr_atr * 3.0)

                        positions[ticker] = {
                            'qty': qty,
                            'entry_price': curr_close,
                            'trailing_stop': initial_stop,
                            'highest_price': curr_high,
                        }

                        signal_type = "BREAKOUT" if is_breakout else "DIP"
                        reason_str = f"VIX: {curr_vix:.1f}"
                        if is_breakout: reason_str += f" | ADX: {daily['ADX']:.1f}"

                        trade_log.append({
                            "Date": current_date.date(), "Ticker": ticker,
                            "Action": f"BUY ({signal_type}) 🛰️",
                            "Price": curr_close, "Qty": qty,
                            "Reason": reason_str,
                            "PnL_Eur": 0, "PnL_Pct": 0, "Capital": cash
                        })

    progress_bar.empty()

    for ticker, pos in positions.items():
        last_price = market_data[ticker]['Close'].iloc[-1]
        val = pos['qty'] * last_price
        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": ticker, "Action": "HELD (Open)",
            "Price": last_price, "Qty": pos['qty'],
            "Reason": f"Stop: {pos['trailing_stop']:.2f}",
            "PnL_Eur": val - (pos['qty'] * pos['entry_price']),
            "PnL_Pct": ((last_price / pos['entry_price']) - 1) * 100, "Capital": cash + val
        })
        cash += val

    return pd.DataFrame(trade_log), cash