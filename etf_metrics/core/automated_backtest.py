# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import yfinance as yf
from etf_metrics.core.metrics import calculate_atr_series, calculate_adx_series


def _get_scalar(val):
    """Helper per estrarre un float pulito da Series, DataFrame o scalari."""
    if isinstance(val, (pd.Series, pd.DataFrame)):
        if val.empty:
            return 0.0
        val = val.iloc[0]

    try:
        return float(val)
    except Exception:
        return 0.0


def calculate_position_size(capital, current_portfolio_value, vix_value, base_alloc_pct=0.20):
    if pd.isna(vix_value) or vix_value <= 0: vix_value = 20.0

    if current_portfolio_value < 2500:
        adjusted_pct = 0.95
    else:
        adjusted_pct = min(0.35, base_alloc_pct * (20.0 / vix_value))

    target_amount = current_portfolio_value * adjusted_pct
    return min(capital, target_amount)


def prepare_market_data(tickers, period="10y"):
    market_data = {}
    if not tickers: return {}

    try:
        bulk_data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=False, threads=False)
    except Exception as e:
        st.error(f"Errore download backtest: {e}")
        return {}

    is_single = len(tickers) == 1

    for t in tickers:
        try:
            if is_single:
                df = bulk_data.copy()
            else:
                df = bulk_data[t].copy()

            df = df.dropna(how='all')
            df = df[~df.index.duplicated(keep='first')]

            if df.empty: continue

            if df.index.tz is not None: df.index = df.index.tz_localize(None)

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
        except KeyError:
            pass

    return market_data


def run_market_aware_backtest(tickers: list, start_date="2019-01-01", initial_capital=10000, use_tp_only=False,
                              preloaded_data=None, commission=1.0, tax_rate=26.0):
    """
    Backtest fiscale realistico con correzione robusta per i tipi di dati.
    """
    if preloaded_data:
        market_data = preloaded_data
    else:
        market_data = prepare_market_data(tickers)

    if not market_data: return pd.DataFrame(), initial_capital

    try:
        vix_df = yf.download("^VIX", period="10y", auto_adjust=False, progress=False)
        if not vix_df.empty:
            vix_df = vix_df[~vix_df.index.duplicated(keep='first')]
            if vix_df.index.tz is not None: vix_df.index = vix_df.index.tz_localize(None)
            vix_series = vix_df['Close']
        else:
            vix_series = pd.Series(dtype=float)

        sp500_df = yf.download("^GSPC", period="10y", auto_adjust=False, progress=False)
        if not sp500_df.empty:
            sp500_df = sp500_df[~sp500_df.index.duplicated(keep='first')]
            if sp500_df.index.tz is not None: sp500_df.index = sp500_df.index.tz_localize(None)
            sp500_df['SMA200'] = sp500_df['Close'].rolling(200).mean()
        else:
            sp500_df = None
    except:
        vix_series = pd.Series(dtype=float)
        sp500_df = None

    sample_ticker = list(market_data.keys())[0]
    sim_dates = market_data[sample_ticker].index[market_data[sample_ticker].index >= pd.to_datetime(start_date)]

    cash = initial_capital
    positions = {}
    trade_log = []
    tax_credit = 0.0

    for i, current_date in enumerate(sim_dates):
        try:
            if not vix_series.empty:
                curr_vix = _get_scalar(vix_series.asof(current_date))
            else:
                curr_vix = 20.0
        except:
            curr_vix = 20.0

        market_is_bullish = True
        if sp500_df is not None:
            try:
                if current_date in sp500_df.index:
                    sp_today = sp500_df.loc[current_date]
                    if isinstance(sp_today, pd.DataFrame): sp_today = sp_today.iloc[-1]

                    sp_close = _get_scalar(sp_today['Close'])
                    sp_sma200 = _get_scalar(sp_today['SMA200'])

                    if sp_sma200 > 0 and sp_close < sp_sma200:
                        market_is_bullish = False
            except Exception:
                pass

        portfolio_value = cash
        for t, p in positions.items():
            if current_date in market_data[t].index:
                daily_data = market_data[t].loc[current_date]
                if isinstance(daily_data, pd.DataFrame): daily_data = daily_data.iloc[-1]
                curr_price = _get_scalar(daily_data['Close'])
                portfolio_value += (p['qty'] * curr_price)

        tickers_with_data = [t for t in positions.keys() if current_date in market_data[t].index]

        for ticker in tickers_with_data:
            pos = positions[ticker]
            df = market_data[ticker]
            daily = df.loc[current_date]
            if isinstance(daily, pd.DataFrame): daily = daily.iloc[-1]

            curr_close = _get_scalar(daily['Close'])
            curr_high = _get_scalar(daily['High'])
            curr_atr = _get_scalar(daily['ATR'])

            if curr_atr == 0: continue

            should_sell = False
            sell_price = curr_close
            sell_reason = ""

            if use_tp_only:
                if curr_high >= pos['take_profit']:
                    should_sell = True
                    sell_price = pos['take_profit']
                    sell_reason = "TARGET"
                elif curr_close < (pos['entry_price'] * 0.85):
                    should_sell = True
                    sell_price = curr_close
                    sell_reason = "HARD STOP"
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
                    sell_reason = "TRAILING STOP"

            if should_sell:
                qty = pos['qty']
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

                trade_log.append({
                    "Date": current_date.date(), "Ticker": ticker, "Action": "SELL",
                    "Price": sell_price, "Qty": qty, "Reason": sell_reason,
                    "Comm": commission, "Tax": tax_amount,
                    "PnL_Net": realized_pnl_eur, "Capital": cash
                })
                del positions[ticker]

        daily_candidates = []
        for ticker, df in market_data.items():
            if ticker in positions: continue
            if current_date not in df.index: continue

            daily = df.loc[current_date]
            if isinstance(daily, pd.DataFrame): daily = daily.iloc[-1]

            curr_close = _get_scalar(daily['Close'])
            daily_sma200 = _get_scalar(daily['SMA200'])

            if curr_vix > 35: continue
            if not market_is_bullish: continue
            if daily_sma200 == 0 or curr_close < daily_sma200: continue

            sma20 = _get_scalar(daily['SMA20'])
            std20 = _get_scalar(daily['STD20'])
            vol_sma20 = _get_scalar(daily['Vol_SMA20'])
            volume = _get_scalar(daily['Volume'])
            rsi = _get_scalar(daily['RSI'])
            adx_val = _get_scalar(daily['ADX'])
            daily_sma50 = _get_scalar(daily['SMA50'])

            upper_band = sma20 + (2 * std20)
            vol_rel = (volume / vol_sma20) if vol_sma20 > 0 else 1.0

            is_breakout = (curr_close > upper_band) and (vol_rel > 1.5) and (adx_val > 25)
            is_dip = (curr_close > daily_sma50) and (rsi < 35)

            if is_breakout or is_dip:
                score = 0
                if is_breakout:
                    score = adx_val + ((vol_rel - 1.0) * 10)
                    if rsi > 75: score -= 10
                elif is_dip:
                    score = (100 - rsi) + (adx_val / 2)

                if score >= 70:
                    curr_atr = _get_scalar(daily['ATR'])
                    if curr_atr == 0: curr_atr = curr_close * 0.02
                    curr_high = _get_scalar(daily['High'])

                    daily_candidates.append({
                        'ticker': ticker, 'price': curr_close, 'atr': curr_atr,
                        'high': curr_high, 'type': "BREAKOUT" if is_breakout else "DIP",
                        'score': score
                    })

        daily_candidates.sort(key=lambda x: x['score'], reverse=True)

        for cand in daily_candidates:
            alloc_eur = calculate_position_size(cash, portfolio_value, curr_vix)
            max_buy_eur = alloc_eur - (commission * 2)

            if max_buy_eur < cand['price']: continue
            qty = int(max_buy_eur / cand['price'])
            if qty < 1: continue

            total_cost = (qty * cand['price']) + commission
            if total_cost > cash: continue

            cash -= total_cost

            sl_mult = 2.0 if cand['type'] == 'BREAKOUT' else 3.0
            tp_mult = 4.0

            initial_stop = cand['price'] - (cand['atr'] * sl_mult)
            take_profit_target = cand['price'] + (cand['atr'] * tp_mult)

            positions[cand['ticker']] = {
                'qty': qty, 'entry_price': cand['price'], 'total_cost_basis': total_cost,
                'trailing_stop': initial_stop, 'highest_price': cand['high'],
                'take_profit': take_profit_target, 'sl_mult': sl_mult
            }

            trade_log.append({
                "Date": current_date.date(), "Ticker": cand['ticker'],
                "Action": "BUY",
                "Price": cand['price'], "Qty": qty,
                "Reason": f"Score: {cand['score']:.0f}",
                "Comm": commission, "Tax": 0.0,
                "PnL_Net": 0, "Capital": cash
            })

    return pd.DataFrame(trade_log), cash