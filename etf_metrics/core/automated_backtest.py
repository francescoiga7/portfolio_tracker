# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from etf_metrics.clients.yahoo_client import get_series

# --- CONFIGURAZIONE V25 (THE SMART HOLDER) ---
MAX_POSITIONS = 4  # Top 4 titoli (25% ciascuno)
REBALANCE_DAYS = 20  # Ribilanciamento Mensile


def calculate_indicators(df):
    """
    Calcola indicatori per V25.
    Solo SMA200 (Trend Lungo) e Momentum (Forza Relativa).
    """
    # 1. Trend Filter (L'unico che conta davvero per il lungo termine)
    df['SMA200'] = df['Close'].rolling(200).mean()

    # 2. Momentum a 6 mesi (126 giorni) - Classico Dual Momentum
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
    """
    Backtest V25 (The Smart Holder):
    - FREQUENZA: Mensile (Ogni 20gg). Zero stress intraday.
    - SELEZIONE: Top 4 titoli per Momentum a 6 mesi.
    - FILTRO ON/OFF: Si compra/tiene SOLO se Prezzo > SMA200.
      Se un titolo va sotto la SMA200, si vende e si tiene il CASH.
    - NO STOP LOSS, NO TAKE PROFIT: Si cavalca il trend finché dura.
    """

    if preloaded_data:
        market_data = preloaded_data
    else:
        market_data = prepare_market_data(tickers)

    if not market_data: return pd.DataFrame(), initial_capital

    sample_ticker = list(market_data.keys())[0]
    sim_dates = market_data[sample_ticker].index[market_data[sample_ticker].index >= pd.to_datetime(start_date)]

    cash = initial_capital
    positions = {}  # {ticker: {qty, entry_price, cost_basis, last_value}}
    trade_log = []
    tax_credit = 0.0

    try:
        progress_bar = st.progress(0)
    except:
        progress_bar = None

    days_counter = 0

    # 2. LOOP TEMPORALE
    for i, current_date in enumerate(sim_dates):
        if progress_bar and i % 20 == 0:
            progress_bar.progress((i + 1) / len(sim_dates))

        days_counter += 1

        # --- A. AGGIORNAMENTO EQUITY ---
        portfolio_equity = cash
        for t, pos in positions.items():
            if current_date in market_data[t].index:
                curr_price = market_data[t].loc[current_date]['Close']
                pos['last_known_price'] = curr_price
                portfolio_equity += (pos['qty'] * curr_price)
            else:
                portfolio_equity += (pos['qty'] * pos.get('last_known_price', 0))

        # --- B. HARD STOP DI ESTREMA EMERGENZA (OPZIONALE - Cigno Nero) ---
        # Solo se un titolo crolla del 30% in un giorno, usciamo.
        tickers_to_dump = []
        for t, pos in positions.items():
            if current_date in market_data[t].index:
                curr_close = market_data[t].loc[current_date]['Close']
                if curr_close < (pos['entry_price'] * 0.70):
                    tickers_to_dump.append((t, curr_close, "☠️ BLACK SWAN STOP (-30%)"))

        for t, price, reason in tickers_to_dump:
            pos = positions[t]
            gross = pos['qty'] * price
            net = gross - commission
            gain = net - pos['cost_basis']
            tax = gain * (tax_rate / 100) if gain > 0 else 0
            cash += (net - tax)
            portfolio_equity -= (pos['qty'] * price)

            trade_log.append({
                "Date": current_date.date(), "Ticker": t, "Action": "SELL",
                "Price": price, "Qty": pos['qty'], "Reason": reason,
                "Comm": commission, "Tax": tax,
                "PnL_Net": (net - tax) - pos['cost_basis'],
                "PnL_Pct": 0, "Capital": cash, "Portfolio_Value": cash + portfolio_equity
            })
            del positions[t]

        # --- C. RIBILANCIAMENTO MENSILE ---
        if days_counter >= REBALANCE_DAYS:
            days_counter = 0

            # 1. Analisi Candidati
            candidates = []
            for t, df in market_data.items():
                if current_date not in df.index: continue
                daily = df.loc[current_date]

                if pd.isna(daily['SMA200']) or pd.isna(daily['Momentum']): continue

                # IL FILTRO D'ORO: Siamo sopra la media a 200 giorni?
                # Se sì, il titolo è "sano". Se no, è "malato" -> Cash.
                if daily['Close'] < daily['SMA200']: continue

                # Filtro Momentum Positivo (Deve salire)
                if daily['Momentum'] <= 0: continue

                candidates.append({
                    'ticker': t,
                    'score': daily['Momentum'],
                    'close': daily['Close']
                })

            # Ordina per Momentum (Forza Relativa)
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_picks = candidates[:MAX_POSITIONS]
            top_tickers = [c['ticker'] for c in top_picks]

            # 2. VENDITE (Rotazione Out)
            tickers_out = []
            for t in list(positions.keys()):
                # Vendiamo se:
                # a) Non è più nella Top 4 (abbiamo trovato di meglio)
                # b) È sceso sotto la SMA 200 (è diventato ribassista) -> Questo gestisce il 2022

                keep = False
                if t in top_tickers:
                    # Verifica extra: è ancora sopra la SMA200 oggi?
                    if current_date in market_data[t].index:
                        curr_price = market_data[t].loc[current_date]['Close']
                        sma200 = market_data[t].loc[current_date]['SMA200']
                        if curr_price > sma200:
                            keep = True

                if not keep:
                    if current_date in market_data[t].index:
                        curr_price = market_data[t].loc[current_date]['Close']
                        tickers_out.append((t, curr_price, "📉 MONTHLY ROTATION"))

            for t, price, reason in tickers_out:
                pos = positions[t]
                gross = pos['qty'] * price
                net = gross - commission
                gain = net - pos['cost_basis']
                tax = gain * (tax_rate / 100) if gain > 0 else 0
                cash += (net - tax)
                portfolio_equity -= (pos['qty'] * price)

                trade_log.append({
                    "Date": current_date.date(), "Ticker": t, "Action": "SELL",
                    "Price": price, "Qty": pos['qty'], "Reason": reason,
                    "Comm": commission, "Tax": tax,
                    "PnL_Net": (net - tax) - pos['cost_basis'],
                    "PnL_Pct": 0, "Capital": cash, "Portfolio_Value": cash + portfolio_equity
                })
                del positions[t]

            # 3. ACQUISTI (Riempiamo i buchi)
            free_slots = MAX_POSITIONS - len(positions)

            # Calcolo budget dinamico: Cash / Slot Liberi
            if free_slots > 0 and cash > 50:
                budget_per_slot = cash / free_slots

                for cand in top_picks:
                    if cand['ticker'] in positions: continue
                    if free_slots <= 0: break

                    if budget_per_slot < 50: continue

                    # Acquisto Frazionato
                    qty = round(budget_per_slot / cand['close'], 4)
                    cost = (qty * cand['close']) + commission

                    if cost <= cash:
                        cash -= cost
                        free_slots -= 1

                        positions[cand['ticker']] = {
                            'qty': qty,
                            'entry_price': cand['close'],
                            'cost_basis': cost,
                            'last_known_price': cand['close']
                        }

                        portfolio_equity += (qty * cand['close'])

                        trade_log.append({
                            "Date": current_date.date(), "Ticker": cand['ticker'], "Action": "BUY",
                            "Price": cand['close'], "Qty": qty,
                            "Reason": f"SMART ENTRY (Mom: {cand['score']:.2%})",
                            "Comm": commission, "Tax": 0.0, "PnL_Net": 0, "PnL_Pct": 0,
                            "Capital": cash, "Portfolio_Value": cash + portfolio_equity
                        })

    if progress_bar: progress_bar.empty()

    # Chiusura Finale
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
            "Capital": cash, "Portfolio_Value": cash
        })

    return pd.DataFrame(trade_log), cash