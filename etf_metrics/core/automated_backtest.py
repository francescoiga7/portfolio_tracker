# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import logging
from datetime import timedelta
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.data_manager import MarketDataManager

# NOTA: Importiamo concetti logici dal chatbot, ma li implementiamo in modo vettoriale
# per velocità (evitando chiamate API esterne durante il backtest).

logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE "AI ENHANCED STRATEGY" ---
CONFIG = {
    'MAX_POSITIONS': 3,  # Focus su poche idee migliori
    'REBALANCE_DAYS': 5,  # Controllo settimanale

    # 1. GESTIONE COSTI E LIMITI
    'ENABLE_MONTHLY_LIMIT': True,
    'MAX_BUYS_PER_MONTH': 4,  # Aumentato leggermente per permettere rotazione
    'COMMISSION': 2.0,  # Costo per trade

    # 2. FILTRO MACRO (SPY)
    'SPY_TICKER': 'SPY',
    'MACRO_TREND_FILTER': True,
    'BEAR_VOLATILITY_THRESHOLD': 3.0,  # Se ATR di SPY esplode, andiamo cash

    # 3. CRITERI DI QUALITÀ (Derived from Chatbot logic)
    'MIN_RS_SCORE': 80,  # Relative Strength percentile (Top 20% del mercato)
    'MIN_ADX': 25,  # Trend deciso
    'MIN_PROX_HIGH': 0.85,  # (Chatbot Insight) Deve essere entro il 15% dai massimi a 52w

    # 4. GESTIONE RISCHIO DINAMICA
    'STOP_LOSS_ATR_MULT': 2.5,  # Stop più stretto (era 3.0)
    'TIME_STOP_DAYS': 21,  # Dai più tempo al trade se è in trend
    'TP1_PCT': 0.08,  # Prendi profitto a +8% (non +5%, lascia correre di più)

    # 5. DIVERSIFICAZIONE
    'MAX_CORRELATION': 0.65
}


def calculate_advanced_metrics_vectorized(df, spy_df=None):
    """
    Calcola indicatori tecnici avanzati ispirati alla 'biopsia' del chatbot.
    """
    if 'Close' not in df.columns: return df

    # --- 1. TREND BASE ---
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    # --- 2. VOLATILITÀ (ATR & DEVIAZIONE) ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()

    # Volatilità storica (per penalizzare nello score)
    df['Vol_20'] = df['Close'].pct_change().rolling(20).std() * 100

    # --- 3. MOMENTUM (RSI) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- 4. ADX (Forza del Trend) ---
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr14 = ranges.max(axis=1).rolling(14).sum()
    plus_di14 = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di14 = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    df['ADX'] = dx.rolling(14).mean()

    # --- 5. CHATBOT METRIC: DISTANZA DAI MASSIMI (Prox High) ---
    # Fondamentale per evitare "falling knives" nel 2026
    df['High_52w'] = df['Close'].rolling(252).max()
    df['Prox_High'] = df['Close'] / df['High_52w']

    # --- 6. RELATIVE STRENGTH vs SPY ---
    df['RS_Score'] = 0.0
    if spy_df is not None:
        common = df.index.intersection(spy_df.index)
        if len(common) > 65:
            # Calcolo RS Mansfield Style (Performance relativa a 3 mesi)
            stock_ret = df['Close'].pct_change(63)
            spy_ret = spy_df['Close'].pct_change(63)
            # Normalizziamo su scala 0-100 (approssimata) per semplificare lo scoring
            rel_perf = (stock_ret.loc[common] - spy_ret.loc[common]) * 100
            df.loc[common, 'RS_Score'] = rel_perf

    return df


def _assess_market_regime(price_matrix, current_date, market_data):
    """
    Valuta se il mercato è sicuro (BULL) o pericoloso (BEAR/VOLATILE).
    Usa SPY come proxy.
    """
    spy = CONFIG['SPY_TICKER']

    # Fallback se non abbiamo dati SPY
    if spy not in market_data or current_date not in market_data[spy].index:
        # Usa la media di tutti i titoli come proxy
        start = current_date - timedelta(days=200)
        proxy = price_matrix.loc[start:current_date].mean(axis=1)
        if len(proxy) < 200: return "NEUTRAL"
        return "BULL" if proxy.iloc[-1] > proxy.mean() else "BEAR"

    row = market_data[spy].loc[current_date]

    # 1. Filtro SMA Classico
    if row['Close'] < row['SMA200']:
        return "BEAR"

    # 2. Filtro Volatilità (Safety Check)
    # Se SPY è molto volatile (es. ATR > 3% del prezzo), è un mercato nervoso
    atr_pct = (row['ATR'] / row['Close']) * 100
    if atr_pct > CONFIG['BEAR_VOLATILITY_THRESHOLD']:
        return "VOLATILE"

    # 3. Crash Protection
    if row['RSI'] < 35:  # Ipervenduto estremo su indici spesso anticipa crash ulteriori
        return "DANGER"

    return "BULL"


def check_correlation_strict(candidate, portfolio, price_matrix, current_date):
    if not portfolio: return True, None
    if candidate not in price_matrix.columns: return True, None

    # Usiamo 60 giorni per correlazione più stabile
    start = current_date - timedelta(days=60)
    hist = price_matrix.loc[start:current_date]
    if len(hist) < 30: return True, None

    cand_series = hist[candidate]
    for p in portfolio:
        if p not in hist.columns: continue
        corr = cand_series.corr(hist[p])
        if corr > CONFIG['MAX_CORRELATION']:
            return False, p
    return True, None


def calculate_ai_smart_score(row):
    """
    Genera uno score basato sulla logica del Chatbot (Biopsia).
    """
    # 1. HARD FILTERS (Gatekeepers)
    if row['Close'] < row['SMA50']: return 0, "Below SMA50"
    if row['Prox_High'] < CONFIG['MIN_PROX_HIGH']: return 0, "Too far from Highs"  # Evita titoli crollati
    if row['ADX'] < CONFIG['MIN_ADX']: return 0, "Weak Trend"

    # 2. SCORING SYSTEM (0-100)
    score = 50

    # Momentum Bonus
    score += (row['RS_Score'] * 2)  # Più forte è rispetto a SPY, meglio è

    # Trend Strength Bonus
    score += (row['ADX'] / 2)

    # Proximity Bonus (Preferiamo titoli sui massimi -> Breakout)
    if row['Prox_High'] > 0.95: score += 10

    # Volatility Penalty (Evita titoli "pazzi")
    if row['Vol_20'] > 5.0: score -= 15  # Penalizza alta volatilità giornaliera

    # RSI Check (Non comprare ipercomprato estremo)
    if row['RSI'] > 85: score -= 25

    return max(0, score), f"AI Score: {score:.0f} | ProxHigh: {row['Prox_High']:.2f}"


def prepare_market_data(tickers, period="5y"):  # Periodo ridotto per velocità
    db_manager = MarketDataManager()
    # Aggiungi SPY se manca
    all_tickers = list(set(tickers + [CONFIG['SPY_TICKER']]))

    # Caricamento e aggiornamento dati
    missing = db_manager.get_tickers_needing_update(all_tickers)
    if missing:
        new_data = {t: get_series(t, period=period, as_dataframe=True) for t in missing}
        # Pulisci None
        new_data = {k: v for k, v in new_data.items() if v is not None and not v.empty}
        if new_data: db_manager.save_bulk_data(new_data)

    loaded = db_manager.load_data(all_tickers)
    spy_df = loaded.get(CONFIG['SPY_TICKER'])

    # Pre-calcolo indicatori su SPY
    if spy_df is not None and not spy_df.empty:
        spy_df = calculate_advanced_metrics_vectorized(spy_df, None)  # SPY non ha benchmark
        loaded[CONFIG['SPY_TICKER']] = spy_df

    processed = {}
    for t, df in loaded.items():
        if not df.empty and len(df) > 200:  # Minimo storico necessario
            processed[t] = calculate_advanced_metrics_vectorized(df, spy_df)

    return processed


def run_market_aware_backtest(tickers: list, start_date="2015-01-01", initial_capital=10000,
                              preloaded_data=None, tax_rate=26.0):
    # --- PREPARAZIONE DATI ---
    if preloaded_data:
        market_data = preloaded_data
    else:
        market_data = prepare_market_data(tickers)

    if not market_data: return pd.DataFrame(), initial_capital

    ref = CONFIG['SPY_TICKER'] if CONFIG['SPY_TICKER'] in market_data else list(market_data.keys())[0]
    # Filtra date valide
    sim_dates = market_data[ref].index[market_data[ref].index >= pd.to_datetime(start_date)]

    # Matrice prezzi per calcoli veloci correlazione/proxy
    price_matrix = pd.DataFrame({t: d['Close'] for t, d in market_data.items()}).fillna(method='ffill')

    # --- STATO PORTAFOGLIO ---
    cash = initial_capital
    positions = {}  # {ticker: {qty, cost, stop, ...}}
    trade_log = []
    tax_credit = 0.0  # Minusvalenze

    # Variabili di stato
    current_sim_month = -1
    buys_this_month = 0

    try:
        progress = st.progress(0)
    except:
        progress = None

    # --- LOOP DI SIMULAZIONE ---
    for i, current_date in enumerate(sim_dates):
        if progress and i % 50 == 0: progress.progress((i + 1) / len(sim_dates))

        # Reset mensile
        if current_date.month != current_sim_month:
            current_sim_month = current_date.month
            buys_this_month = 0

        # ANALISI MACRO (Regime)
        regime = _assess_market_regime(price_matrix, current_date, market_data)
        is_bull = (regime == "BULL")
        is_crash = (regime == "DANGER" or regime == "VOLATILE")

        tokens_to_sell = []

        # ----------------------------
        # 1. GESTIONE USCITE (SELL)
        # ----------------------------
        for t, pos in positions.items():
            if t not in market_data or current_date not in market_data[t].index: continue

            row = market_data[t].loc[current_date]
            curr_price = row['Close']
            high = row['High']
            low = row['Low']
            atr = row['ATR']

            # A) STOP LOSS HARD (Basato su ATR)
            if low < pos['stop_loss']:
                exit_price = max(row['Open'], pos['stop_loss'])  # Slippage simulato
                tokens_to_sell.append((t, exit_price, "STOP LOSS", 1.0))
                continue

            # B) TAKE PROFIT PARZIALE (Let Winners Run logic)
            roi = (curr_price / pos['entry_price']) - 1
            if not pos.get('tp1_taken', False) and roi >= CONFIG['TP1_PCT']:
                # Vendi solo il 33%, sposta stop a Breakeven
                tokens_to_sell.append((t, curr_price, "TP1 (Lock)", 0.33))
                positions[t]['stop_loss'] = pos['entry_price'] * 1.01
                positions[t]['tp1_taken'] = True
                continue

            # C) TRAILING STOP DINAMICO
            # Se siamo in profitto o TP1 preso, alziamo lo stop
            if pos.get('tp1_taken', False) or roi > 0.03:
                # Usa un Chandelier Exit (High - 2.5 ATR)
                new_stop = curr_price - (atr * CONFIG['STOP_LOSS_ATR_MULT'])
                if new_stop > positions[t]['stop_loss']:
                    positions[t]['stop_loss'] = new_stop

            # D) MACRO EXIT (Emergency Brake)
            # Se il mercato crolla (Regime DANGER) e il titolo sta perdendo forza, esci.
            if is_crash and roi < 0.05:
                tokens_to_sell.append((t, curr_price, "MACRO RISK", 1.0))
                continue

            # E) TIME STOP (Dead Money)
            # Se dopo X giorni non ha performato, libera capitale
            days_held = (current_date - pos['entry_date']).days
            if days_held >= CONFIG['TIME_STOP_DAYS'] and roi < 0.01:
                tokens_to_sell.append((t, curr_price, "TIME STOP", 1.0))

        # ESECUZIONE VENDITE
        for t, price, reason, portion in tokens_to_sell:
            pos = positions[t]
            qty_sell = pos['qty'] * portion
            if qty_sell <= 0: continue

            net = (qty_sell * price) - CONFIG['COMMISSION']
            cost_portion = pos['cost_basis'] * portion
            gain = net - cost_portion

            # Calcolo Tasse (Zainetto Fiscale)
            tax = 0.0
            if gain > 0:
                taxable = max(0, gain - tax_credit)
                tax = taxable * (tax_rate / 100.0)
                tax_credit = max(0, tax_credit - gain)  # riduci minus usata
            else:
                tax_credit += abs(gain)  # accumula minus

            cash += (net - tax)

            # Logging
            trade_log.append({
                "Date": current_date.date(), "Ticker": t, "Action": "SELL",
                "Price": price, "Reason": reason,
                "PnL_Net": gain - tax, "Capital": cash
            })

            # Aggiorna posizione
            if portion >= 0.99:
                del positions[t]
            else:
                positions[t]['qty'] -= qty_sell
                positions[t]['cost_basis'] -= cost_portion

        # ----------------------------
        # 2. GESTIONE INGRESSI (BUY)
        # ----------------------------
        # Si compra solo in regime BULL e se non abbiamo superato il limite mensile
        buy_allowed = is_bull and (not CONFIG['ENABLE_MONTHLY_LIMIT'] or buys_this_month < CONFIG['MAX_BUYS_PER_MONTH'])

        # Check rebalance day
        if buy_allowed and i % CONFIG['REBALANCE_DAYS'] == 0:
            free_slots = CONFIG['MAX_POSITIONS'] - len(positions)

            if free_slots > 0 and cash > 2000:
                candidates = []

                # Screening veloce
                for t in tickers:
                    if t == CONFIG['SPY_TICKER'] or t in positions: continue
                    if t not in market_data or current_date not in market_data[t].index: continue

                    row = market_data[t].loc[current_date]

                    # Usa il nuovo AI Smart Score
                    score, reason = calculate_ai_smart_score(row)

                    if score >= 60:  # Threshold più alta (Qualità sopra quantità)
                        candidates.append({'t': t, 'score': score, 'row': row, 'reason': reason})

                # Ordina per score (i migliori prima)
                candidates.sort(key=lambda x: x['score'], reverse=True)

                for cand in candidates:
                    if free_slots <= 0 or cash < 1000: break
                    if CONFIG['ENABLE_MONTHLY_LIMIT'] and buys_this_month >= CONFIG['MAX_BUYS_PER_MONTH']: break

                    t = cand['t']
                    row = cand['row']

                    # Check Correlazione
                    is_safe_corr, conflict = check_correlation_strict(t, list(positions.keys()), price_matrix,
                                                                      current_date)
                    if not is_safe_corr: continue

                    # Position Sizing
                    # Se il mercato è incerto (BULL ma vicino a VOLATILE), riduci la size
                    risk_factor = 1.0
                    # Allocazione dinamica
                    alloc_per_slot = (cash / free_slots) * risk_factor * 0.98

                    qty = alloc_per_slot / row['Close']
                    cost = (qty * row['Close']) + CONFIG['COMMISSION']

                    if cost <= cash:
                        cash -= cost
                        # Stop Loss Iniziale (ATR Based)
                        initial_stop = row['Close'] - (row['ATR'] * CONFIG['STOP_LOSS_ATR_MULT'])

                        positions[t] = {
                            'qty': qty,
                            'cost_basis': cost,
                            'entry_price': row['Close'],
                            'entry_date': current_date,
                            'stop_loss': initial_stop,
                            'tp1_taken': False
                        }

                        trade_log.append({
                            "Date": current_date.date(), "Ticker": t, "Action": "BUY",
                            "Price": row['Close'], "Reason": cand['reason'],
                            "PnL_Net": 0.0, "Capital": cash
                        })
                        free_slots -= 1
                        buys_this_month += 1

    if progress: progress.empty()

    # Chiusura forzata fine backtest per calcolo NAV finale
    final_nav = cash
    for t, pos in positions.items():
        try:
            p = market_data[t].iloc[-1]['Close']
        except:
            p = pos['entry_price']
        val = pos['qty'] * p
        final_nav += val

        # Log fittizio per vedere posizione aperta
        gain = val - pos['cost_basis']
        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": t, "Action": "HOLD (End)",
            "Price": p, "Reason": "Portfolio Value",
            "PnL_Net": gain, "Capital": final_nav
        })

    return pd.DataFrame(trade_log), final_nav