# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import logging
from datetime import timedelta, datetime
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.data_manager import MarketDataManager

logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE STOCK (Swing Trading Classico) ---
CONFIG = {
    'MAX_POSITIONS': 3,  # Slot totali del portafoglio misto
    'REBALANCE_DAYS': 5,
    'ENABLE_MONTHLY_LIMIT': True,
    'MAX_BUYS_PER_MONTH': 4,
    'COMMISSION': 2.0,
    'SPY_TICKER': 'SPY',  # Benchmark Azionario
    'MACRO_TREND_FILTER': True,
    'BEAR_VOLATILITY_THRESHOLD': 3.0,
    'MIN_RS_SCORE': 80,
    'MIN_ADX': 25,
    'MIN_PROX_HIGH': 0.85,
    'STOP_LOSS_ATR_MULT': 2.5,
    'TIME_STOP_DAYS': 21,  # Time Stop attivo per azioni (efficienza capitale)
    'TP1_PCT': 0.12,  # Take Profit fisso 15%
    'MAX_CORRELATION': 0.65
}

# --- CONFIGURAZIONE CRYPTO (Halving Cycle Strategy) ---
CRYPTO_BENCHMARK = 'BTC-USD'

HALVING_DATES = [
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
    pd.Timestamp("2028-04-15")
]


def is_crypto_asset(ticker):
    """Distingue tra Stock e Crypto."""
    t = ticker.upper()
    # Suffissi comuni o lista esplicita di coin principali
    if t.endswith("-USD") or t.endswith("-EUR"): return True
    if t in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "MSTR", "COIN", "MARA"]: return True
    # Nota: MSTR/COIN/MARA si comportano come crypto leva, meglio trattarle come tali.
    return False


def get_crypto_cycle_phase(current_date):
    """
    Determina la fase del ciclo Bitcoin basata sull'Halving.
    Questa è la logica 'Macro' che piaceva: detta il ritmo.
    """
    last_halving = None
    for h in HALVING_DATES:
        if current_date >= h:
            last_halving = h
        else:
            break

    if not last_halving: return "ACCUMULATION"

    days_since = (current_date - last_halving).days

    # Fasi del ciclo (approssimazione storica)
    if 0 <= days_since <= 550:
        return "BULL_RUN"  # La fase parabolica post-halving
    elif 550 < days_since <= 1000:
        return "BEAR_WINTER"  # Il crollo e il lungo inverno
    else:
        return "ACCUMULATION"  # La fase laterale pre-halving


def get_crypto_config(phase):
    """
    Restituisce i parametri operativi specifici per la fase del ciclo.
    """
    cfg = CONFIG.copy()  # Base di partenza

    # Override specifici Crypto
    cfg['MAX_CORRELATION'] = 0.92  # Crypto si muovono insieme, tolleriamo di più

    if phase == "BULL_RUN":
        # Modalità: AGGRESSIVA
        cfg['STOP_LOSS_ATR_MULT'] = 4.5  # Stop molto larghi per evitare flash crash
        cfg['MIN_ADX'] = 25  # Solo trend forti
        cfg['TP1_PCT'] = 10.0  # NO TP PRECOCE: Lasciamo correre (Moonbag logic)
        cfg['TIME_STOP_DAYS'] = 999  # Disattiviamo Time Stop: HODL
        cfg['ENABLE_MONTHLY_LIMIT'] = False  # Compriamo ogni dip possibile

    elif phase == "BEAR_WINTER":
        # Modalità: DIFENSIVA
        cfg['STOP_LOSS_ATR_MULT'] = 2.0  # Stop strettissimi
        cfg['ENABLE_MONTHLY_LIMIT'] = True
        cfg['MAX_BUYS_PER_MONTH'] = 1  # Quasi fermi

    elif phase == "ACCUMULATION":
        # Modalità: SWING TRADING
        cfg['STOP_LOSS_ATR_MULT'] = 3.5
        cfg['TP1_PCT'] = 0.30  # Prendiamo profitto sui pump (+30%)
        cfg['TIME_STOP_DAYS'] = 90  # Pazienza media

    return cfg


def calculate_advanced_metrics_vectorized(df, benchmark_df=None):
    if 'Close' not in df.columns: return df
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()

    # CORREZIONE APPLICATA QUI
    df['Vol_20'] = df['Close'].ffill().pct_change(fill_method=None).rolling(20).std() * 100

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
    denom = plus_di14 + minus_di14
    dx = 100 * np.abs(plus_di14 - minus_di14) / denom.replace(0, 1)
    df['ADX'] = dx.rolling(14).mean()

    df['High_52w'] = df['Close'].rolling(252).max()
    df['Prox_High'] = df['Close'] / df['High_52w']

    # RS Score vs Benchmark specifico (passato come argomento)
    df['RS_Score'] = 50.0
    if benchmark_df is not None:
        common = df.index.intersection(benchmark_df.index)
        if len(common) > 65:
            # CORREZIONE PREVENTIVA APPLICATA ANCHE QUI
            stock_ret = df['Close'].ffill().pct_change(63, fill_method=None)
            bench_ret = benchmark_df['Close'].ffill().pct_change(63, fill_method=None)
            rel_perf = (stock_ret.loc[common] - bench_ret.loc[common]) * 100
            df.loc[common, 'RS_Score'] = rel_perf

    return df


def _assess_market_regime(current_date, market_data, benchmark_ticker, is_crypto_context):
    """
    Valuta il regime (BULL/BEAR) specifico per l'asset class.
    """
    if benchmark_ticker not in market_data or current_date not in market_data[benchmark_ticker].index:
        return "NEUTRAL"

    bench_row = market_data[benchmark_ticker].loc[current_date]

    if is_crypto_context:
        # --- LOGICA CRYPTO HALVING ---
        cycle_phase = get_crypto_cycle_phase(current_date)
        is_above_sma200 = bench_row['Close'] > bench_row['SMA200']

        if cycle_phase == "BEAR_WINTER":
            # In inverno cripto, permettiamo trade SOLO se BTC mostra forza estrema (Mini rally)
            if bench_row['Close'] > bench_row['SMA50']: return "NEUTRAL"
            return "BEAR"  # Altrimenti blocco totale

        if cycle_phase == "BULL_RUN":
            # In Bull Run ignoriamo l'ipercomprato, compriamo i dip aggressivamente
            return "BULL"

        # Accumulation Phase
        if is_above_sma200: return "BULL"
        return "NEUTRAL"

    else:
        # --- LOGICA STOCK CLASSICA ---
        if bench_row['Close'] < bench_row['SMA200']: return "BEAR"

        # Volatilità eccessiva su SPY?
        atr_pct = (bench_row['ATR'] / bench_row['Close']) * 100
        if atr_pct > CONFIG['BEAR_VOLATILITY_THRESHOLD']: return "VOLATILE"

        if bench_row['RSI'] < 30: return "DANGER"
        return "BULL"


def calculate_stock_score(row):
    """Score per Azioni"""
    if row['Close'] < row['SMA50']: return 0, "Below SMA50"
    if row['Prox_High'] < CONFIG['MIN_PROX_HIGH']: return 0, "Too far from Highs"
    if row['ADX'] < CONFIG['MIN_ADX']: return 0, "Weak Trend"

    score = 50
    score += (row['RS_Score'] * 2)
    score += (row['ADX'] / 2)
    if row['Prox_High'] > 0.95: score += 10
    if row['Vol_20'] > 5.0: score -= 15  # Penalizza volatilità su stock

    return max(0, score), f"Stock Score: {score:.0f} | RS: {row['RS_Score']:.1f}"


def calculate_crypto_score(row, phase):
    """Score per Crypto (Influenzato dal Ciclo)"""
    score = 50
    reason = []

    # Trend (SMA 50 è cruciale)
    if row['Close'] > row['SMA50']:
        score += 20
    else:
        if phase == "BULL_RUN": score -= 20  # Grave in bull run

    # Momentum
    if row['RSI'] > 55: score += 15
    if row['ADX'] > 25: score += 15

    # Relative Strength vs BTC
    if row['RS_Score'] > 0:
        score += 25
        reason.append("Alpha vs BTC")

    # Cycle Specifics
    if phase == "ACCUMULATION" and row['RSI'] < 40:
        score += 30  # Buy the dip accumulation
        reason.append("Accumulation Dip")

    if phase == "BULL_RUN" and row['Prox_High'] > 0.9:
        score += 25  # Breakout
        reason.append("Near ATH")

    return max(0, score), f"Crypto Score: {score:.0f} | {', '.join(reason)}"


# Wrapper per compatibilità UI (Scanner live)
def calculate_ai_smart_score(row):
    # Di default usa la logica stock se chiamato dall'esterno senza contesto
    return calculate_stock_score(row)


def prepare_market_data(tickers, period="10y"):
    db_manager = MarketDataManager()

    spy_ticker = CONFIG['SPY_TICKER']
    btc_ticker = CRYPTO_BENCHMARK

    # Lista unica di tutto ciò che serve
    all_tickers = list(set(tickers + [spy_ticker, btc_ticker]))

    # Download mancanti
    missing = db_manager.get_tickers_needing_update(all_tickers)
    if missing:
        try:
            new_data = {t: get_series(t, period=period, as_dataframe=True) for t in missing}
            new_data = {k: v for k, v in new_data.items() if v is not None and not v.empty}
            if new_data: db_manager.save_bulk_data(new_data)
        except Exception as e:
            logger.error(f"Error bulk download: {e}")

    loaded = db_manager.load_data(all_tickers)

    # Prepara Benchmarks
    bench_stock_df = loaded.get(spy_ticker)
    bench_crypto_df = loaded.get(btc_ticker)

    if bench_stock_df is not None:
        bench_stock_df = calculate_advanced_metrics_vectorized(bench_stock_df, None)
        loaded[spy_ticker] = bench_stock_df

    if bench_crypto_df is not None:
        bench_crypto_df = calculate_advanced_metrics_vectorized(bench_crypto_df, None)
        loaded[btc_ticker] = bench_crypto_df

    processed = {}
    for t, df in loaded.items():
        if not df.empty and len(df) > 50:
            # Decide quale benchmark usare per il calcolo RS_Score di questo asset
            is_c = is_crypto_asset(t)
            ref_bench = bench_crypto_df if is_c else bench_stock_df

            processed[t] = calculate_advanced_metrics_vectorized(df, ref_bench)

    return processed, bench_stock_df, bench_crypto_df


def run_market_aware_backtest(tickers: list, start_date="2015-01-01", initial_capital=10000,
                              preloaded_data=None, tax_rate=26.0, allow_fractional=True,
                              is_crypto=False):
    # NOTA: `is_crypto` qui è ignorato, usiamo la rilevazione automatica asset per asset.

    spy_ticker = CONFIG['SPY_TICKER']
    btc_ticker = CRYPTO_BENCHMARK

    # 1. Preparazione Dati Ibridi
    if preloaded_data:
        market_data = preloaded_data
        bench_stock_df = market_data.get(spy_ticker)
        bench_crypto_df = market_data.get(btc_ticker)
    else:
        # MODIFICA: Imposta period="max" per avere dati dal 2015 o prima
        market_data, bench_stock_df, bench_crypto_df = prepare_market_data(tickers)

    if not market_data: return pd.DataFrame(), initial_capital

    # Timeline principale (SPY è più stabile per le date)
    ref = spy_ticker if spy_ticker in market_data else list(market_data.keys())[0]
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

    def _get_current_nav(curr_cash, curr_positions, curr_date):
        equity = curr_cash
        for p_ticker, p_data in curr_positions.items():
            try:
                curr_p = market_data[p_ticker].loc[curr_date]['Close']
            except:
                curr_p = p_data['entry_price']
            equity += p_data['qty'] * curr_p
        return equity

    # --- MAIN LOOP ---
    for i, current_date in enumerate(sim_dates):
        if progress and i % 50 == 0: progress.progress((i + 1) / len(sim_dates))

        if current_date.month != current_sim_month:
            current_sim_month = current_date.month
            buys_this_month = 0

        # A. Analisi Macro (Doppio Binario)
        # 1. Regime Stock (SPY)
        regime_stock = _assess_market_regime(current_date, market_data, spy_ticker, False)
        is_bull_stock = (regime_stock == "BULL")

        # 2. Regime Crypto (BTC + Cicli Halving)
        cycle_phase_crypto = get_crypto_cycle_phase(current_date)
        regime_crypto = _assess_market_regime(current_date, market_data, btc_ticker, True)
        is_bull_crypto = (regime_crypto == "BULL")

        # Configurazione attiva per Crypto in questo momento (quella Stock è statica)
        active_crypto_config = get_crypto_config(cycle_phase_crypto)

        tokens_to_sell = []

        # --- GESTIONE USCITE ---
        for t, pos in positions.items():
            if t not in market_data or current_date not in market_data[t].index: continue

            is_crypto_pos = is_crypto_asset(t)
            # Seleziona Configurazione corretta per questo asset
            cfg = active_crypto_config if is_crypto_pos else CONFIG

            row = market_data[t].loc[current_date]
            curr_price = row['Close']
            atr = row['ATR']

            # 1. STOP LOSS
            if row['Low'] < pos['stop_loss']:
                exit_price = max(row['Open'], pos['stop_loss'])
                tokens_to_sell.append((t, exit_price, "STOP LOSS", 1.0))
                continue

            roi = (curr_price / pos['entry_price']) - 1

            # 2. TAKE PROFIT (Logica Divergente)
            if is_crypto_pos and cycle_phase_crypto == "BULL_RUN":
                # MOONBAG: Vendi 25% a +100%, lascia correre il resto per l'Halving Pump
                if not pos.get('moonbag_secured', False) and roi >= 1.0:
                    tokens_to_sell.append((t, curr_price, "TP 100% (Moonbag)", 0.25))
                    positions[t]['moonbag_secured'] = True
                    positions[t]['stop_loss'] = pos['entry_price'] * 1.5  # Stop abbondantemente in profit
                    continue
            else:
                # STOCK o Crypto Bear/Accumulation: Take Profit Fisso
                if not pos.get('tp1_taken', False) and roi >= cfg['TP1_PCT']:
                    tokens_to_sell.append((t, curr_price, "TP1 (Lock)", 0.33))  # Vendi 1/3
                    positions[t]['stop_loss'] = pos['entry_price'] * 1.02  # Break-even
                    positions[t]['tp1_taken'] = True
                    continue

            # 3. TRAILING STOP
            if pos.get('tp1_taken', False) or roi > 0.05:
                new_stop = curr_price - (atr * cfg['STOP_LOSS_ATR_MULT'])
                if new_stop > positions[t]['stop_loss']:
                    positions[t]['stop_loss'] = new_stop

            # 4. TIME STOP
            # Disattivato (999 giorni) per Crypto in Bull Run, attivo per Stock (21gg)
            days_held = (current_date - pos['entry_date']).days
            if days_held >= cfg['TIME_STOP_DAYS'] and roi < 0.02:
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

            if portion >= 0.99 or (pos['qty'] - qty_sell) < (0.001 if allow_fractional else 1):
                del positions[t]
            else:
                positions[t]['qty'] -= qty_sell
                positions[t]['cost_basis'] -= cost_portion

            current_nav = _get_current_nav(cash, positions, current_date)
            trade_log.append({
                "Date": current_date.date(), "Ticker": t, "Action": "SELL",
                "Price": price, "Reason": reason, "PnL_Net": gain - tax,
                "Capital": cash, "Total_Equity": current_nav, "Qty": qty_sell
            })

        # --- GESTIONE INGRESSI (Ibrida) ---
        if i % CONFIG['REBALANCE_DAYS'] == 0:
            free_slots = CONFIG['MAX_POSITIONS'] - len(positions)

            if free_slots > 0 and cash > 50:
                candidates = []
                for t in tickers:
                    if t in [spy_ticker, btc_ticker] and len(tickers) > 2: continue
                    if t in positions: continue
                    if t not in market_data or current_date not in market_data[t].index: continue

                    # Riconoscimento Asset Class
                    is_crypto_t = is_crypto_asset(t)

                    # CHECK REGIME:
                    # Se è crypto, controlla regime crypto. Se stock, controlla regime stock.
                    if is_crypto_t:
                        # Extra check: Bear Winter -> No buy a meno che non sia inversione confermata
                        if cycle_phase_crypto == "BEAR_WINTER": continue
                        if not is_bull_crypto: continue
                    else:
                        if not is_bull_stock: continue

                        # Monthly Limit Check
                    cfg = active_crypto_config if is_crypto_t else CONFIG
                    if cfg['ENABLE_MONTHLY_LIMIT'] and buys_this_month >= cfg['MAX_BUYS_PER_MONTH']:
                        continue

                    row = market_data[t].loc[current_date]

                    # Scoring Specifico
                    if is_crypto_t:
                        score, reason = calculate_crypto_score(row, cycle_phase_crypto)
                        min_score = 55 if cycle_phase_crypto == "BULL_RUN" else 75
                    else:
                        score, reason = calculate_stock_score(row)
                        min_score = 65

                    if score >= min_score:
                        candidates.append(
                            {'t': t, 'score': score, 'row': row, 'reason': reason, 'is_crypto': is_crypto_t})

                candidates.sort(key=lambda x: x['score'], reverse=True)

                for cand in candidates:
                    if free_slots <= 0 or cash < 50: break

                    t = cand['t']
                    row = cand['row']
                    is_c = cand['is_crypto']
                    cfg = active_crypto_config if is_c else CONFIG

                    # Controllo correlazione
                    is_safe_corr, conflict = check_correlation_strict(
                        t, list(positions.keys()), price_matrix, current_date,
                        threshold=cfg['MAX_CORRELATION']
                    )
                    if not is_safe_corr: continue

                    # Sizing
                    alloc_per_slot = (cash / free_slots) * 0.98

                    if allow_fractional:
                        qty = alloc_per_slot / row['Close']
                    else:
                        qty = int(alloc_per_slot / row['Close'])

                    cost = (qty * row['Close']) + CONFIG['COMMISSION']

                    if qty > 0 and cost <= cash:
                        cash -= cost
                        initial_stop = row['Close'] - (row['ATR'] * cfg['STOP_LOSS_ATR_MULT'])

                        positions[t] = {
                            'qty': qty, 'cost_basis': cost, 'entry_price': row['Close'],
                            'entry_date': current_date, 'stop_loss': initial_stop,
                            'tp1_taken': False, 'moonbag_secured': False
                        }

                        current_nav = _get_current_nav(cash, positions, current_date)
                        trade_log.append({
                            "Date": current_date.date(), "Ticker": t, "Action": "BUY",
                            "Price": row['Close'], "Reason": cand['reason'],
                            "PnL_Net": 0.0, "Capital": cash, "Total_Equity": current_nav, "Qty": qty
                        })
                        free_slots -= 1
                        buys_this_month += 1

    if progress: progress.empty()

    # Mark to Market Finale
    final_nav = cash
    for t, pos in positions.items():
        try:
            p = market_data[t].iloc[-1]['Close']
        except:
            p = pos['entry_price']
        val = pos['qty'] * p
        final_nav += val

        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": t, "Action": "HOLD (End)",
            "Price": p, "Reason": "Portfolio Value",
            "PnL_Net": val - pos['cost_basis'],
            "Capital": cash, "Total_Equity": final_nav, "Qty": pos['qty']
        })

    return pd.DataFrame(trade_log), final_nav


# Helper per correlazione
def check_correlation_strict(candidate, portfolio, price_matrix, current_date, threshold):
    if not portfolio: return True, None
    if candidate not in price_matrix.columns: return True, None
    start = current_date - timedelta(days=60)
    hist = price_matrix.loc[start:current_date]
    if len(hist) < 20: return True, None
    cand_series = hist[candidate]
    for p in portfolio:
        if p not in hist.columns: continue
        corr = cand_series.corr(hist[p])
        if corr > threshold: return False, p
    return True, None