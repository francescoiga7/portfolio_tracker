# -*- coding: utf-8 -*-
"""Portafoglio algoritmico live: mette in PRATICA le strategie del backtest.

Lo stato reale (cassa, posizioni, log) è persistito in JSON. `run_daily_update`
replica esattamente la logica del motore di backtest (`run_market_aware_backtest`)
sull'ultimo giorno disponibile, applicandola allo stato REALE del portafoglio
(es. Trade Republic) e producendo le operazioni consigliate (acquisto/vendita).

Nessuna dipendenza da Streamlit: la UI (o lo script cron `scripts/live_daily.py`)
si occupa solo di rendering e stato.
"""
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from etf_metrics.core.automated_backtest import (
    CONFIG,
    STRATEGIES,
    STRATEGY_ORDER,
    BENCH_PROXY_KEY,
    _MarketCtx,
    _assess_market_regime,
    build_benchmark_proxy,
    calculate_advanced_metrics_vectorized,
    check_correlation_strict,
    get_strategy,
    load_recent_market_data,
)
from etf_metrics.shared.config import ALGO_STATE_FILE

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "ai_momentum"


def load_algo_state(filename: str = ALGO_STATE_FILE) -> Dict:
    """Carica lo stato persistito del portafoglio algoritmico."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Impossibile leggere %s: %s", filename, e)
    return _default_state()


def _default_state() -> Dict:
    return {
        "cash": 10000.0, "positions": {}, "trade_log": [], "last_update": None,
        "strategy": DEFAULT_STRATEGY, "commission": None, "max_positions": None,
        "universe_tickers": None, "month_key": None, "buys_this_month": 0,
        "last_buy_scan": None,
    }


def save_algo_state(state: Dict, filename: str = ALGO_STATE_FILE) -> None:
    """Salva lo stato del portafoglio algoritmico su file."""
    try:
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4, default=str)
    except Exception as e:
        logger.warning("Errore salvataggio %s: %s", filename, e)


def get_latest_market_data(tickers: List[str], allow_download: bool = True) -> Dict[str, pd.DataFrame]:
    """Dati 2y per i ticker richiesti (più SPY) con indicatori calcolati.

    DB-first: i ticker già aggiornati oggi si leggono dal DB locale (zero rete);
    solo i mancanti/vecchi vengono scaricati da Yahoo e salvati nel DB.
    Gli indicatori sono calcolati CON il benchmark come riferimento (come nel
    backtest), così le colonne RS sono disponibili alle strategie.
    """
    benchmark = CONFIG['SPY_TICKER']
    # SPY nel DB (anche se non nell'universo): usato come benchmark REALE per
    # regime e RS. Senza SPY si ricade sul proxy equal-weight dell'universo.
    load_list = [t for t in tickers if t != benchmark]
    try:
        from etf_metrics.core.data_manager import MarketDataManager
        if benchmark in MarketDataManager().get_all_tickers():
            load_list.append(benchmark)
    except Exception:
        pass
    raw = load_recent_market_data(load_list, period="2y",
                                  allow_download=allow_download, min_rows=30)
    data: Dict[str, pd.DataFrame] = {}
    spy_df = raw.get(benchmark)
    if spy_df is not None and not spy_df.empty:
        data[benchmark] = calculate_advanced_metrics_vectorized(spy_df)
    else:
        # Senza SPY (es. universo Xetra puro): benchmark di riserva equal-weight
        # dell'universo, così regime e RS restano significativi
        proxy = build_benchmark_proxy(raw)
        if proxy is not None:
            data[BENCH_PROXY_KEY] = calculate_advanced_metrics_vectorized(proxy)
    bench_ref = data.get(benchmark)
    if bench_ref is None:
        bench_ref = data.get(BENCH_PROXY_KEY)
    for t, df in raw.items():
        if t in (benchmark, BENCH_PROXY_KEY):
            continue
        data[t] = calculate_advanced_metrics_vectorized(df, bench_ref)
    return data


def _normalize_positions_dates(state: Dict) -> None:
    """entry_date (stringa JSON) -> pd.Timestamp per l'aritmetica delle strategie."""
    for pos in (state.get('positions') or {}).values():
        ed = pos.get('entry_date')
        if ed is not None and not isinstance(ed, pd.Timestamp):
            try:
                pos['entry_date'] = pd.Timestamp(str(ed))
            except Exception:
                pos['entry_date'] = pd.Timestamp(datetime.now().date())


def _serialize_positions_dates(state: Dict) -> None:
    """entry_date (Timestamp) -> stringa ISO 'YYYY-MM-DD' per il JSON."""
    for pos in (state.get('positions') or {}).values():
        ed = pos.get('entry_date')
        if isinstance(ed, pd.Timestamp):
            pos['entry_date'] = ed.strftime('%Y-%m-%d')


def run_daily_update(state: Dict, tickers_list: List[str], allow_fractional: bool,
                     market_data: Dict[str, pd.DataFrame] = None,
                     strategy: str = None) -> Tuple[Dict, List[str]]:
    """Analisi giornaliera del portafoglio REALE con la strategia scelta.

    Replica il motore del backtest sull'ultimo giorno di dati disponibile:
    - USCITE: stop loss ATR del motore + regole di uscita della strategia
      (exit_signal) + manutenzione posizione (update_position / trailing stop)
    - INGRESSI: filtro di regime (can_buy), scoring della strategia (entry),
      controllo correlazione, sizing per slot, limite acquisti/mese e
      frequenza di ribilanciamento (REBALANCE_DAYS)

    Args:
        state: stato persistito (cassa, posizioni, log, strategia, costi)
        tickers_list: candidati all'acquisto (le posizioni aperte sono sempre monitorate)
        allow_fractional: azioni frazionarie ammesse
        market_data: dati già caricati (opzionale; default: fetch DB-first)
        strategy: chiave strategia (default: quella salvata nello stato)

    Returns:
        (nuovo_stato, messaggi) — i messaggi con 🔴/🟢 sono operazioni da eseguire.
    """
    strat_key = strategy or state.get('strategy') or DEFAULT_STRATEGY
    if strat_key not in STRATEGIES:
        # strategia rimossa (es. dual_momentum/mean_reversion/orderflow cancellate):
        # ricade su quella di default senza crashare
        strat_key = DEFAULT_STRATEGY
    state['strategy'] = strat_key
    strat = get_strategy(strat_key)

    cfg = dict(CONFIG)
    cfg.update(strat.defaults)
    if state.get('commission') is not None:
        cfg['COMMISSION'] = float(state['commission'])
    if state.get('max_positions') is not None:
        cfg['MAX_POSITIONS'] = int(state['max_positions'])
    benchmark = cfg['SPY_TICKER']
    commission = float(cfg['COMMISSION'])

    if market_data is None:
        market_data = get_latest_market_data(
            sorted(set(tickers_list or []) | set((state.get('positions') or {}).keys())))
    # Benchmark effettivo: SPY se disponibile, altrimenti proxy equal-weight
    # dell'universo (universi Xetra senza titoli USA)
    if benchmark not in market_data and BENCH_PROXY_KEY in market_data:
        benchmark = BENCH_PROXY_KEY
    if benchmark not in market_data:
        return state, [f"⚠️ Dati insufficienti (nemmeno il benchmark di riserva): "
                       "aggiorna i dati dalla pagina 📥 Gestione Dati."]

    # Indicatori specifici della strategia (come nel motore di backtest)
    spy_df = market_data.get(benchmark)
    market_data = {t: strat.prepare(df, spy_df, cfg) for t, df in market_data.items()}
    price_matrix = pd.DataFrame({t: d['Close'] for t, d in market_data.items()}).ffill()

    ref = benchmark if benchmark in market_data else list(market_data.keys())[0]
    last_date = market_data[ref].index[-1]

    regime = _assess_market_regime(price_matrix, last_date, market_data, benchmark)
    is_bull = (regime == "BULL")
    is_crash = (regime in ("DANGER", "VOLATILE"))

    ctx = _MarketCtx()
    ctx.cfg = cfg
    ctx.benchmark = benchmark
    ctx.market_data = market_data
    ctx.price_matrix = price_matrix
    ctx.regime = regime
    ctx.is_crash = is_crash
    ctx.current_date = last_date

    messages = [
        f"📅 Data dati: **{last_date.date()}**",
        f"🧬 Strategia: **{strat.name}**",
        f"🌍 Regime: **{regime}**",
    ]

    _normalize_positions_dates(state)
    positions = state['positions']

    # Limite acquisti mensile (azzerato al cambio mese)
    month_key = f"{last_date.year}-{last_date.month:02d}"
    if state.get('month_key') != month_key:
        state['month_key'] = month_key
        state['buys_this_month'] = 0
    buys_this_month = int(state.get('buys_this_month', 0))

    # ------------------------------------------------------------------
    # 1. GESTIONE USCITE (vendite consigliate)
    # ------------------------------------------------------------------
    for t in list(positions.keys()):
        pos = positions.get(t)
        if pos is None:
            continue
        if t not in market_data or last_date not in market_data[t].index:
            continue
        row = market_data[t].loc[last_date]

        action = None
        # 1a. Stop loss "fisico" del motore (se la strategia usa gli stop)
        if strat.use_stops and pos.get('stop_loss') is not None \
                and pd.notna(pos['stop_loss']) and row['Low'] < pos['stop_loss']:
            exit_price = max(row['Open'], pos['stop_loss'])
            action = ("STOP LOSS", 1.0, exit_price)
        else:
            # 1b. Regole di uscita specifiche della strategia
            try:
                sig = strat.exit_signal(pos, row, ctx)
            except Exception:
                sig = None
            if sig is not None:
                reason, portion = sig[0], sig[1]
                price = sig[2] if len(sig) > 2 and sig[2] is not None else row['Close']
                action = (reason, portion, price)
            else:
                # 1c. Manutenzione posizione (trailing stop, flag TP, ecc.)
                try:
                    strat.update_position(pos, row, ctx)
                except Exception:
                    pass

        if action is None:
            continue

        reason, portion, price = action
        qty_sell = pos['qty'] * portion
        if not allow_fractional:
            qty_sell = int(qty_sell)
            if qty_sell == 0 and portion > 0.9:
                qty_sell = pos['qty']
        if qty_sell <= 0:
            continue

        cash_in = (qty_sell * price) - commission
        state['cash'] += cash_in
        pnl = (price - pos['entry_price']) * qty_sell - commission

        if portion >= 0.99 or (pos['qty'] - qty_sell) < (0.001 if allow_fractional else 1):
            del positions[t]
        else:
            pos['qty'] -= qty_sell

        state['trade_log'].insert(0, {
            "Date": last_date.strftime('%Y-%m-%d'), "Ticker": t, "Action": "SELL",
            "Price": float(price), "Reason": reason, "PnL_Net": round(float(pnl), 2),
            "Qty": qty_sell, "Commission": commission, "Strategy": strat.key,
        })
        messages.append(f"🔴 VENDI **{t}**: {qty_sell:g} pz @ {price:.2f} — {reason}")

    # ------------------------------------------------------------------
    # 2. GESTIONE INGRESSI (acquisti consigliati)
    # ------------------------------------------------------------------
    buy_allowed = strat.can_buy(ctx) and \
        (not cfg['ENABLE_MONTHLY_LIMIT'] or buys_this_month < cfg['MAX_BUYS_PER_MONTH'])

    last_scan = state.get('last_buy_scan')
    scan_due = True
    if last_scan:
        try:
            scan_due = (last_date - pd.Timestamp(str(last_scan))).days >= cfg['REBALANCE_DAYS']
        except Exception:
            scan_due = True

    if not strat.can_buy(ctx):
        messages.append(f"ℹ️ Regime **{regime}**: la strategia non acquista in questo contesto.")
    elif not scan_due:
        messages.append(f"ℹ️ Prossima scansione acquisti tra "
                        f"{cfg['REBALANCE_DAYS'] - (last_date - pd.Timestamp(str(last_scan))).days} gg.")
    else:
        state['last_buy_scan'] = last_date.strftime('%Y-%m-%d')
        free_slots = cfg['MAX_POSITIONS'] - len(positions)

        if free_slots <= 0:
            messages.append(f"ℹ️ Portafoglio pieno ({cfg['MAX_POSITIONS']} posizioni).")
        elif state['cash'] <= 50:
            messages.append("ℹ️ Cassa insufficiente per nuovi acquisti.")
        else:
            candidates = []
            for t in tickers_list:
                if t == benchmark or t in positions or t not in market_data:
                    continue
                if last_date not in market_data[t].index:
                    continue
                row = market_data[t].loc[last_date]
                try:
                    score, reason = strat.entry(row, ctx)
                except Exception:
                    continue
                if score is not None and pd.notna(score) and score >= strat.min_entry_score:
                    candidates.append({'t': t, 'score': float(score),
                                       'row': row, 'reason': reason})
            candidates.sort(key=lambda x: x['score'], reverse=True)

            if not candidates:
                messages.append("ℹ️ Nessun candidato con punteggio sufficiente oggi.")

            for cand in candidates:
                if free_slots <= 0 or state['cash'] < 50:
                    break
                if cfg['ENABLE_MONTHLY_LIMIT'] and buys_this_month >= cfg['MAX_BUYS_PER_MONTH']:
                    break

                t, row = cand['t'], cand['row']
                is_safe_corr, _ = check_correlation_strict(
                    t, list(positions.keys()), price_matrix, last_date)
                if not is_safe_corr:
                    continue

                try:
                    risk_factor = strat.position_size_factor(row, ctx)
                except Exception:
                    risk_factor = 1.0

                alloc_per_slot = (state['cash'] / free_slots) * risk_factor * 0.98
                price = row['Close']
                if allow_fractional:
                    qty = round(alloc_per_slot / price, 4)
                else:
                    qty = int(alloc_per_slot / price)
                cost = (qty * price) + commission

                if qty > 0 and cost <= state['cash']:
                    state['cash'] -= cost
                    stop = strat.initial_stop(row, ctx, cfg)
                    positions[t] = {
                        'qty': qty, 'entry_price': float(price),
                        'entry_date': last_date,
                        'stop_loss': float(stop) if (stop is not None and pd.notna(stop)) else None,
                        'tp1_taken': False, 'regime_at_entry': regime,
                    }
                    state['trade_log'].insert(0, {
                        "Date": last_date.strftime('%Y-%m-%d'), "Ticker": t, "Action": "BUY",
                        "Price": float(price), "Reason": cand['reason'], "PnL_Net": 0.0,
                        "Qty": qty, "Commission": commission, "Strategy": strat.key,
                    })
                    buys_this_month += 1
                    state['buys_this_month'] = buys_this_month
                    free_slots -= 1
                    stop_txt = f" (stop iniziale {float(stop):.2f})" if stop is not None else ""
                    messages.append(f"🟢 COMPRA **{t}**: {qty:g} pz @ {price:.2f}{stop_txt} — {cand['reason']}")

    # ------------------------------------------------------------------
    # 3. Riepilogo
    # ------------------------------------------------------------------
    nav = state['cash']
    for t, p in positions.items():
        try:
            nav += p['qty'] * market_data[t]['Close'].iloc[-1]
        except Exception:
            nav += p['qty'] * p['entry_price']
    messages.append(f"💰 Cassa: **{state['cash']:.2f}** | Posizioni aperte: **{len(positions)}** "
                    f"| NAV stimato: **{nav:.2f}**")

    _serialize_positions_dates(state)
    state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return state, messages
