# -*- coding: utf-8 -*-
"""Backtest mensile dei Satelliti PAC: replay storico del consiglio mensile.

Ogni mese versa `monthly_amount` e compra il miglior ETF dell'universo
secondo la strategia: la scansione dei candidati è GIORNALIERA finché il
versamento del mese non viene investito (max 1 satellite al mese, come
"chiedere all'algoritmo quando arrivano i soldi"); vince il top-1 per
punteggio (soglia min_entry_score) tra i candidati non correlati con le
posizioni detenute e ammessi dal filtro di regime (can_buy). Le posizioni
sono gestite GIORNO PER GIORNO con le regole di uscita della strategia
(stop ATR, TP1 parziale, uscite momentum/regime), esattamente come nel
motore di backtest e nell'analisi live di `analyze_pac_portfolio`:
questo modulo ne è il replay storico.

I mesi senza segnale (regime sfavorevole o nessun candidato) destinano
l'importo al CORE buy & hold (core_fallback=True, come il consiglio live)
oppure lo lasciano in cassa.

Benchmark di riferimento: PAC puro sullo stesso core (tutti i mesi al core,
nessuna vendita), calcolato nello stesso loop.

Nessun lookahead: gli indicatori sono causali (rolling/shift) e ogni
decisione usa la riga del ticker alla data corrente — ciò che l'algoritmo
live avrebbe visto quel giorno.
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from etf_metrics.core.automated_backtest import (
    CONFIG, STRATEGIES, STRATEGY_ORDER, get_strategy,
    _ensure_processed, prepare_market_data, _prepare_strategy_frames,
    _precompute_regimes, _assess_market_regime, check_correlation_strict,
    _FastRowPanel, _MarketCtx, BENCH_PROXY_KEY)


def etf_universe_in_db() -> List[str]:
    """Universo ETF (discovery) ∩ DB, SENZA fallback: solo ETF/ETC/ETP.

    Diversamente da `pac_universe_tickers` (pagina Satelliti PAC) non ricade
    mai su tutti i ticker del DB: il backtest PAC lavora solo sugli ETF.
    """
    from etf_metrics.core.etf_search_engine import load_etf_universe
    from etf_metrics.core.data_manager import MarketDataManager
    try:
        uni = set(load_etf_universe())
    except Exception:
        uni = set()
    if not uni:
        return []
    try:
        db = set(MarketDataManager().get_all_tickers())
    except Exception:
        db = set()
    return sorted(uni & db)


def _yearly_returns(ret: pd.Series) -> pd.Series:
    """Rendimento per anno civile, composto dai rendimenti giornalieri
    corretti per i versamenti (time-weighted: immune ai flussi di cassa)."""
    out = {}
    for year, grp in ret.groupby(ret.index.year):
        out[year] = float((1.0 + grp).prod() - 1.0)
    return pd.Series(out)


def run_pac_backtest(
    strategy_key: str,
    tickers: List[str],
    monthly_amount: float = 200.0,
    start_date: str = "2019-01-01",
    commission: float = 1.0,
    core_ticker: str = "VWCE.MI",
    core_fallback: bool = True,
    preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Optional[Dict]:
    """Backtest PAC satelliti per UNA strategia. Ritorna None se i dati non
    bastano. Vedi il docstring del modulo per il modello."""
    strat = get_strategy(strategy_key)
    cfg = dict(CONFIG)
    cfg.update(strat.defaults)
    cfg['COMMISSION'] = float(commission)
    benchmark = cfg['SPY_TICKER']

    if preloaded_data is not None:
        market_data, _bk = _ensure_processed(
            {t: df for t, df in preloaded_data.items()
             if df is not None and not df.empty}, benchmark)
    else:
        market_data = prepare_market_data(
            list(tickers) + [core_ticker], period="max", allow_download=False)
    if not market_data:
        return None
    if benchmark not in market_data and BENCH_PROXY_KEY in market_data:
        benchmark = BENCH_PROXY_KEY
    if benchmark not in market_data:
        return None

    spy_df = market_data.get(benchmark)
    market_data = _prepare_strategy_frames(strat, market_data, spy_df, cfg)

    core_available = core_ticker in market_data
    ref = benchmark if benchmark in market_data else list(market_data.keys())[0]
    start_ts = pd.to_datetime(start_date)
    sim_dates = market_data[ref].index[market_data[ref].index >= start_ts]
    n_sim = len(sim_dates)
    if n_sim == 0:
        return None

    price_matrix = pd.DataFrame(
        {t: d['Close'] for t, d in market_data.items()}).ffill()
    regime_series = _precompute_regimes(market_data.get(benchmark), sim_dates)
    panel = _FastRowPanel(market_data, sim_dates)
    universe = [t for t in tickers
                if t in market_data and t not in (benchmark, core_ticker)]

    ctx = _MarketCtx()
    ctx.cfg = cfg
    ctx.benchmark = benchmark
    ctx.market_data = market_data
    ctx.price_matrix = price_matrix

    cash = 0.0
    deposited = 0.0
    commissions_paid = 0.0
    positions: Dict[str, Dict] = {}
    core_qty = 0.0            # core comprato nei mesi senza segnale (fallback)
    core_pure_qty = 0.0       # benchmark: PAC puro sul core, ogni mese
    trades: List[Dict] = []
    monthly: List[Dict] = []
    nav_values = np.full(n_sim, np.nan)
    core_nav_values = np.full(n_sim, np.nan)
    deposit_values = np.zeros(n_sim)
    current_month = None
    pending = 0.0             # versamento del mese non ancora investito
    pending_month = None      # chiave 'YYYY-MM' del versamento in sospeso
    n_buys = n_sells = 0
    months_total = months_no_signal = months_core = 0

    def _sell(t: str, pos: Dict, qty: float, price: float, reason: str,
              date: pd.Timestamp):
        nonlocal cash, n_sells, commissions_paid
        qty = min(qty, pos['qty'])
        if qty <= 1e-9:
            return
        pnl = (price - pos['entry_price']) * qty
        cash += qty * price - commission
        commissions_paid += commission
        n_sells += 1
        pos['qty'] -= qty
        trades.append({
            'data': date.date(), 'tipo': 'VENDITA', 'ticker': t,
            'qty': round(qty, 4), 'prezzo': round(price, 4),
            'motivo': reason, 'P&L (€)': round(pnl, 2),
            'cash dopo (€)': round(cash, 2),
        })
        if pos['qty'] <= 1e-9:
            del positions[t]

    def _buy_core(amount: float, i: int, date: pd.Timestamp, why: str):
        nonlocal core_qty, commissions_paid, cash
        cprice = panel.close_ff(core_ticker, i)
        if cprice is None or cprice <= 0 or amount <= commission:
            return False
        q = (amount - commission) / cprice
        core_qty += q
        cash -= q * cprice + commission
        commissions_paid += commission
        trades.append({
            'data': date.date(), 'tipo': 'ACQUISTO CORE', 'ticker': core_ticker,
            'qty': round(q, 4), 'prezzo': round(cprice, 4),
            'motivo': why, 'P&L (€)': 0.0,
            'cash dopo (€)': round(cash, 2),
        })
        return True

    for i, current_date in enumerate(sim_dates):
        ctx.current_date = current_date
        if regime_series is not None:
            ctx.regime = regime_series.iloc[i]
        else:
            ctx.regime = _assess_market_regime(
                price_matrix, current_date, market_data, benchmark)
        ctx.is_crash = ctx.regime in ("DANGER", "VOLATILE")

        m = (current_date.year, current_date.month)
        new_month = (m != current_month)
        if new_month:
            # il mese precedente non ha dato segnali: il versamento in sospeso
            # va al core (come il consiglio live) oppure resta in cassa
            if pending > 0:
                months_no_signal += 1
                action = {
                    'mese': pending_month, 'azione': '⛔ nessun segnale nel mese',
                    'ticker': '', 'score': '', 'motivo': '', 'qty': '', 'prezzo': '',
                }
                if core_fallback and core_available \
                        and _buy_core(pending, i, current_date,
                                      "mese senza segnale → core"):
                    months_core += 1
                    action['azione'] += " → 🛡️ core"
                monthly.append(action)
                pending = 0.0
            current_month = m
            months_total += 1
            cash += monthly_amount
            deposited += monthly_amount
            deposit_values[i] += monthly_amount
            pending = monthly_amount
            pending_month = f"{current_date.year}-{current_date.month:02d}"
            # benchmark: PAC puro sul core, ogni mese, senza vendite
            if core_available:
                cprice = panel.close_ff(core_ticker, i)
                if cprice and cprice > 0 and monthly_amount > commission:
                    core_pure_qty += (monthly_amount - commission) / cprice

        # --- USCITE giornaliere sulle posizioni satellite ---
        for t in sorted(list(positions.keys())):
            pos = positions[t]
            row = panel.row(t, i)
            if row is None:
                continue
            stop = pos.get('stop_loss')
            if stop is not None and pd.notna(stop) and row['Low'] < stop:
                _sell(t, pos, pos['qty'], float(stop), "STOP LOSS", current_date)
                continue
            try:
                sig = strat.exit_signal(pos, row, ctx)
            except Exception:
                sig = None
            if sig:
                reason, portion, price = sig
                _sell(t, pos, pos['qty'] * float(portion), float(price),
                      str(reason), current_date)
            else:
                try:
                    strat.update_position(pos, row, ctx)
                except Exception:
                    pass

        # --- ACQUISTO: scansione giornaliera finché il versamento del mese
        #     non viene investito (max 1 satellite al mese, come il consiglio
        #     live "chiedi all'algoritmo quando hai i soldi") ---
        if pending > 0 and strat.can_buy(ctx):
            candidates = []
            for t in universe:
                if t in positions:
                    continue
                row = panel.row(t, i)
                if row is None:
                    continue
                try:
                    score, reason = strat.entry(row, ctx)
                except Exception:
                    continue
                if score is not None and pd.notna(score) \
                        and score >= strat.min_entry_score:
                    candidates.append({'t': t, 'score': float(score),
                                       'reason': reason, 'row': row})
            candidates.sort(key=lambda x: x['score'], reverse=True)
            pick = None
            blocked = 0
            for cand in candidates[:50]:
                is_safe, _ = check_correlation_strict(
                    cand['t'], sorted(positions.keys()),
                    price_matrix, current_date)
                if is_safe:
                    pick = cand
                    break
                blocked += 1
            if pick is not None:
                row = pick['row']
                price = float(row['Close'])
                investable = pending - commission
                if investable > 0 and price > 0:
                    qty = investable / price
                    stop = strat.initial_stop(row, ctx, cfg)
                    positions[pick['t']] = {
                        'qty': qty, 'entry_price': price,
                        'entry_date': current_date,
                        'stop_loss': (float(stop)
                                      if stop is not None and pd.notna(stop)
                                      else None),
                        'max_close': price, 'tp1_taken': False,
                        'regime_at_entry': ctx.regime,
                        'entry_score': pick['score'],
                    }
                    cash -= qty * price + commission
                    commissions_paid += commission
                    n_buys += 1
                    pending = 0.0
                    monthly.append({
                        'mese': pending_month, 'azione': '🛰️ acquistato',
                        'ticker': pick['t'], 'score': round(pick['score'], 1),
                        'motivo': pick['reason'],
                        'qty': round(qty, 4), 'prezzo': round(price, 2),
                    })

        # --- NAV mark-to-market ---
        cprice = panel.close_ff(core_ticker, i) if core_available else None
        nav = cash + (core_qty * cprice if cprice else 0.0)
        for t, pos in positions.items():
            cp = panel.close_ff(t, i)
            nav += pos['qty'] * (cp if cp is not None else pos['entry_price'])
        nav_values[i] = nav
        core_nav_values[i] = (core_pure_qty * cprice) if cprice else np.nan

        if progress_callback is not None and (i % max(1, n_sim // 50) == 0):
            try:
                progress_callback(i / n_sim)
            except Exception:
                pass

    # ultimo mese: risolvi il versamento ancora in sospeso
    if pending > 0:
        months_no_signal += 1
        action = {
            'mese': pending_month, 'azione': '⛔ nessun segnale nel mese',
            'ticker': '', 'score': '', 'motivo': '', 'qty': '', 'prezzo': '',
        }
        if core_fallback and core_available \
                and _buy_core(pending, n_sim - 1, sim_dates[-1],
                              "mese senza segnale → core"):
            months_core += 1
            action['azione'] += " → 🛡️ core"
        monthly.append(action)
        pending = 0.0

    nav_series = pd.Series(nav_values, index=sim_dates).dropna()
    core_nav = pd.Series(core_nav_values, index=sim_dates).dropna()
    deposits = pd.Series(np.cumsum(deposit_values), index=sim_dates)

    # colonne numeriche omogenee (i mesi senza azione hanno campi vuoti:
    # serve dtype coerente per la serializzazione arrow/st.dataframe)
    monthly_df = pd.DataFrame(monthly)
    if not monthly_df.empty:
        for col in ('score', 'qty', 'prezzo'):
            monthly_df[col] = pd.to_numeric(monthly_df[col], errors='coerce')

    # Rendimenti giornalieri corretti per i versamenti (il deposito entra a
    # inizio giornata, prima del movimento di mercato)
    pos_in_sim = {d: j for j, d in enumerate(sim_dates)}
    dep_arr = deposit_values[[pos_in_sim[d] for d in nav_series.index]]
    vals = nav_series.values
    prev = np.concatenate([[np.nan], vals[:-1]])
    ok = prev > 0
    adj = np.where(ok, (vals - dep_arr) / np.where(ok, prev, np.nan) - 1.0, np.nan)
    ret = pd.Series(adj, index=nav_series.index)
    ret = ret[ret.notna() & np.isfinite(ret)]
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) \
        if len(ret) > 1 and ret.std() > 0 else None

    dd = nav_series / nav_series.cummax() - 1.0
    final = float(nav_series.iloc[-1]) if len(nav_series) else 0.0
    core_final = float(core_nav.iloc[-1]) if len(core_nav) else None

    return {
        'strategy': strat.name, 'strategy_key': strategy_key,
        'final': final, 'invested': float(deposited),
        'net': final - deposited,
        'multiple': (final / deposited) if deposited > 0 else None,
        'max_dd': float(dd.min()) if len(dd) else None,
        'sharpe': sharpe,
        'n_buys': n_buys, 'n_sells': n_sells,
        'months_total': months_total, 'months_no_signal': months_no_signal,
        'months_core': months_core,
        'commissions': float(commissions_paid),
        'nav_daily': nav_series, 'core_nav': core_nav, 'deposits': deposits,
        'trades': pd.DataFrame(trades),
        'monthly': monthly_df,
        'yearly': _yearly_returns(ret),
        'core_available': core_available, 'core_final': core_final,
        'universe_size': len(universe),
    }


def run_pac_comparison(
    tickers: List[str],
    strategies: Optional[List[str]] = None,
    monthly_amount: float = 200.0,
    start_date: str = "2019-01-01",
    commission: float = 1.0,
    core_ticker: str = "VWCE.MI",
    core_fallback: bool = True,
    preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, Optional[Dict]]:
    """Confronto PAC su più strategie, stessi dati e stessi versamenti.

    Ritorna {strategy_key: risultato_di_run_pac_backtest} (valore None per
    le strategie senza dati sufficienti).
    """
    keys = [k for k in (strategies or STRATEGY_ORDER) if k in STRATEGIES]
    results: Dict[str, Optional[Dict]] = {}
    for idx, key in enumerate(keys):
        results[key] = run_pac_backtest(
            key, tickers, monthly_amount=monthly_amount, start_date=start_date,
            commission=commission, core_ticker=core_ticker,
            core_fallback=core_fallback, preloaded_data=preloaded_data)
        if progress_callback is not None:
            try:
                progress_callback((idx + 1) / len(keys))
            except Exception:
                pass
    return results
