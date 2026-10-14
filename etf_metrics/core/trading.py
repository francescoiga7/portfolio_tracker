# -*- coding: utf-8 -*-
"""Trading live sul portafoglio REALE: algoritmo giornaliero, Satelliti PAC
e stop Chandelier per posizioni satellite.

- `run_daily_update` mette in PRATICA le strategie del backtest: lo stato reale
  (cassa, posizioni, log) e' persistito in JSON e replica la logica del motore
  di backtest (`run_market_aware_backtest`) sull'ultimo giorno disponibile,
  applicandola allo stato REALE del portafoglio (es. Trade Republic) e
  producendo le operazioni consigliate.
- `analyze_pac_portfolio` e' l'analisi consulenza core-satellite (Satelliti
  PAC, read-only) sullo stesso motore.
- `check_satellite_status` e' il trailing stop Chandelier delle posizioni
  satellite: `atr_multiplier_from_vol` e' la scala UNICA condivisa tra logica
  live e backtest (strategia GoldenChandelier) — quello che si valida in
  backtest e' esattamente quello che il portafoglio reale esegue.

Nota sull'import circolare: `core.automated_backtest` importa
  `atr_multiplier_from_vol` da questo modulo; per evitare il ciclo, i nomi di
  `core.automated_backtest` usati qui vengono importati DENTRO le funzioni.

Nessuna dipendenza da Streamlit: la UI si occupa solo di rendering e stato.
"""
import json
import os
import logging
import re
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from etf_metrics.shared.config import ALGO_STATE_FILE
from etf_metrics.clients.yahoo_client import get_series, resolve_isin_details
from etf_metrics.core.metrics import calculate_atr_series
from etf_metrics.core.etf_search_engine import load_etf_universe
from etf_metrics.core.portfolio_tracker import _calculate_open_positions

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "ai_momentum"

# Chiave ISIN: 2 lettere (paese) + 9 alfanumerici + 1 cifra di controllo
_ISIN_RE = re.compile(r'^[A-Z]{2}[A-Z0-9]{9}\d$')


# --- Stop Chandelier per posizioni satellite (scala condivisa col backtest) ---

def _get_latest_vix() -> Optional[float]:
    """Ultimo valore del VIX da Yahoo (None se offline)."""
    try:
        vix_series = get_series("^VIX", period="5d")
        return float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else None
    except Exception:
        return None


def atr_multiplier_from_vol(vol_pct: Optional[float]) -> float:
    """Moltiplicatore ATR del trailing stop in funzione della volatilita'
    annualizzata del mercato (VIX in punti, o proxy = std annua dei rendimenti
    del benchmark in %).

    Un satellite ha bisogno di stop piu' larghi quando il mercato e' nervoso,
    per evitare uscite premature su noise (whipsaw):
      vol < 15 -> 3.0 | < 25 -> 3.5 | < 35 -> 4.5 | panico -> 6.0 | None -> 3.5
    """
    if vol_pct is None or pd.isna(vol_pct):
        return 3.5  # Default senza dato di volatilità
    if vol_pct < 15: return 3.0  # Bassa volatilità: stop stretto ma non soffocante
    if vol_pct < 25: return 3.5  # Volatilità normale: spazio di manovra
    if vol_pct < 35: return 4.5  # Volatilità alta: serve aria
    return 6.0  # Panico (VIX > 35): stop molto largo per evitare whipsaw


def check_satellite_status(df: pd.DataFrame, purchase_date: pd.Timestamp, current_shares: float,
                           market_vol_pct: Optional[float] = None) -> Dict:
    """Stato di una posizione satellite con Trailing Stop Chandelier su Close.

    market_vol_pct: volatilita' annualizzata del mercato in % (proxy del VIX).
    Se None prova a scaricare il VIX da Yahoo. Passando la proxy si replica
    ESATTAMENTE la strategia GoldenChandelier del backtest (stessa scala).
    Ritorna {'action': 'HOLD'|'SELL', 'reason', 'stop_price'}.
    """
    if df is None or df.empty: return {"action": "HOLD", "reason": "Dati non disponibili"}

    df_clean = df.copy()
    if df_clean.index.tz is not None: df_clean.index = df_clean.index.tz_localize(None)

    p_date = purchase_date.tz_localize(None) if purchase_date.tzinfo else purchase_date
    df_since = df_clean[df_clean.index >= p_date]

    if df_since.empty: return {"action": "HOLD", "reason": "Data acquisto futura"}

    atr_series = calculate_atr_series(df_clean)
    if atr_series is None or atr_series.empty:
        return {"action": "HOLD", "reason": "ATR non calcolabile"}

    atr_val = atr_series.iloc[-1]

    # Moltiplicatore dinamico: proxy della volatilita' se fornita, altrimenti VIX da Yahoo
    vol = market_vol_pct if market_vol_pct is not None else _get_latest_vix()
    mult = atr_multiplier_from_vol(vol)

    # Chandelier Exit sul massimo dei CLOSE dall'acquisto (non High:
    # fedele alla logica live e alla strategia GoldenChandelier)
    highest_close_since = df_since['Close'].max()

    stop_price = highest_close_since - (atr_val * mult)

    current_price = df_clean['Close'].iloc[-1]

    dist_pct = ((current_price - stop_price) / current_price) * 100

    if current_price < stop_price:
        return {
            "action": "SELL",
            "stop_price": stop_price,
            "reason": f"Prezzo {current_price:.2f} < Stop Dinamico {stop_price:.2f} (MaxClose: {highest_close_since:.2f})"
        }

    return {
        "action": "HOLD",
        "stop_price": stop_price,
        "reason": f"Trend OK. Stop @ {stop_price:.2f} ({dist_pct:.1f}% di margine)"
    }


# --- Portafoglio algoritmico live ---


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


def get_latest_market_data(tickers: List[str], allow_download: bool = True,
                           progress: Optional[Callable] = None) -> Dict[str, pd.DataFrame]:
    """Dati 2y per i ticker richiesti (più SPY) con indicatori calcolati.

    DB-first: i ticker già aggiornati oggi si leggono dal DB locale (zero rete);
    solo i mancanti/vecchi vengono scaricati da Yahoo e salvati nel DB.
    Gli indicatori sono calcolati CON il benchmark come riferimento (come nel
    backtest), così le colonne RS sono disponibili alle strategie.
    progress(done, total, msg) opzionale per la UI (calcolo indicatori).
    """
    # import locale (core.automated_backtest importa da questo modulo: ciclo)
    from etf_metrics.core.automated_backtest import (
        BENCH_PROXY_KEY, CONFIG, build_benchmark_proxy,
        calculate_advanced_metrics_vectorized, load_recent_market_data)

    benchmark = CONFIG['SPY_TICKER']
    # SPY nel DB (anche se non nell'universo): benchmark REALE per regime e
    # RS. Senza SPY si ricade sul proxy equal-weight dell'universo.
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
        # Senza SPY (es. universo Xetra puro): proxy equal-weight, cosi' regime
        # e RS restano significativi
        proxy = build_benchmark_proxy(raw)
        if proxy is not None:
            data[BENCH_PROXY_KEY] = calculate_advanced_metrics_vectorized(proxy)
    bench_ref = data.get(benchmark)
    if bench_ref is None:
        bench_ref = data.get(BENCH_PROXY_KEY)
    items = [(t, df) for t, df in raw.items() if t not in (benchmark, BENCH_PROXY_KEY)]
    for i, (t, df) in enumerate(items, 1):
        data[t] = calculate_advanced_metrics_vectorized(df, bench_ref)
        if progress:
            try:
                progress(i, len(items), f"Indicatori calcolati: {i}/{len(items)}")
            except Exception:
                pass
    return data


def resolve_portfolio_tickers(keys: List[str],
                             isin_map: Dict[str, str] = None) -> Tuple[Dict[str, str], List[str]]:
    """Chiavi del portfolio (ISIN o ticker) -> ticker Yahoo.

    ISIN: cache isin_map (aggiornata in place) -> ricerca Yahoo (rete, una
    sola volta per ISIN: poi resta nella cache dello stato). Ticker: usati
    direttamente. Ritorna (key->ticker, chiavi non risolte).
    """
    isin_map = isin_map if isin_map is not None else {}
    mapping, unresolved = {}, []
    for key in keys or []:
        if not _ISIN_RE.match(key):
            mapping[key] = key
            continue
        t = isin_map.get(key)
        if not t:
            try:
                t, _name = resolve_isin_details(key)
            except Exception:
                t = None
            if t:
                isin_map[key] = t
        if t:
            mapping[key] = t
        else:
            unresolved.append(key)
    return mapping, unresolved


def portfolio_position_tickers(state: Dict, transactions: List[Dict]):
    """(ticker, mapping, non_risolti) delle posizioni aperte del portfolio.

    Da chiamare PRIMA del caricamento dati per conoscere i ticker da leggere;
    la cache isin_map dello stato evita chiamate ripetute a Yahoo.
    """
    open_pos = _calculate_open_positions(transactions)
    keys = [k for k, d in open_pos.items() if d['quantity'] > 1e-9]
    mapping, unresolved = resolve_portfolio_tickers(keys, state.setdefault('isin_map', {}))
    return sorted(set(mapping.values())), mapping, unresolved


def _reconstruct_from_history(pos: Dict, df, strat, cfg: Dict) -> None:
    """Ricostruisce lo stato strategico di una posizione importata.

    GC-compatibile: max_close = massimo Close dall'ingresso (chandelier
    esatto alla prossima analisi); stop iniziale = Close - ATR×mult del
    giorno di ingresso (il trailing lo alzerà secondo le regole della
    strategia). tp1_taken resta False: conservativo per AI Momentum,
    che potrebbe ri-armare il take-profit parziale.
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return
    entry = pos.get('entry_date')
    if entry is None:
        return
    entry = pd.Timestamp(str(entry))
    hist = df.loc[df.index >= entry]
    if hist.empty:
        hist = df.tail(1)
    pos['max_close'] = float(hist['Close'].max())
    if strat.use_stops:
        row = hist.iloc[0]
        atr = row.get('ATR')
        try:
            atr = float(atr) if atr is not None and pd.notna(atr) else None
        except (TypeError, ValueError):
            atr = None
        if atr and atr > 0:
            pos['stop_loss'] = float(row['Close'] - atr * cfg['STOP_LOSS_ATR_MULT'])


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
    # import locale (core.automated_backtest importa da questo modulo: ciclo)
    from etf_metrics.core.automated_backtest import (
        BENCH_PROXY_KEY, CONFIG, STRATEGIES, _MarketCtx, _assess_market_regime,
        check_correlation_strict, get_strategy)

    strat_key = strategy or state.get('strategy') or DEFAULT_STRATEGY
    if strat_key not in STRATEGIES:
        # strategia ritirata o non nota: ricade sul default senza crashare
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
    if benchmark not in market_data and BENCH_PROXY_KEY in market_data:
        benchmark = BENCH_PROXY_KEY
    if benchmark not in market_data:
        return state, ["⚠️ Dati insufficienti (nemmeno il benchmark di riserva): "
                       "aggiorna i dati dalla pagina 📥 Gestione Dati."]

    # Indicatori specifici della strategia (come nel motore di backtest)
    spy_df = market_data.get(benchmark)
    market_data = {t: strat.prepare(df, spy_df, cfg) for t, df in market_data.items()}
    price_matrix = pd.DataFrame({t: d['Close'] for t, d in market_data.items()}).ffill()

    ref = benchmark if benchmark in market_data else list(market_data.keys())[0]
    last_date = market_data[ref].index[-1]

    regime = _assess_market_regime(price_matrix, last_date, market_data, benchmark)
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

# 1. GESTIONE USCITE (vendite consigliate) ---
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
            # riepilogo delle condizioni di uscita (informativo, non decisionale)
            try:
                pos['exit_outlook'] = strat.exit_outlook(pos, row, ctx)
            except Exception:
                pos['exit_outlook'] = ""
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

# 2. GESTIONE INGRESSI (acquisti consigliati) ---
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
                    try:
                        positions[t]['exit_outlook'] = strat.exit_outlook(positions[t], row, ctx)
                    except Exception:
                        pass
                    stop_txt = f" (stop iniziale {float(stop):.2f})" if stop is not None else ""
                    messages.append(f"🟢 COMPRA **{t}**: {qty:g} pz @ {price:.2f}{stop_txt} — {cand['reason']}")

# 3. Riepilogo ---
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


# --- Satelliti PAC: analisi consulenza core-satellite (READ-ONLY) ---

def pac_universe_tickers() -> List[str]:
    """Universo dei candidati per i satelliti PAC: ETF scoperti con la
    discovery (📥 Gestione Dati → 🌍 Universo ETF) presenti nel DB.

    Se la discovery non è mai stata fatta, ricade su TUTTI i ticker del DB
    (l'analisi resta possibile, l'universo è semplicemente meno mirato).
    """
    from etf_metrics.core.data_manager import MarketDataManager
    try:
        uni = set(load_etf_universe())
    except Exception:
        uni = set()
    try:
        db = set(MarketDataManager().get_all_tickers())
    except Exception:
        db = set()
    if uni:
        return sorted(uni & db)
    return sorted(db)


def _simulate_position_state(pos: Dict, df: pd.DataFrame, strat, ctx) -> None:
    """Ricostruisce lo stato strategico di una posizione dallo storico e ne
    replica la manutenzione giornaliera (trailing stop, flag TP) fino
    all'ultima barra disponibile: lo stesso risultato che avrebbe il motore
    giornaliero, senza persistere nulla. Usato dall'analisi consulenza
    (Satelliti PAC), che lavora su stati effimeri ricostruiti a ogni run.
    """
    _reconstruct_from_history(pos, df, strat, ctx.cfg)
    entry = pos.get('entry_date')
    if entry is None:
        return
    if not isinstance(entry, pd.Timestamp):
        entry = pd.Timestamp(str(entry))
    hist = df.loc[df.index >= entry]
    for _, row in hist.iterrows():
        try:
            strat.update_position(pos, row, ctx)
        except Exception:
            break


def analyze_pac_portfolio(state: Dict, transactions: List[Dict],
                          strategy: str = None,
                          market_data: Dict[str, pd.DataFrame] = None,
                          core_tickers: List[str] = None,
                          pac_amount: float = None) -> Tuple[Dict, List[str]]:
    """Analisi consulenza core-satellite sul portfolio REALE (READ-ONLY).

    Legge le posizioni aperte dalle transazioni (saved_portfolio) e:
    - USCITE: regole di uscita della strategia sulle posizioni NON core
      (stato strategico ricostruito dallo storico, mai persistito)
    - INGRESSO: miglior candidato dell'universo ETF per il PAC del mese
      (top-1 per punteggio, controllo correlazione con le posizioni detenute,
      guard mensile sulle transazioni satellite)

    Non modifica né lo stato né il portfolio: ritorna (report, messaggi).
    Il chiamante può salvare lo stato solo per persistere la cache isin_map.
    """
    # import locale (core.automated_backtest importa da questo modulo: ciclo)
    from etf_metrics.core.automated_backtest import (
        BENCH_PROXY_KEY, CONFIG, STRATEGIES, _MarketCtx, _assess_market_regime,
        check_correlation_strict, get_strategy)

    strat_key = strategy or state.get('strategy') or DEFAULT_STRATEGY
    if strat_key not in STRATEGIES:
        strat_key = DEFAULT_STRATEGY
    strat = get_strategy(strat_key)
    cfg = dict(CONFIG)
    cfg.update(strat.defaults)
    benchmark = cfg['SPY_TICKER']

    report = {
        'strategy': strat.name, 'strategy_key': strat.key,
        'positions': [], 'core': [], 'unresolved': [], 'no_data': [],
        'sell_count': 0, 'buy': None, 'buy_status': None, 'buy_note': '',
        'candidates_seen': 0, 'regime': None, 'date': None,
    }
    messages: List[str] = []

    open_pos = _calculate_open_positions(transactions)
    keys = [k for k, d in open_pos.items() if d['quantity'] > 1e-9]
    mapping, unresolved = resolve_portfolio_tickers(keys, state.setdefault('isin_map', {}))
    report['unresolved'] = unresolved
    if unresolved:
        messages.append(f"⚠️ ISIN non risolti (posizioni ignorate): {', '.join(unresolved)}")

    core = {str(t).strip().upper() for t in (core_tickers or []) if t and str(t).strip()}
    held = {mapping[k] for k in keys if k in mapping}
    sat_keys = [k for k in keys if mapping.get(k) not in core]

    universe = pac_universe_tickers()
    if market_data is None:
        market_data = get_latest_market_data(
            sorted(held | set(universe)), allow_download=False)

    if benchmark not in market_data and BENCH_PROXY_KEY in market_data:
        benchmark = BENCH_PROXY_KEY
    if benchmark not in market_data:
        report['buy_status'] = 'no_data'
        report['buy_note'] = ("Dati insufficienti (nemmeno il benchmark di riserva): "
                              "aggiorna i dati dalla pagina 📥 Gestione Dati.")
        messages.append("⚠️ " + report['buy_note'])
        return report, messages

    spy_df = market_data.get(benchmark)
    market_data = {t: strat.prepare(df, spy_df, cfg) for t, df in market_data.items()}
    price_matrix = pd.DataFrame({t: d['Close'] for t, d in market_data.items()}).ffill()

    ref = benchmark if benchmark in market_data else list(market_data.keys())[0]
    last_date = market_data[ref].index[-1]

    regime = _assess_market_regime(price_matrix, last_date, market_data, benchmark)
    ctx = _MarketCtx()
    ctx.cfg = cfg
    ctx.benchmark = benchmark
    ctx.market_data = market_data
    ctx.price_matrix = price_matrix
    ctx.regime = regime
    ctx.is_crash = regime in ("DANGER", "VOLATILE")
    ctx.current_date = last_date

    report['regime'] = regime
    report['date'] = last_date
    messages.extend([
        f"📅 Data dati: **{last_date.date()}**",
        f"🧬 Strategia: **{strat.name}**",
        f"🌍 Regime: **{regime}**",
    ])
    if (pd.Timestamp.now().normalize() - last_date).days > 5:
        messages.append(f"⚠️ Dati datati (ultima barra {last_date.date()}): "
                        "aggiorna il DB per un consiglio affidabile.")

    from etf_metrics.core.data_manager import MarketDataManager
    try:
        names = MarketDataManager().get_ticker_names(sorted(held | set(universe)))
    except Exception:
        names = {}

# 1. USCITE: regole della strategia sulle posizioni non-core ---
    for key in sorted(sat_keys):
        t = mapping.get(key)
        if t is None:
            continue
        d = open_pos[key]
        qty = float(d['quantity'])
        avg_cost = float(d['total_cost']) / qty
        entry = {
            'ticker': t, 'name': names.get(t) or t, 'qty': qty,
            'entry_price': avg_cost, 'entry_date': d['first_buy_date'],
            'satellite': bool(d.get('satellite')),
        }
        if t not in market_data or last_date not in market_data[t].index:
            report['no_data'].append(t)
            entry.update({'verdict': 'HOLD', 'action': None, 'outlook': '',
                          'last_price': None, 'pnl_pct': None, 'stop': None,
                          'note': 'dati non disponibili nel DB'})
            report['positions'].append(entry)
            messages.append(f"⚠️ {t}: dati non disponibili nel DB (posizione non analizzata).")
            continue

        # stato strategico effimero, ricostruito dallo storico. Se risulta già
        # una vendita parziale dopo l'ingresso, si assume che il take-profit
        # parziale (TP1) sia stato eseguito: senza questo accorgimento il
        # consiglio "vendi il 25%" si ripeterebbe a ogni analisi.
        first_buy = pd.Timestamp(d['first_buy_date'])
        has_sell_after_entry = any(
            tr.get('type') == 'sell' and tr.get('isin') == key
            and pd.to_datetime(str(tr.get('date', '')), errors='coerce') >= first_buy
            for tr in transactions)
        pos = {
            'qty': qty, 'entry_price': avg_cost, 'entry_date': d['first_buy_date'],
            'stop_loss': None, 'tp1_taken': bool(has_sell_after_entry),
            'regime_at_entry': None, 'max_close': None,
        }
        _simulate_position_state(pos, market_data[t], strat, ctx)

        row = market_data[t].loc[last_date]
        last_price = float(row['Close'])
        pnl_pct = (last_price / avg_cost) - 1.0

        action = None
        # 1a. stop loss "fisico" del motore
        if strat.use_stops and pos.get('stop_loss') is not None \
                and pd.notna(pos['stop_loss']) and row['Low'] < pos['stop_loss']:
            action = ("STOP LOSS", 1.0, max(row['Open'], pos['stop_loss']))
        else:
            # 1b. regole di uscita specifiche della strategia
            try:
                sig = strat.exit_signal(pos, row, ctx)
            except Exception:
                sig = None
            if sig is not None:
                reason, portion = sig[0], sig[1]
                price = sig[2] if len(sig) > 2 and sig[2] is not None else row['Close']
                action = (reason, portion, price)

        if action is None:
            try:
                outlook = strat.exit_outlook(pos, row, ctx)
            except Exception:
                outlook = ""
            entry.update({'verdict': 'HOLD', 'action': None, 'outlook': outlook,
                          'last_price': last_price, 'pnl_pct': pnl_pct,
                          'stop': pos.get('stop_loss'), 'note': ''})
        else:
            reason, portion, price = action
            qty_sell = qty * portion
            entry.update({'verdict': 'SELL', 'action': {
                'reason': reason, 'portion': portion,
                'price': float(price), 'qty': qty_sell},
                'outlook': '', 'last_price': last_price, 'pnl_pct': pnl_pct,
                'stop': pos.get('stop_loss'), 'note': ''})
            report['sell_count'] += 1
            messages.append(f"🔴 VENDI **{t}**: {qty_sell:g} pz @ {price:.2f} — {reason}")
        report['positions'].append(entry)

    # posizioni core: mostrate ma mai analizzate
    for key in sorted(keys):
        t = mapping.get(key)
        if t is None or t not in core:
            continue
        d = open_pos[key]
        last_price = None
        if t in market_data and not market_data[t].empty:
            try:
                last_price = float(market_data[t]['Close'].iloc[-1])
            except Exception:
                last_price = None
        report['core'].append({
            'ticker': t, 'name': names.get(t) or t,
            'qty': float(d['quantity']), 'last_price': last_price,
        })

# 2. INGRESSO: miglior candidato per il PAC del mese (top-1) ---
    month_key = pd.Timestamp.now().strftime('%Y-%m')
    sat_buys = [tr for tr in transactions
                if tr.get('type') == 'buy' and tr.get('satellite')
                and str(tr.get('date', ''))[:7] == month_key]

    if sat_buys:
        report['buy_status'] = 'already_bought'
        report['buy_note'] = ", ".join(
            f"{tr.get('isin')} ({str(tr.get('date'))[:10]})" for tr in sat_buys)
        messages.append(f"ℹ️ PAC di questo mese già eseguito: {report['buy_note']}.")
    elif not strat.can_buy(ctx):
        report['buy_status'] = 'no_regime'
        report['buy_note'] = f"regime {regime}: la strategia non acquista"
        messages.append(f"ℹ️ Regime **{regime}**: nessun satellite questo mese → "
                        "destina il PAC al core (buy & hold).")
    else:
        candidates = []
        for t in universe:
            if t == benchmark or t in held or t in core or t not in market_data:
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
                                   'reason': reason, 'price': float(row['Close'])})
        report['candidates_seen'] = len(candidates)
        candidates.sort(key=lambda x: x['score'], reverse=True)

        pick = None
        for cand in candidates[:50]:
            is_safe, _ = check_correlation_strict(
                cand['t'], sorted(held), price_matrix, last_date)
            if is_safe:
                pick = cand
                break

        if pick is not None:
            buy = {
                'ticker': pick['t'], 'name': names.get(pick['t']) or pick['t'],
                'score': pick['score'], 'reason': pick['reason'],
                'price': pick['price'],
            }
            if pac_amount and pick['price'] > 0:
                buy['qty'] = round(float(pac_amount) / pick['price'], 4)
                buy['amount'] = float(pac_amount)
            report['buy'] = buy
            report['buy_status'] = 'ok'
            messages.append(f"🟢 COMPRA **{pick['t']}** @ {pick['price']:.2f} — "
                            f"{pick['reason']} (score {pick['score']:.0f})")
        else:
            report['buy_status'] = 'no_candidate'
            report['buy_note'] = (
                f"{len(candidates)} candidati sopra soglia, tutti già detenuti "
                "o troppo correlati" if candidates
                else "nessun candidato con punteggio sufficiente")
            messages.append("ℹ️ Nessun candidato utilizzabile questo mese → "
                            "destina il PAC al core (buy & hold).")

    if not report['sell_count'] and report['positions']:
        messages.append("✅ Nessuna vendita da eseguire sulle posizioni satellite.")
    return report, messages
