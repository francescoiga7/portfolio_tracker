# -*- coding: utf-8 -*-
"""Pagina UI: 💹 Backtest (strategie algoritmiche).

Tre modalità:
- 🤖 Automated Backtest: simulazione storica di una strategia singola
- 🏁 Confronto Strategie: tutte le strategie sugli stessi dati (confronto fair)
- 🧭 Confronto Strategie PAC: replay storico del consiglio mensile dei
  Satelliti PAC (rendering in `ui/pac_backtest.py`, come in core)

I risultati delle simulazioni vengono conservati in `st.session_state`: cambiare
parametro, tab o selezione NON rilancia la simulazione — i risultati spariscono
solo quando una nuova esecuzione li sostituisce (con avviso se i parametri
correnti differiscono da quelli dell'ultima run).
"""
import streamlit as st
import pandas as pd
import numpy as np

try:  # richiesto da Styler.background_gradient (tabelle colorate)
    import matplotlib  # noqa: F401
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from etf_metrics.clients.yahoo_client import get_failed_tickers
from etf_metrics.core.automated_backtest import (
    run_market_aware_backtest,
    run_strategy_comparison,
    prepare_market_data,
    compute_buy_and_hold_curves,
    BENCH_PROXY_KEY,
    CONFIG,
    STRATEGIES,
    STRATEGY_ORDER,
)
from etf_metrics.ui.components import (
    render_universe_selector,
    render_page_help,
    list_db_tickers,
    UNIVERSE_WHOLE_DB,
)
from etf_metrics.ui.pac_backtest import render_pac_backtest_ui
from etf_metrics.core.metrics import (compute_backtest_performance_metrics,
                                      compute_metrics_from_series,
                                      yearly_returns)


DEFAULT_TRADING_LIST = """NVDA
TSLA
AMD
COIN
MARA
PLTR
META
AMZN
NFLX
QQQ
TQQQ
SQQQ"""

# Chiavi di session_state per i risultati delle due modalità.
_BT_RUN_KEY = 'trading_backtest_run'    # simulazione singola
_CMP_RUN_KEY = 'trading_comparison_run'  # confronto strategie


# 1. CARICAMENTO DATI (cache di sessione) ---

@st.cache_data(show_spinner=False)
def _load_real_data(tickers_tuple):
    """Carica i dati dal DB Parquet locale (nessun accesso a Yahoo)."""
    return prepare_market_data(list(tickers_tuple), allow_download=False)


def _load_real_data_safe(tickers):
    """Wrapper anti-cache-dei-fallimenti: un caricamento vuoto non viene
    memorizzato, così il retry (es. dopo aver aggiornato il DB) funziona."""
    res = _load_real_data(tuple(tickers))
    if not res or not any(t in res for t in tickers):
        _load_real_data.clear()
    return res


def _describe_real_load(market_data, tickers):
    """Caption diagnostica: cosa è stato caricato e perché manca il resto."""
    found = [t for t in tickers if t in market_data]
    missing = [t for t in tickers if t not in market_data]
    if not missing:
        return f"🗄️ Dati reali: {len(found)}/{len(tickers)} ticker caricati."

    try:
        blacklisted = get_failed_tickers()
    except Exception:
        blacklisted = set()
    dead = [t for t in missing if t in blacklisted]
    other = [t for t in missing if t not in blacklisted]

    parts = [f"🗄️ Dati reali: {len(found)}/{len(tickers)} ticker caricati."]
    if dead:
        parts.append(f"⏭️ In cache negativa 'no data found' (non vengono ritentati): {', '.join(dead)}.")
    if other:
        parts.append(f"⚠️ Non nel DB locale (aggiornali da 📥 Gestione Dati): {', '.join(other)}.")
    return " ".join(parts)


def _offer_spy_benchmark(market_data):
    """Se il benchmark di riserva è il proxy dell'universo (SPY assente dal DB)
    lo segnala e offre il download immediato di SPY.

    Il benchmark di confronto DEVE essere l'S&P500: il proxy equal-weight
    dell'universo è solo un ripiego offline (regime e RS riferiti ai titoli
    stessi dell'universo, non a un indice di mercato).
    """
    bench = CONFIG['SPY_TICKER']
    if bench in market_data or BENCH_PROXY_KEY not in market_data:
        return
    with st.expander("🏁 Benchmark: S&P500 non disponibile (uso il proxy dell'universo)"):
        st.markdown(
            "**SPY non è nel DB locale**, quindi il benchmark e il filtro macro di "
            "regime usano un indice equal-weight dei tuoi ticker (etichetta "
            "\"Universo (proxy)\"). Per confrontarti con il **vero S&P500** scarica "
            f"{bench} dalla pagina 📥 **Gestione Dati**: resterà nel DB e verrà usato "
            "automaticamente (anche per regime e RS).")


def _load_real_data_for_backtest(tickers, universe_mode, start_date_backtest):
    """Carica i dati reali per il backtest dal DB Parquet locale.

    Ritorna (market_data, start_date_str, sim_tickers) oppure None se l'universo è
    vuoto o nessun ticker è utilizzabile (in tal caso l'errore è già stato mostrato).
    """
    if not tickers:
        if universe_mode == UNIVERSE_WHOLE_DB:
            st.error("Il DB locale non contiene ticker con storico sufficiente: scarica i "
                     "dati dalla pagina 📥 Gestione Dati e riprova.")
        else:
            st.warning("Inserisci almeno un ticker nella watchlist.")
        return None

    with st.spinner("Caricamento dati da parquet"):
        market_data = _load_real_data_safe(tickers)

    if not any(t in market_data for t in tickers):
        if universe_mode == UNIVERSE_WHOLE_DB:
            st.error("Nessun ticker del DB locale è utilizzabile (storico insufficiente?). "
                     "Aggiorna i dati dalla pagina 📥 Gestione Dati e riprova.")
        else:
            avail = list_db_tickers()
            hint = ("\n\nTicker con storico sufficiente nel DB: **"
                    + ", ".join(avail[:40]) + "**") if avail else ""
            st.error("Nessuno dei ticker richiesti è presente nel DB locale. Aggiorna i "
                     "dati dalla pagina 📥 Gestione Dati o scegli ticker presenti nel DB."
                     + hint)
        return None

    st.caption(_describe_real_load(market_data, tickers))
    _offer_spy_benchmark(market_data)
    return market_data, str(start_date_backtest), tickers


# 2. KPI E METRICHE DI SUPPORTO ---

def _bench_curve_and_label(bh_curves):
    """(serie_benchmark, etichetta): S&P500 se presente, proxy universo altrimenti."""
    if not bh_curves:
        return None, None
    if 'benchmark' in bh_curves:
        return bh_curves['benchmark'], "S&P500"
    if 'benchmark_proxy' in bh_curves:
        return bh_curves['benchmark_proxy'], "Universo (proxy)"
    return None, None


def _build_strategy_params(commission, max_positions):
    return {'COMMISSION': float(commission), 'MAX_POSITIONS': int(max_positions)}


def _bench_kpis(curve, initial_capital):
    """CAGR/MaxDD/Sharpe per le righe B&H della classifica, con la STESSA
    convenzione delle strategie: (CAGR - rf 3%) / vol daily annualizzata
    (vedi compute_backtest_performance_metrics in metrics.py)."""
    m = compute_metrics_from_series(curve)
    if not m:
        return np.nan, np.nan, np.nan
    cagr, vol, mdd = m.get('cagr'), m.get('vol_ann'), m.get('mdd')
    sharpe = (cagr / 100.0 - 0.03) / (vol / 100.0) if (cagr is not None and vol) else np.nan
    return (cagr if cagr is not None else np.nan,
            mdd if mdd is not None else np.nan, sharpe)


def _fragility_kpis(df_trades, diag):
    """Metriche di fragilita': se il profitto poggia su poche
    vendite o su una posizione sola, il risultato e' path-dependence, non edge.

    - Top-5 % utili: quota del profitto lordo portata dalle 5 migliori vendite
    - Peso max pos.: massimo peso mai raggiunto da UNA posizione sul NAV
    - Max ADV %: massima quota del turnover giornaliero medio usata da un
      ingresso (il gate la tiene <= MAX_POSITION_VS_ADV: ~20%)"""
    if df_trades is None or df_trades.empty:
        return np.nan, np.nan, np.nan
    sells = df_trades[df_trades['Action'] == 'SELL']
    gains = sells.loc[sells['PnL_Net'] > 0, 'PnL_Net'] if not sells.empty else pd.Series(dtype=float)
    gross = float(gains.sum()) if len(gains) else 0.0
    top5 = float(gains.nlargest(5).sum()) if len(gains) else 0.0
    top5_pct = (top5 / gross * 100.0) if gross > 0 else np.nan
    mpw = (diag or {}).get('max_position_weight')
    max_pos_pct = (mpw * 100.0) if mpw is not None else np.nan
    max_adv = df_trades['ADV_Pct'].max() if 'ADV_Pct' in df_trades.columns else np.nan
    return top5_pct, max_pos_pct, max_adv


def _strategy_equity_curve(r, initial_capital):
    """Curva equity di una strategia: NAV giornaliero se disponibile,
    ricostruzione dal log dei trade altrimenti."""
    m = compute_backtest_performance_metrics(r["trades"], initial_capital,
                                             nav_daily=r.get("nav"))
    if m and m.get('Equity_Curve') is not None:
        return m['Equity_Curve']
    if r["trades"] is not None and not r["trades"].empty:
        return r["trades"].drop_duplicates(
            subset=['Date'], keep='last').set_index('Date')['Capital']
    return None


def _multi_start_cagr_table(results, selected, sim_tickers, market_data, params,
                            initial_capital, start_date, progress=None,
                            allow_fractional=True):
    """Ri-esegue le strategie da 3 date di inizio aggiuntive (+1/+2/+3 anni)
    e ritorna {nome strategia: {data inizio: CAGR %}}.

    Misure di path-dependence: stesso motore, stessi dati, cambia SOLO la
    data di partenza. Se il CAGR minimo e' molto sotto quello della run
    principale, il risultato e' fortuna di percorso, non edge."""
    navs = [r["nav"] for r in results.values()
            if r.get("nav") is not None and len(r["nav"]) > 1]
    if not navs:
        return {}
    end_ts = navs[0].index[-1]
    starts = [pd.Timestamp(start_date)]
    for k in (1, 2, 3):
        s = pd.Timestamp(start_date) + pd.DateOffset(years=k)
        if s < end_ts - pd.DateOffset(years=1):
            starts.append(s)
    if len(starts) < 2:
        return {}

    out = {}
    for key, r in results.items():
        m = compute_backtest_performance_metrics(r["trades"], initial_capital,
                                                 nav_daily=r.get("nav"))
        if m:
            out.setdefault(r["name"], {})[str(starts[0].date())] = m["CAGR"] * 100.0

    for j, s in enumerate(starts[1:]):
        res_j = run_strategy_comparison(
            sim_tickers, strategies=selected, start_date=str(s.date()),
            initial_capital=initial_capital, preloaded_data=market_data,
            allow_fractional=allow_fractional, strategy_params=params)
        years_j = max((end_ts - s).days / 365.25, 1e-9)
        for key, r in res_j.items():
            out.setdefault(r["name"], {})[str(s.date())] = \
                ((r["final"] / initial_capital) ** (1.0 / years_j) - 1.0) * 100.0
        if progress:
            try:
                progress((j + 1) / len(starts[1:]))
            except Exception:
                pass
    return out


# 3. PERSISTENZA DELLE RUN (st.session_state) ---

def _run_signature(**fields):
    """Fingerprint dei parametri di una run: se un parametro cambia, i risultati
    in cache vengono marcati 'vecchi' invece di sparire silenziosamente."""
    return tuple(sorted((k, repr(v)) for k, v in fields.items()))


def _store_run(key, sig, payload):
    payload['sig'] = sig
    st.session_state[key] = payload


def _load_run(key, sig):
    """Ritorna il payload dell'ultima run (se esiste). Se i parametri correnti
    differiscono da quelli della run, lo segnala: i risultati restano visibili
    finché non viene eseguita una nuova simulazione."""
    run = st.session_state.get(key)
    if run is None:
        return None
    if run.get('sig') != sig:
        st.caption("ℹ️ Stai vedendo i risultati dell'**ultima esecuzione**: alcuni parametri "
                   "sono cambiati da allora. Premi di nuovo ▶️ per rieseguire con i valori "
                   "attuali.")
    return run


# 4. REPORT DELLA SINGOLA STRATEGIA ---

def _style_backtest_rows(row):
    act = row['Action']
    if act == 'BUY':
        return ['background-color: #202020'] * len(row)
    if act == 'SELL':
        if "TREND BREAK" in str(row['Reason']):
            return ['background-color: #3d0000'] * len(row)
        if row['PnL_Net'] > 0:
            return ['background-color: #003300'] * len(row)
        return ['background-color: #3d0000'] * len(row)
    return [''] * len(row)


def _render_kpi_header(final_cap, initial_capital, strategy_name, metrics, diag=None):
    st.markdown("### 📊 Performance Report")
    st.caption(f"Strategia: **{strategy_name}**")
    if diag and diag.get('liquidity_blocked'):
        st.caption(
            f"🚧 Gate di liquidità: {diag['liquidity_blocked']} segnali d'ingresso scartati "
            "(prezzo/volume/ADV insufficienti per un'esecuzione realistica).")

    total_return_pct = ((final_cap - initial_capital) / initial_capital) * 100
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Capitale Finale", f"€{final_cap:,.0f}")
    kpi2.metric("Rendimento Totale", f"{total_return_pct:+.2f}%",
                delta_color="normal" if total_return_pct > 0 else "inverse")
    if metrics:
        kpi3.metric("CAGR (Annuo)", f"{metrics['CAGR'] * 100:.2f}%")
        kpi4.metric("Max Drawdown", f"{metrics['Max_Drawdown'] * 100:.2f}%",
                    help="Massima perdita dal picco. Più è basso (vicino a 0), meglio è.",
                    delta_color="inverse")


def _render_cost_impact(df_trades, initial_capital, metrics):
    """Focus 'poche operazioni': quanto pesano le commissioni."""
    comm_total = float(df_trades['Commission'].sum()) if ('Commission' in df_trades.columns) else 0.0
    n_buys = int((df_trades['Action'] == 'BUY').sum())
    n_sells = int((df_trades['Action'] == 'SELL').sum())
    years_sim = 1.0
    if metrics and metrics.get('Equity_Curve') is not None and len(metrics['Equity_Curve']) > 252:
        years_sim = len(metrics['Equity_Curve']) / 252.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Commissioni Totali", f"€{comm_total:,.0f}",
              help="Somma delle commissioni pagate su acquisti e vendite.")
    c2.metric("🔁 Operazioni (BUY/SELL)", f"{n_buys} / {n_sells}")
    c3.metric("📉 Acquisti per Anno", f"{n_buys / years_sim:.1f}",
              help="Meno acquisti all'anno = meno commissioni. Obiettivo: < 15/anno.")
    c4.metric("⚖️ Commissioni / Capitale", f"{(comm_total / max(initial_capital, 1)) * 100:.1f}%",
              help="Impatto delle commissioni sul capitale iniziale.")


def _render_risk_stats(metrics):
    if not metrics:
        return
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}",
                  help="> 1.0 Buono, > 2.0 Ottimo. Misura il rendimento per unità di rischio.")
    col_r2.metric("Sortino Ratio", f"{metrics['Sortino']:.2f}",
                  help="Simile allo Sharpe, ma considera solo la volatilità negativa (i crolli).")
    col_r3.metric("Profit Factor", f"{metrics['Profit_Factor']:.2f}",
                  help="Rapporto tra vincite lorde e perdite lorde. > 1.5 è solido.")
    col_r4.metric("Volatilità", f"{metrics['Volatility'] * 100:.1f}%")


def _render_trade_stats(df_trades):
    realized = df_trades[df_trades['Action'] == 'SELL']
    if realized.empty:
        return
    wins = len(realized[realized['PnL_Net'] > 0])
    losses = len(realized[realized['PnL_Net'] <= 0])
    total = len(realized)
    win_rate = (wins / total) * 100

    st.markdown(f"""
    **Statistiche Operative (Significatività):**
    - Totale Operazioni: **{total}**
    - ✅ Vincite: **{wins}**
    - ❌ Perdite: **{losses}**
    - 🎯 Win Rate: **{win_rate:.1f}%**
    """)
    if total < 30:
        st.warning("⚠️ Attenzione: Numero di operazioni basso (<30). I risultati statistici "
                   "potrebbero non essere affidabili.")


def _render_benchmark_beat(final_cap, initial_capital, bh_curves):
    if not bh_curves:
        return
    bench, bench_lbl = _bench_curve_and_label(bh_curves)
    if bench is None or len(bench) <= 1:
        return
    bench_final = bench.iloc[-1]
    alpha = final_cap - bench_final
    st.markdown(
        f"**🏁 Obiettivo 'battere {bench_lbl}':** Benchmark B&H = **€{bench_final:,.0f}** → "
        f"{'🟢 strategia VINCE' if final_cap > bench_final else '🔴 strategia perde'} "
        f"di **€{abs(alpha):,.0f}** ({(alpha / initial_capital) * 100:+.1f}% sul capitale iniziale)."
    )


def _render_trade_log(df_trades):
    st.markdown("### 📋 Lista Operazioni")
    fmt = {"Price": "{:.2f}", "PnL_Net": "{:+.2f}", "Capital": "€{:,.0f}",
           "Total_Equity": "€{:,.0f}", "Qty": "{:.4f}"}
    if 'Commission' in df_trades.columns:
        fmt["Commission"] = "€{:.2f}"
    if 'ADV_EUR' in df_trades.columns:
        fmt["ADV_EUR"] = "€{:,.0f}"
        fmt["ADV_Pct"] = "{:.1f}%"
    df_display = df_trades.style.apply(_style_backtest_rows, axis=1).format(fmt)
    st.dataframe(df_display, width="stretch")


def _render_strategy_report(df_trades, final_cap, initial_capital, strategy_name,
                            bh_curves=None, nav_daily=None, diag=None):
    """Report: KPI, impatto commissioni, rischio, statistiche trade e log."""
    metrics = compute_backtest_performance_metrics(df_trades, initial_capital,
                                                   nav_daily=nav_daily)

    _render_kpi_header(final_cap, initial_capital, strategy_name, metrics, diag)
    _render_cost_impact(df_trades, initial_capital, metrics)
    st.divider()
    _render_risk_stats(metrics)
    _render_trade_stats(df_trades)
    _render_benchmark_beat(final_cap, initial_capital, bh_curves)
    st.divider()
    _render_trade_log(df_trades)


# 5. CONFRONTO STRATEGIE (classifica, robustezza, log operazioni) ---

def _comparison_summary_rows(results, bh_curves, initial_capital, commission):
    """Righe della classifica: strategie + benchmark B&H, con metriche di
    fragilita' per capire su cosa poggia il rendimento."""
    rows = []
    for key, r in results.items():
        df_trades, final = r["trades"], r["final"]
        m = compute_backtest_performance_metrics(df_trades, initial_capital,
                                                 nav_daily=r.get("nav"))
        n_sells = int((df_trades['Action'] == 'SELL').sum()) if not df_trades.empty else 0
        comm = float(df_trades['Commission'].sum()) if ('Commission' in df_trades.columns) else 0.0
        top5_pct, max_pos_pct, max_adv = _fragility_kpis(df_trades, r.get("diag"))
        rows.append({
            "Strategia": r["name"],
            "Finale (€)": final,
            "Rend. Tot. %": ((final - initial_capital) / initial_capital) * 100,
            "CAGR %": (m['CAGR'] * 100) if m else np.nan,
            "Max DD %": (m['Max_Drawdown'] * 100) if m else np.nan,
            "Sharpe": m['Sharpe'] if m else np.nan,
            "Profit Factor": m['Profit_Factor'] if m else np.nan,
            "Vendite": n_sells,
            "Commissioni (€)": comm,
            "Top-5 % utili": top5_pct,
            "Peso max pos. %": max_pos_pct,
            "Max ADV %": max_adv,
        })

    bench_s, bench_name = _bench_curve_and_label(bh_curves)
    if bench_s is not None:
        bench_final = float(bench_s.iloc[-1])
        b_cagr, b_mdd, b_sharpe = _bench_kpis(bench_s, initial_capital)
        rows.append({
            "Strategia": f"🏁 {bench_name} Buy & Hold",
            "Finale (€)": bench_final,
            "Rend. Tot. %": ((bench_final - initial_capital) / initial_capital) * 100,
            "CAGR %": b_cagr, "Max DD %": b_mdd, "Sharpe": b_sharpe,
            "Profit Factor": np.nan, "Vendite": 1,
            "Commissioni (€)": float(commission),
            "Top-5 % utili": np.nan, "Peso max pos. %": np.nan, "Max ADV %": np.nan,
        })
    return rows


def _render_comparison_leaderboard(rows):
    df_summary = pd.DataFrame(rows).set_index("Strategia").sort_values("Finale (€)", ascending=False)
    st.markdown("### 🏆 Classifica (per capitale finale)")
    st.caption(
        "ℹ️ *Top-5 % utili*: quota del profitto lordo nelle 5 migliori vendite (alta = "
        "risultato concentrato in poche operazioni). *Peso max pos.*: massimo peso di "
        "una posizione sul NAV. *Max ADV*: massima quota del turnover giornaliero medio "
        "usata da un ingresso (gate di liquidità: ≤ 20%).")
    st.dataframe(
        df_summary.style.format({
            "Finale (€)": "€{:,.0f}", "Rend. Tot. %": "{:+.1f}",
            "CAGR %": "{:.1f}", "Max DD %": "{:.1f}", "Sharpe": "{:.2f}",
            "Profit Factor": "{:.2f}", "Vendite": "{:.0f}", "Commissioni (€)": "€{:,.0f}",
            "Top-5 % utili": "{:.0f}", "Peso max pos. %": "{:.0f}", "Max ADV %": "{:.0f}",
        }),
        width="stretch",
    )
    return df_summary


def _render_comparison_callouts(rows):
    bench_final = next((r["Finale (€)"] for r in rows
                        if str(r["Strategia"]).startswith("🏁")), None)
    valid = [r for r in rows if not str(r["Strategia"]).startswith("🏁")]
    if not valid:
        return
    best = max(valid, key=lambda r: r["Finale (€)"])
    beats = best["Finale (€)"] > (bench_final if bench_final is not None else -np.inf)
    st.success(
        f"🥇 Migliore: **{best['Strategia']}** con €{best['Finale (€)']:,.0f} "
        f"({best['Rend. Tot. %']:+.1f}%) — "
        + (f"**batte il benchmark** (€{bench_final:,.0f})." if beats
           else "sotto il benchmark in questo scenario.")
    )
    cheapest = min(valid, key=lambda r: r["Commissioni (€)"])
    st.caption(
        f"💸 Meno commissioni: **{cheapest['Strategia']}** "
        f"(€{cheapest['Commissioni (€)']:,.0f} totali, {cheapest['Vendite']:.0f} vendite)."
    )


def _bg_gradient(styler, **kwargs):
    """Colora la tabella se matplotlib e' disponibile (altrimenti resta leggibile,
    solo senza sfumature: la pagina non deve crashare per una dipendenza visiva)."""
    return styler.background_gradient(**kwargs) if HAS_MATPLOTLIB else styler


def _render_multi_start(cagr_ms):
    if not cagr_ms:
        st.info("Periodo troppo corto per il multi-start (servono almeno la data "
                "di inizio + 1 anno + 1 anno di simulazione residua).")
        return
    st.markdown("### 🔁 Robustezza: CAGR per data di inizio")
    st.caption(
        "Stessa strategia, stessi dati, cambia SOLO la data di inizio. Un CAGR "
        "minimo molto sotto quello della run principale significa che il "
        "risultato è path-dependence (fortuna di pochi trade o di una "
        "posizione concentrata), non edge ripetibile.")
    df_ms = pd.DataFrame(cagr_ms).T
    df_ms["CAGR min %"] = df_ms.min(axis=1)
    df_ms["CAGR mediano %"] = df_ms.median(axis=1)
    date_cols = [c for c in df_ms.columns
                 if c not in ("CAGR min %", "CAGR mediano %")]
    st.dataframe(
        _bg_gradient(df_ms.style.format("{:.1f}"), cmap='RdYlGn', axis=None,
                     subset=date_cols),
        width="stretch",
    )


def _render_yearly_table(results, bh_curves, initial_capital):
    yearly_rows = {}
    for key, r in results.items():
        curve = _strategy_equity_curve(r, initial_capital)
        if curve is not None and len(curve) > 1:
            yearly_rows[r["name"]] = yearly_returns(curve)
    bench_s, bench_lbl = _bench_curve_and_label(bh_curves)
    if bench_s is not None and len(bench_s) > 1:
        yearly_rows[f"🏁 {bench_lbl} B&H"] = yearly_returns(bench_s)
    if 'equal_weight' in bh_curves:
        yearly_rows["Universo Eq. Weight"] = yearly_returns(bh_curves['equal_weight'])

    if not yearly_rows:
        return
    st.markdown("### 📅 Rendimenti per anno civile (check robustezza)")
    st.caption(
        "Una strategia affidabile rende in più regimi di mercato. Rendimenti "
        "concentrati in un solo anno suggeriscono overfitting o fortuna; "
        "ricorda anche il survivorship bias (nel DB ci sono solo i ticker "
        "quotati oggi: i delistati mancano)."
    )
    df_yearly = pd.DataFrame(yearly_rows).T
    df_yearly.columns = [str(c) for c in df_yearly.columns]
    st.dataframe(
        _bg_gradient(df_yearly.style.format("{:+.1%}"), cmap='RdYlGn', axis=None),
        width="stretch",
    )


def _render_comparison_results(run):
    """Rendering dell'ultima run di confronto (dalla cache di sessione):
    cambiare la selezione del log NON rilancia la simulazione."""
    results = run["results"]
    bh_curves = run["bh_curves"]
    initial_capital = run["capital"]
    commission = run["commission"]

    rows = _comparison_summary_rows(results, bh_curves, initial_capital, commission)
    _render_comparison_leaderboard(rows)
    _render_comparison_callouts(rows)
    if run.get("multi_start_requested"):
        _render_multi_start(run.get("multi_start"))
    _render_yearly_table(results, bh_curves, initial_capital)

    log_key = st.selectbox(
        "📋 Log operazioni di quale strategia?",
        list(results.keys()),
        format_func=lambda k: results[k]["name"],
    )
    df_sel = results[log_key]["trades"]
    if df_sel is not None and not df_sel.empty:
        st.dataframe(df_sel, width="stretch")


# 6. PAGINA PRINCIPALE ---

def render_automated_backtest_ui():
    st.title("💹 Backtest")

    render_page_help(
        f"**Cosa fa**: backtest delle **{len(STRATEGY_ORDER)} strategie algoritmiche** sui dati reali del DB locale.",
        "**A cosa serve**: scegliere la strategia da mettere in pratica — obiettivo: battere il S&P500 con poche posizioni e poche operazioni.",
        "Registro completo delle strategie (attive e ritirate, con numeri e motivi) in `docs/STRATEGY_LEDGER.md`.",
        "Prezzi letti dal DB; i ticker mancanti o non aggiornati oggi vengono scaricati da Yahoo con logica delta e salvati.",
        "🧭 *Confronto Strategie PAC*: replay storico del consiglio mensile dei Satelliti PAC — valida la logica mensile (satelliti vs core) prima di seguirla.",
        "Flusso: valida qui in *Confronto Strategie* → applica la migliore in ⚡ **Trading**.",
    )

    st.sidebar.header("⚙️ Configurazione")

    mode = st.sidebar.radio(
        "Modalità Operativa",
        ["🤖 Automated Backtest", "🏁 Confronto Strategie", "🧭 Confronto Strategie PAC"],
        help="Simulazione storica di una strategia, confronto multi-strategia o "
             "replay del consiglio mensile dei Satelliti PAC."
    )

    st.sidebar.divider()

    if mode == "🧭 Confronto Strategie PAC":
        # Modalità con sidebar propria (importo PAC, strategie, core, commissioni):
        # i parametri comuni del backtest (capitale, universo, posizioni max)
        # non si applicano al PAC mensile. La logica PAC vive in
        # ui/pac_backtest.py (specchia core/pac_backtest.py).
        render_pac_backtest_ui()
        return

    initial_capital = st.sidebar.number_input(
        "💰 Capitale Iniziale (€)",
        min_value=100.0,
        value=10000.0,
        step=500.0,
        help="Capitale di partenza per la simulazione."
    )

    st.sidebar.subheader("⏳ Periodo Simulazione")
    start_date_backtest = st.sidebar.date_input("Data Inizio:", pd.to_datetime("2018-01-01"))

    with st.sidebar:
        st.subheader("🎯 Universo Ticker")
        tickers, universe_mode = render_universe_selector(
            default_watchlist=DEFAULT_TRADING_LIST,
            textarea_height=200,
            default_mode=UNIVERSE_WHOLE_DB,
        )

    st.sidebar.subheader("💸 Costi & Rischio")
    commission = st.sidebar.number_input(
        "Commissione per operazione (€)", 0.0, 20.0, 1.0, 0.5,
        help="Commissione del tuo broker per ogni acquisto/vendita (Trade Republic: 1€)."
    )
    max_positions = st.sidebar.slider(
        "Posizioni massime", 1, 5, 3,
        help="Pochi titoli = concentrazione sui migliori segnali e meno commissioni."
    )
    allow_fractional = st.sidebar.checkbox(
        "Ammetti Azioni Frazionate",
        value=False,
        help=("Se attivo, compra es. 0.5 azioni (come in ⚡ Trading). "
              "Se disattivo, solo numeri interi: scegli questa opzione se sul "
              "broker vuoi piazzare veri ordini stop-loss (non disponibili sulle frazioni).")
    )

    if mode == "🤖 Automated Backtest":
        _render_single_backtest_mode(initial_capital, start_date_backtest, tickers,
                                     universe_mode, commission,
                                     max_positions, allow_fractional)
    else:
        _render_comparison_mode(initial_capital, start_date_backtest, tickers,
                                universe_mode, commission,
                                max_positions, allow_fractional)


# Modalità 1: Automated Backtest (strategia singola) ---

def _render_single_backtest_mode(initial_capital, start_date_backtest, tickers,
                                 universe_mode, commission,
                                 max_positions, allow_fractional):
    st.subheader("🤖 Simulazione Portafoglio Algoritmico")
    st.caption("Validazione Strategia: Rischio, Rendimento, Drawdown e impatto commissioni.")

    strategy_key = st.selectbox(
        "🧬 Algoritmo di trading",
        STRATEGY_ORDER,
        index=0,
        format_func=lambda k: STRATEGIES[k].name,
    )
    strat = STRATEGIES[strategy_key]

    with st.expander(f"📖 Logica: {strat.name}", expanded=True):
        st.markdown(strat.description)

    sig = _run_signature(
        strategy=strategy_key, start=start_date_backtest, capital=initial_capital,
        commission=commission, positions=max_positions,
        tickers=tuple(tickers),
        fractional=allow_fractional)

    if st.button("▶️ Avvia Simulazione Completa"):
        params = _build_strategy_params(commission, max_positions)
        loaded = _load_real_data_for_backtest(
            tickers, universe_mode, start_date_backtest)
        if loaded is not None:
            market_data, start_date_str, sim_tickers = loaded
            progress_bar = st.progress(
                0, text=f"Simulazione dal {start_date_str} con capitale €{initial_capital:,.2f}...")
            df_trades, final_cap, diag = run_market_aware_backtest(
                sim_tickers,
                start_date=start_date_str,
                initial_capital=initial_capital,
                preloaded_data=market_data,
                allow_fractional=allow_fractional,
                progress_callback=progress_bar.progress,
                strategy=strategy_key,
                strategy_params=params,
            )
            progress_bar.empty()

            if not df_trades.empty:
                bh_curves = compute_buy_and_hold_curves(
                    market_data, sim_tickers, start_date_str, initial_capital,
                    benchmark_ticker='SPY')
                _store_run(_BT_RUN_KEY, sig, {
                    'trades': df_trades, 'final': final_cap, 'capital': initial_capital,
                    'strategy_name': strat.name, 'bh_curves': bh_curves,
                    'nav': diag.get('nav_daily'), 'diag': diag,
                })
            else:
                st.warning(
                    "Nessun trade generato nel periodo. Prova ad ampliare l'universo ticker, "
                    "un'altra strategia o cambiare la data di inizio.")

    run = _load_run(_BT_RUN_KEY, sig)
    if run:
        _render_strategy_report(run['trades'], run['final'], run['capital'],
                                run['strategy_name'], bh_curves=run['bh_curves'],
                                nav_daily=run['nav'], diag=run['diag'])


# Modalità 2: Confronto Strategie ---

def _render_comparison_mode(initial_capital, start_date_backtest, tickers,
                            universe_mode, commission,
                            max_positions, allow_fractional):
    st.subheader("🏁 Confronto Strategie sullo stesso mercato")
    st.caption(
        "Tutte le strategie girano sugli stessi dati, con le stesse commissioni: "
        "il confronto è fair. L'obiettivo è battere il S&P500 (Buy & Hold del benchmark)."
    )

    with st.expander("⚠️ Limiti del backtest: come leggere i risultati senza farsi ingannare"):
        st.markdown(
            "**Survivorship bias (distorsione da sopravvivenza):** il DB contiene solo i "
            "ticker quotati **oggi**. I titoli/ETF **delistati** (fallimenti, chiusure "
            "prodotti) mancano: le strategie momentum/breakout — che comprano i "
            "vincitori — risultano sistematicamente **troppe ottimiste**. Se un "
            "risultato sembra troppo bello, sospetta prima di tutto questo bias."
        )
        st.markdown(
            "**Igiene dei dati:** le serie vengono ripulite automaticamente da errori "
            "di livello prezzo (prezzi segnaposto, tick corrotti, errori di valuta "
            "rilevati come tripli/dimezzamenti in un passaggio, **anche attraverso "
            "buchi di dati o sospensioni lunghe** — è la porta usata dai reverse split "
            "non aggiustati dei prodotti a leva). La pulizia avviene al caricamento e "
            "il DB non viene mai modificato."
        )
        st.markdown(
            "**Commerciabilità:** ogni ingresso supera un gate di liquidità (prezzo "
            "minimo, volume del giorno, posizione ≤ 20% del turnover medio 20gg in "
            "euro): un segnale su un titolo che nessuno scambia non è eseguibile dal "
            "vivo. Il NAV è prezzato a mercato ogni seduta, quindi drawdown e Sharpe "
            "contano anche i giorni senza operazioni."
        )
        st.markdown(
            "**Anti-overfitting:** guarda la tabella *Rendimenti per anno civile*: una "
            "strategia solida rende in **più regimi** (bull 2021, bear 2022, rimbalzo "
            "2023-26). Vinci solo in un anno = probabile che i parametri siano "
            "aggiustati su quel periodo. Confronta anche il numero di operazioni: "
            "pochi trade = poca significatività statistica."
        )

    selected = st.multiselect(
        "Strategie da confrontare",
        STRATEGY_ORDER,
        default=STRATEGY_ORDER,
        format_func=lambda k: STRATEGIES[k].name,
    )

    with st.expander("📖 Descrizione delle strategie"):
        for k in selected:
            st.markdown(f"**{STRATEGIES[k].name}**")
            st.markdown(STRATEGIES[k].description)
            st.divider()

    robustness_multi = st.checkbox(
        "🔁 Robustezza multi-start (3 date di inizio aggiuntive)",
        value=False,
        help=("Ri-esegue il confronto da +1/+2/+3 anni dalla data di inizio e riporta "
              "il CAGR minimo/mediano per strategia. Se il CAGR cambia molto al "
              "variare della data di partenza, il risultato è path-dependence (fortuna "
              "di pochi trade), non edge. Tempo di esecuzione ~4x."))

    sig = _run_signature(
        strategies=tuple(selected), start=start_date_backtest, capital=initial_capital,
        commission=commission, positions=max_positions,
        tickers=tuple(tickers),
        multi_start=robustness_multi, fractional=allow_fractional)

    if st.button("▶️ Avvia Confronto"):
        if not selected:
            st.warning("Seleziona almeno una strategia.")
        else:
            params = _build_strategy_params(commission, max_positions)
            loaded = _load_real_data_for_backtest(
                tickers, universe_mode, start_date_backtest)
            if loaded is not None:
                market_data, start_date_str, sim_tickers = loaded

                progress_bar = st.progress(0, text="Esecuzione strategie...")
                results = run_strategy_comparison(
                    sim_tickers,
                    strategies=selected,
                    start_date=start_date_str,
                    initial_capital=initial_capital,
                    preloaded_data=market_data,
                    allow_fractional=allow_fractional,
                    progress_callback=progress_bar.progress,
                    strategy_params=params,
                )
                progress_bar.empty()

                bh_curves = compute_buy_and_hold_curves(
                    market_data, sim_tickers, start_date_str, initial_capital,
                    benchmark_ticker='SPY')

                multi_start_tbl = None
                if robustness_multi:
                    prog_ms = st.progress(0.0, text="🔁 Multi-start: simulazioni aggiuntive...")
                    multi_start_tbl = _multi_start_cagr_table(
                        results, selected, sim_tickers, market_data, params,
                        initial_capital, start_date_backtest,
                        progress=prog_ms.progress, allow_fractional=allow_fractional)
                    prog_ms.empty()

                _store_run(_CMP_RUN_KEY, sig, {
                    'results': results, 'bh_curves': bh_curves,
                    'capital': initial_capital, 'commission': commission,
                    'multi_start': multi_start_tbl,
                    'multi_start_requested': robustness_multi,
                })

    run = _load_run(_CMP_RUN_KEY, sig)
    if run:
        _render_comparison_results(run)
