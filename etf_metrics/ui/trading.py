# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from etf_metrics.clients.yahoo_client import (get_failed_tickers,
                                              is_rate_limited, rate_limit_wait_seconds)
from etf_metrics.core.automated_backtest import (
    run_market_aware_backtest,
    run_strategy_comparison,
    prepare_market_data,
    compute_buy_and_hold_curves,
    align_equity_curves,
    BENCH_PROXY_KEY,
    CONFIG,
    STRATEGIES,
    STRATEGY_ORDER,
)
from etf_metrics.ui.components import (
    render_universe_selector,
    list_db_tickers,
    UNIVERSE_WATCHLIST,
    UNIVERSE_WHOLE_DB,
)
from etf_metrics.core.metrics import compute_backtest_performance_metrics


def _bench_curve_and_label(bh_curves):
    """(serie_benchmark, etichetta): S&P500 se presente, proxy universo altrimenti."""
    if not bh_curves:
        return None, None
    if 'benchmark' in bh_curves:
        return bh_curves['benchmark'], "S&P500"
    if 'benchmark_proxy' in bh_curves:
        return bh_curves['benchmark_proxy'], "Universo (proxy)"
    return None, None


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


# ----------------------------------------------------------------------
# Caricamento dati (cache di sessione)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_real_data(tickers_tuple, allow_download):
    """Carica i dati dal DB locale; se allow_download=True scarica da Yahoo
    (in batch paralleli) i ticker mancanti — prima volta con DB vuoto inclusa."""
    return prepare_market_data(list(tickers_tuple), allow_download=allow_download)


def _load_real_data_safe(tickers, allow_download):
    """Wrapper anti-cache-dei-fallimenti: un caricamento vuoto non viene
    memorizzato, così il retry (es. dopo aver ripristinato la rete) funziona."""
    res = _load_real_data(tuple(tickers), allow_download)
    if not res or not any(t in res for t in tickers):
        _load_real_data.clear()
    return res


def _describe_real_load(market_data, tickers, download_attempted):
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
        if download_attempted:
            parts.append(f"⚠️ Non trovati su Yahoo (delisted o simbolo errato?): {', '.join(other)}.")
        else:
            parts.append(f"⚠️ Non nel DB e download disattivato: {', '.join(other)}.")
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
            "\"Universo (proxy)\"). Per confrontarti con il **vero S&P500** "
            "scarica SPY una volta: resterà nel DB e verrà usato automaticamente "
            "(anche per regime e RS).")
        if st.button(f"📥 Scarica {bench} nel DB (benchmark S&P500)",
                     key="download_spy_benchmark"):
            try:
                from etf_metrics.core.data_updater import update_tickers
                with st.spinner(f"Download {bench}..."):
                    update_tickers([bench], period="max")
                _load_real_data.clear()  # il cache dei dati non contiene SPY
                st.success(f"{bench} salvato nel DB: ricarico i dati...")
                st.rerun()
            except Exception as e:
                st.error(f"Download di {bench} non riuscito (connessione assente?): {e}")


def _load_real_data_for_backtest(tickers, universe_mode, allow_download_ui, start_date_backtest):
    """Carica i dati reali per il backtest (DB locale + aggiornamento delta opzionale).

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

    if is_rate_limited():
        st.warning("⏳ Rate limit Yahoo attivo (HTTP 429): i download sono sospesi "
                   f"per ~{max(1, round(rate_limit_wait_seconds() / 60))} minuti. "
                   "Vengono usati solo i dati già presenti nel DB.")

    with st.spinner("📥 Caricamento dati reali" +
                    (" (aggiornamento delta da Yahoo...)" if allow_download_ui
                     else " (solo DB locale)...") + ""):
        market_data = _load_real_data_safe(tickers, allow_download=allow_download_ui)

    if not any(t in market_data for t in tickers):
        if universe_mode == UNIVERSE_WHOLE_DB:
            st.error("Nessun ticker del DB locale è utilizzabile (storico insufficiente?). "
                     "Aggiorna i dati dalla pagina 📥 Gestione Dati e riprova.")
        elif is_rate_limited():
            st.error("Yahoo sta limitando le richieste (HTTP 429) e nessun ticker richiesto "
                     "è nel DB locale. Attendi "
                     f"~{max(1, round(rate_limit_wait_seconds() / 60))} minuti e riprova: "
                     "i dati verranno scaricati automaticamente.")
        elif allow_download_ui:
            st.error("Nessun dato trovato per i ticker richiesti: il download da Yahoo è "
                     "fallito (connessione assente?) oppure i ticker non esistono / sono "
                     "in cache negativa. Verifica la connessione o usa la modalità "
                     "'Dati simulati'.")
        else:
            avail = list_db_tickers()
            hint = ("\n\nTicker con storico sufficiente nel DB: **"
                    + ", ".join(avail[:40]) + "**") if avail else ""
            st.error("Nessuno dei ticker richiesti è presente nel DB locale e il download è "
                     "disattivato. Attiva '📥 Aggiorna da Yahoo', usa la modalità "
                     "'Dati simulati' o scegli ticker presenti nel DB." + hint)
        return None

    st.caption(_describe_real_load(market_data, tickers, allow_download_ui))
    _offer_spy_benchmark(market_data)
    return market_data, str(start_date_backtest), tickers


def _render_strategy_report(df_trades, final_cap, initial_capital, strategy_name,
                            bh_curves=None, show_trades=True):
    """Report comune: KPI, rischio, statistiche trade, equity vs benchmark, log operazioni."""
    metrics = compute_backtest_performance_metrics(df_trades, initial_capital)
    total_return_pct = ((final_cap - initial_capital) / initial_capital) * 100

    # --- SEZIONE 1: KPI PRINCIPALI ---
    st.markdown("### 📊 Performance Report")
    st.caption(f"Strategia: **{strategy_name}**")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Capitale Finale", f"€{final_cap:,.0f}")
    kpi2.metric("Rendimento Totale", f"{total_return_pct:+.2f}%",
                delta_color="normal" if total_return_pct > 0 else "inverse")

    if metrics:
        kpi3.metric("CAGR (Annuo)", f"{metrics['CAGR'] * 100:.2f}%")
        kpi4.metric("Max Drawdown", f"{metrics['Max_Drawdown'] * 100:.2f}%",
                    help="Massima perdita dal picco. Più è basso (vicino a 0), meglio è.",
                    delta_color="inverse")

    # --- SEZIONE 1b: IMPATTO COMMISSIONI (focus "poche operazioni") ---
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

    st.divider()

    # --- SEZIONE 2: RISCHIO E STATISTICHE ---
    st.markdown("### 🛡️ Analisi Rischio & Statistiche (Validation Checklist)")

    if metrics:
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)

        sharpe = metrics['Sharpe']
        col_r1.metric("Sharpe Ratio", f"{sharpe:.2f}",
                      help="> 1.0 Buono, > 2.0 Ottimo. Misura il rendimento per unità di rischio.")

        col_r2.metric("Sortino Ratio", f"{metrics['Sortino']:.2f}",
                      help="Simile allo Sharpe, ma considera solo la volatilità negativa (i crolli).")

        pf = metrics['Profit_Factor']
        col_r3.metric("Profit Factor", f"{pf:.2f}",
                      help="Rapporto tra vincite lorde e perdite lorde. > 1.5 è solido.")

        col_r4.metric("Volatilità", f"{metrics['Volatility'] * 100:.1f}%")

    realized = df_trades[df_trades['Action'] == 'SELL']
    if not realized.empty:
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
            st.warning(
                "⚠️ Attenzione: Numero di operazioni basso (<30). I risultati statistici potrebbero non essere affidabili.")

    # Confronto con benchmark (obiettivo: battere S&P500)
    if bh_curves:
        bench, bench_lbl = _bench_curve_and_label(bh_curves)
        if bench is not None and len(bench) > 1:
            bench_final = bench.iloc[-1]
            alpha = final_cap - bench_final
            st.markdown(
                f"**🏁 Obiettivo 'battere {bench_lbl}':** Benchmark B&H = **€{bench_final:,.0f}** → "
                f"{'🟢 strategia VINCE' if final_cap > bench_final else '🔴 strategia perde'} "
                f"di **€{abs(alpha):,.0f}** ({(alpha / initial_capital) * 100:+.1f}% sul capitale iniziale)."
            )

    st.divider()

    # --- SEZIONE 3: GRAFICI ---
    tab_chart1, tab_chart2 = st.tabs(["📈 Curva Capitale", "📋 Lista Operazioni"])

    with tab_chart1:
        curves = {}
        if metrics and metrics.get('Equity_Curve') is not None:
            curves["Strategia"] = metrics['Equity_Curve']
        else:
            simple_curve = df_trades.drop_duplicates(subset=['Date'], keep='last').set_index('Date')['Capital']
            curves["Strategia"] = simple_curve
        if bh_curves:
            bench_s, bench_lbl = _bench_curve_and_label(bh_curves)
            if bench_s is not None:
                curves[f"{bench_lbl} Buy&Hold"] = bench_s
            if 'equal_weight' in bh_curves:
                curves["Universo Eq. Weight"] = bh_curves['equal_weight']
        st.line_chart(align_equity_curves(curves))

    with tab_chart2:
        if show_trades:
            def style_backtest_rows(row):
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

            fmt = {"Price": "{:.2f}", "PnL_Net": "{:+.2f}", "Capital": "€{:,.0f}",
                   "Total_Equity": "€{:,.0f}", "Qty": "{:.4f}"}
            if 'Commission' in df_trades.columns:
                fmt["Commission"] = "€{:.2f}"
            df_display = df_trades.style.apply(style_backtest_rows, axis=1).format(fmt)
            st.dataframe(df_display, width="stretch")


def _build_strategy_params(commission, max_positions, tax_rate=None):
    params = {'COMMISSION': float(commission), 'MAX_POSITIONS': int(max_positions)}
    return params


def render_trading_ui():
    st.title("💹 Trading & Backtest")

    with st.expander("🧠 Logica Operativa"):
        st.markdown("""
        Suite di **7 strategie algoritmiche** testabili in backtest sui **dati reali del DB
        locale**, con l'obiettivo di **battere il S&P500** restando in poche posizioni e
        facendo **pochissime operazioni** (le commissioni non devono mangiarsi i profitti).

        **Flusso consigliato:** valida le strategie in *Confronto Strategie*, poi metti in
        pratica la migliore con **💲 Portafoglio Live** (stessa logica, operazioni giornaliere
        sul portafoglio reale).

        **Dati:** l'universo può essere **tutto il DB già scaricato** oppure **solo i ticker
        della watchlist**. I prezzi si leggono dal DB; solo i ticker mancanti o non
        aggiornati oggi vengono scaricati da Yahoo con logica **delta** (e salvati nel DB,
        così i giri successivi sono offline). Gestione dati dedicata in **📥 Gestione Dati**.
        """)

    st.sidebar.header("⚙️ Configurazione")

    mode = st.sidebar.radio(
        "Modalità Operativa",
        ["🤖 Automated Backtest", "🏁 Confronto Strategie"],
        help="Simulazione storica di una strategia o confronto multi-strategia."
    )

    st.sidebar.divider()

    initial_capital = st.sidebar.number_input(
        "💰 Capitale Iniziale (€)",
        min_value=100.0,
        value=10000.0,
        step=500.0,
        help="Capitale di partenza per la simulazione."
    )

    start_date_backtest = date(2018, 1, 1)

    st.sidebar.subheader("⏳ Periodo Simulazione")
    start_date_backtest = st.sidebar.date_input("Data Inizio:", pd.to_datetime("2018-01-01"))

    # ------------------------------------------------------------------
    # SIDEBAR: universo ticker — tutto il DB già scaricato oppure watchlist
    # ------------------------------------------------------------------
    with st.sidebar:
        st.subheader("🎯 Universo Ticker")
        tickers, universe_mode, allow_download_ui = render_universe_selector(
            default_watchlist=DEFAULT_TRADING_LIST,
            textarea_height=200,
            show_download=True,
        )

    st.sidebar.subheader("💸 Costi & Rischio")
    commission = st.sidebar.number_input(
        "Commissione per operazione (€)", 0.0, 20.0, 2.0, 0.5,
        help="Commissione del tuo broker per ogni acquisto/vendita."
    )
    max_positions = st.sidebar.slider(
        "Posizioni massime", 1, 5, 3,
        help="Pochi titoli = concentrazione sui migliori segnali e meno commissioni."
    )

    # ------------------------------------------------------------------
    # MODALITÀ 1: AUTOMATED BACKTEST (singola strategia)
    # ------------------------------------------------------------------
    if mode == "🤖 Automated Backtest":
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

        if st.button("▶️ Avvia Simulazione Completa"):
            params = _build_strategy_params(commission, max_positions)

            with st.spinner("Preparazione dati..."):
                loaded = _load_real_data_for_backtest(
                    tickers, universe_mode, allow_download_ui, start_date_backtest)
                if loaded is None:
                    return
                market_data, start_date_str, sim_tickers = loaded

            progress_bar = st.progress(
                0, text=f"Simulazione dal {start_date_str} con capitale €{initial_capital:,.2f}...")
            df_trades, final_cap = run_market_aware_backtest(
                sim_tickers,
                start_date=start_date_str,
                initial_capital=initial_capital,
                preloaded_data=market_data,
                progress_callback=progress_bar.progress,
                strategy=strategy_key,
                strategy_params=params,
            )
            progress_bar.empty()

            if not df_trades.empty:
                bh_curves = compute_buy_and_hold_curves(
                    market_data, sim_tickers, start_date_str, initial_capital,
                    benchmark_ticker='SPY')
                _render_strategy_report(df_trades, final_cap, initial_capital,
                                        strat.name, bh_curves=bh_curves)
            else:
                st.warning(
                    "Nessun trade generato nel periodo. Prova ad ampliare l'universo ticker, "
                    "un'altra strategia o cambiare la data di inizio.")

    # ------------------------------------------------------------------
    # MODALITÀ 2: CONFRONTO STRATEGIE
    # ------------------------------------------------------------------
    elif mode == "🏁 Confronto Strategie":
        st.subheader("🏁 Confronto Strategie sullo stesso mercato")
        st.caption(
            "Tutte le strategie girano sugli stessi dati, con le stesse commissioni: "
            "il confronto è fair. L'obiettivo è battere il S&P500 (Buy & Hold del benchmark)."
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

        if st.button("▶️ Avvia Confronto"):
            if not selected:
                st.warning("Seleziona almeno una strategia.")
                return
            params = _build_strategy_params(commission, max_positions)

            with st.spinner("Preparazione dati..."):
                loaded = _load_real_data_for_backtest(
                    tickers, universe_mode, allow_download_ui, start_date_backtest)
                if loaded is None:
                    return
                market_data, start_date_str, sim_tickers = loaded

            progress_bar = st.progress(0, text="Esecuzione strategie...")
            results = run_strategy_comparison(
                sim_tickers,
                strategies=selected,
                start_date=start_date_str,
                initial_capital=initial_capital,
                preloaded_data=market_data,
                progress_callback=progress_bar.progress,
                strategy_params=params,
            )
            progress_bar.empty()

            bh_curves = compute_buy_and_hold_curves(
                market_data, sim_tickers, start_date_str, initial_capital,
                benchmark_ticker='SPY')

            # --- Tabella riassuntiva ---
            rows = []
            for key, r in results.items():
                df_trades, final = r["trades"], r["final"]
                m = compute_backtest_performance_metrics(df_trades, initial_capital)
                n_sells = int((df_trades['Action'] == 'SELL').sum()) if not df_trades.empty else 0
                comm = float(df_trades['Commission'].sum()) if ('Commission' in df_trades.columns) else 0.0
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
                })

            bench_s, bench_name = _bench_curve_and_label(bh_curves)
            bench_final = float(bench_s.iloc[-1]) if bench_s is not None else None
            if bench_final is not None:
                rows.append({
                    "Strategia": f"🏁 {bench_name} Buy & Hold",
                    "Finale (€)": float(bench_final),
                    "Rend. Tot. %": ((bench_final - initial_capital) / initial_capital) * 100,
                    "CAGR %": np.nan, "Max DD %": np.nan, "Sharpe": np.nan,
                    "Profit Factor": np.nan, "Vendite": 1,
                    "Commissioni (€)": float(commission),
                })

            df_summary = pd.DataFrame(rows).set_index("Strategia").sort_values("Finale (€)", ascending=False)

            st.markdown("### 🏆 Classifica (per capitale finale)")
            st.dataframe(
                df_summary.style.format({
                    "Finale (€)": "€{:,.0f}", "Rend. Tot. %": "{:+.1f}",
                    "CAGR %": "{:.1f}", "Max DD %": "{:.1f}", "Sharpe": "{:.2f}",
                    "Profit Factor": "{:.2f}", "Vendite": "{:.0f}", "Commissioni (€)": "€{:,.0f}",
                }),
                width="stretch",
            )

            valid = [r for r in rows if not str(r["Strategia"]).startswith("🏁")]
            if valid:
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

            # --- Grafico equity ---
            st.markdown("### 📈 Curve Capitale")
            curves = {}
            for key, r in results.items():
                m = compute_backtest_performance_metrics(r["trades"], initial_capital)
                if m and m.get('Equity_Curve') is not None:
                    curves[r["name"]] = m['Equity_Curve']
                elif not r["trades"].empty:
                    curves[r["name"]] = r["trades"].drop_duplicates(
                        subset=['Date'], keep='last').set_index('Date')['Capital']
            bench_s2, bench_lbl2 = _bench_curve_and_label(bh_curves)
            if bench_s2 is not None:
                curves[f"{bench_lbl2} Buy&Hold"] = bench_s2
            if 'equal_weight' in bh_curves:
                curves["Universo Eq. Weight"] = bh_curves['equal_weight']

            if curves:
                st.line_chart(align_equity_curves(curves))

            # --- Log operazioni (selezione strategia) ---
            log_key = st.selectbox(
                "📋 Log operazioni di quale strategia?",
                list(results.keys()),
                format_func=lambda k: results[k]["name"],
            )
            df_sel = results[log_key]["trades"]
            if df_sel is not None and not df_sel.empty:
                st.dataframe(df_sel, width="stretch")
