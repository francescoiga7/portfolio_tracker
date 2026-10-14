# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from etf_metrics.core.trading import (
    DEFAULT_STRATEGY, load_algo_state, save_algo_state, run_daily_update,
    get_latest_market_data, portfolio_position_tickers,
    analyze_pac_portfolio, pac_universe_tickers)
from etf_metrics.core.automated_backtest import CONFIG, STRATEGIES, STRATEGY_ORDER
from etf_metrics.core.data_updater import update_tickers
from etf_metrics.ui.components import (render_universe_selector,
                                       render_page_help,
                                       UNIVERSE_WHOLE_DB, get_portfolio_store)

DEFAULT_LIVE_LIST = "NVDA\nAMD\nTSLA\nAAPL\nMSFT\nAMZN\nGOOGL\nMETA\nNFLX\nPLTR\nCOIN"


def _throttled_logger(step: int = 10):
    """Callback di progresso che scrive in st.status al massimo ogni `step`%:
    con universi grandi (migliaia di ticker) il log resterebbe illeggibile."""
    last = [0.0]

    def _log(done, total, msg=""):
        try:
            frac = (done / total) if total else 1.0
            if frac - last[0] >= (step / 100.0) or frac >= 1.0:
                last[0] = frac
                st.write(msg if msg else f"{done}/{total}")
        except Exception:
            pass

    return _log


def _run_pac_analysis(state, store, portfolio_name, strategy_key,
                      pac_amount, core_tickers, update_data):
    """Esegue l'analisi Satelliti PAC e ne salva il report in session_state."""
    transactions = (store.get(portfolio_name) or {}).get('transactions', [])
    if not transactions:
        st.warning("Il portafoglio selezionato è vuoto: registra prima le tue "
                   "posizioni nel 💼 Portfolio Tracker.")
        return

    with st.status("Analisi Satelliti PAC...", expanded=True) as status:
        st.write("Risoluzione ticker delle posizioni...")
        pos_tickers, _mapping, unresolved = portfolio_position_tickers(state, transactions)
        if unresolved:
            st.write(f"⚠️ ISIN non risolti (ignorati): {', '.join(unresolved)}")

        universe = pac_universe_tickers()
        st.write(f"Universo candidati: **{len(universe)}** strumenti ETF dal DB locale "
                 "(scoperti in 📥 Gestione Dati → 🌍 Universo ETF).")

        load_list = sorted(set(pos_tickers) | set(universe))
        if update_data and load_list:
            st.write(f"Aggiornamento dati (delta) di {len(load_list)} ticker...")
            rep = update_tickers(load_list, period="5y", skip_updated_today=True,
                                 progress_callback=_throttled_logger(20))
            st.write(f"🗄️ DB: {len(rep['results'])} ticker aggiornati, "
                     f"{len(rep['skipped'])} saltati (già a punto o senza dati).")
            if rep.get('rate_limited'):
                st.warning("Rate limit Yahoo: aggiornamento interrotto, l'analisi "
                           "prosegue con i dati già nel DB.")

        st.write("Caricamento dati e calcolo indicatori...")
        market_data = get_latest_market_data(load_list, allow_download=False,
                                             progress=_throttled_logger(10))
        if not market_data:
            status.update(label="Nessun dato disponibile", state="error")
            st.error("Nessun dato disponibile: i ticker non sono nel DB locale. "
                     "Aggiorna i dati dalla pagina 📥 Gestione Dati e riprova.")
            return

        report, msgs = analyze_pac_portfolio(
            state, transactions, strategy=strategy_key,
            market_data=market_data, core_tickers=core_tickers,
            pac_amount=pac_amount)
        # persiste solo la cache isin_map (l'analisi è read-only)
        save_algo_state(state)
        st.session_state.pac_report = report
        st.session_state.pac_messages = msgs
        status.update(label="Analisi completata", state="complete")


def _render_pac_report():
    """Renderizza il report dell'analisi Satelliti PAC (da session_state)."""
    report = st.session_state.get('pac_report')
    if not report:
        return
    msgs = st.session_state.get('pac_messages') or []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data dati", str(report['date'].date()) if report['date'] is not None else "—")
    c2.metric("Regime", report['regime'] or "—")
    c3.metric("Vendite consigl.", report['sell_count'])
    c4.metric("Satelliti monitorati", len(report['positions']))

    # --- Vendite consigliate ---
    sells = [p for p in report['positions'] if p['verdict'] == 'SELL']
    if sells:
        for p in sells:
            a = p['action']
            st.error(f"🔴 **VENDI {p['name']}** ({p['ticker']}): {a['qty']:g} pz @ €{a['price']:.2f} "
                     f"— {a['reason']} · P/L {p['pnl_pct']:+.1%}")
    elif report['positions']:
        st.success("✅ Nessuna vendita da eseguire sulle posizioni satellite.")

    # --- Posizioni satellite (tabella + condizioni di uscita) ---
    rows = [{
        "Ticker": p['ticker'], "Nome": p['name'],
        "Qtà": round(p['qty'], 4),
        "Prezzo medio": f"€{p['entry_price']:.2f}" if p['entry_price'] else "—",
        "Ultimo": f"€{p['last_price']:.2f}" if p['last_price'] else "—",
        "P/L %": f"{p['pnl_pct']:+.1%}" if p['pnl_pct'] is not None else "—",
        "Stop": f"€{p['stop']:.2f}" if p.get('stop') else "—",
        "Verdetto": "🔴 VENDI" if p['verdict'] == 'SELL' else "🟢 HOLD",
        "Satellite": "🛰️" if p.get('satellite') else "",
    } for p in report['positions']]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        with st.expander("📅 Quando vendere? (condizioni di uscita attuali)", expanded=True):
            for p in report['positions']:
                st.markdown(f"**{p['name']}** ({p['ticker']}) — {p['qty']:g} pz @ €{p['entry_price']:.2f}")
                if p.get('note'):
                    st.caption(p['note'])
                elif p.get('outlook'):
                    st.markdown(p['outlook'])
                else:
                    st.caption("Nessuna condizione di uscita calcolabile.")

    # --- Core (buy & hold) ---
    if report['core']:
        with st.expander(f"🛡️ Core — buy & hold ({len(report['core'])} posizioni, non analizzate)"):
            st.dataframe(pd.DataFrame([{
                "Ticker": c['ticker'], "Nome": c['name'],
                "Qtà": round(c['qty'], 4),
                "Ultimo": f"€{c['last_price']:.2f}" if c['last_price'] else "—",
            } for c in report['core']]), width="stretch", hide_index=True)

    # --- Acquisto del mese ---
    buy = report.get('buy')
    if report['buy_status'] == 'ok' and buy:
        qty_txt = f" ≈ **{buy['qty']:g} pz**" if buy.get('qty') else ""
        st.success(f"🟢 **ACQUISTO DEL MESE: {buy['name']}** ({buy['ticker']}) @ €{buy['price']:.2f}"
                   f"{qty_txt} — {buy['reason']} (score {buy['score']:.0f})")
        st.caption("💡 Esegui l'acquisto sul broker e registralo nel 💼 Portfolio Tracker "
                   "con la checkbox 🛰️ satellite attiva: il PAC del mese risulterà già fatto.")
    elif report['buy_status'] == 'already_bought':
        st.info(f"✅ PAC di questo mese già eseguito: **{report['buy_note']}**.")
    elif report['buy_status'] == 'no_regime':
        st.info(f"💤 Nessun satellite questo mese ({report['buy_note']}) → "
                "**destina il PAC al core** (buy & hold).")
    elif report['buy_status'] == 'no_candidate':
        st.info(f"🔎 {report['buy_note']} → **destina il PAC al core** (buy & hold).")
    elif report['buy_status'] == 'no_data':
        st.warning(report['buy_note'])

    if report.get('unresolved'):
        st.caption(f"⚠️ ISIN non risolti su Yahoo (ignorati): {', '.join(report['unresolved'])}")

    with st.expander("🔍 Log analisi"):
        st.markdown("\n".join(f"- {m}" for m in msgs) if msgs else "—")


def render_pac_ui():
    """Pagina 🧭 Satelliti PAC: consiglio mensile core-satellite sul portafoglio reale (read-only)."""
    state = load_algo_state()

    st.title("🧭 Satelliti PAC")
    st.caption("Consiglio mensile core & satellite sul portafoglio reale "
               "(saved_portfolio.json), in sola lettura.")

    render_page_help(
        "**Cosa fa**: analizza il portafoglio reale e consiglia le mosse del mese.",
        "**A cosa serve**: gestire il PAC con metodo core & satellite.",
        "**Core** (es. VWCE): buy & hold — mai analizzato per la vendita, mai candidato all'acquisto.",
        "**Satelliti**: le regole di uscita della strategia (stop ATR, trailing, momentum/regime) dicono se e cosa vendere.",
        "**PAC del mese**: il miglior ETF dell'universo ETF (📥 Gestione Dati) → cosa comprare; max 1 satellite al mese.",
        "Regime sfavorevole o PAC già eseguito → il consiglio è destinare il PAC al core.",
        "Nessuna operazione automatica: esegui sul broker e registra nel 💼 Portfolio Tracker con la checkbox 🛰️ satellite.",
    )

    store = get_portfolio_store()
    pac_portfolios = store.names()
    if not pac_portfolios:
        st.info("Nessun portafoglio salvato: crea il tuo portafoglio nel 💼 Portfolio Tracker "
                "e registra le posizioni (core + satelliti) per iniziare.")
        return

    default_pf = st.session_state.get("current_portfolio_name", "")
    c1, c2 = st.columns(2)
    with c1:
        pac_pf = st.selectbox(
            "📁 Portafoglio da analizzare", pac_portfolios,
            index=pac_portfolios.index(default_pf) if default_pf in pac_portfolios else 0,
            key="pac_portfolio")
    with c2:
        default_strat = state.get('strategy') or DEFAULT_STRATEGY
        pac_strat_key = st.selectbox(
            "🧬 Strategia (regole ingresso/uscita)", STRATEGY_ORDER,
            index=STRATEGY_ORDER.index(default_strat) if default_strat in STRATEGY_ORDER else 0,
            format_func=lambda k: STRATEGIES[k].name,
            key="pac_strategy")

    c3, c4 = st.columns(2)
    with c3:
        pac_amount = st.number_input(
            "💰 Importo PAC mensile (€)", 0.0, 1_000_000.0, 200.0, 50.0,
            format="%.2f", key="pac_amount",
            help="Quanto investi ogni mese nel satellite: serve solo per suggerire "
                 "la quantità da comprare.")
    with c4:
        core_input = st.text_input(
            "🛡️ Ticker CORE (buy & hold, mai vendere)", value="VWCE.MI",
            key="pac_core",
            help="Ticker separati da virgola (es. VWCE.MI, EIMI.MI). Le posizioni core "
                 "non vengono mai analizzate per la vendita né proposte all'acquisto.")

    pac_update = st.checkbox("🔄 Aggiorna i dati (delta) prima dell'analisi",
                             value=True, key="pac_update")

    if st.button("🔍 ANALIZZA E CONSIGLIA", key="pac_run", type="primary",
                 width="stretch"):
        core_tickers = [t.strip() for t in core_input.split(",") if t.strip()]
        _run_pac_analysis(state, store, pac_pf, pac_strat_key,
                          float(pac_amount), core_tickers, pac_update)

    _render_pac_report()


def render_algo_daily_ui():
    """Pagina ⚡ Trading: algoritmo giornaliero sul portafoglio reale."""
    state = load_algo_state()

    st.title("⚡ Trading")
    st.caption("La strategia del backtest applicata al portafoglio reale "
               "(stato: live_algo_portfolio.json).")

    render_page_help(
        "**Cosa fa**: consigli 🔴 vendi / 🟢 compra sul portafoglio reale, ogni giorno.",
        "**A cosa serve**: eseguire sul broker i consigli dell'algoritmo.",
        "🤖 Logica **identica al motore di backtest**: ingressi, uscite, stop, limite acquisti/mese.",
        "Le posizioni aperte restano sempre monitorate, anche fuori universo.",
        "Il backtest del consiglio mensile dei Satelliti PAC è in 💹 Backtest → 🧭 *Confronto Strategie PAC*.",
        "Automazione serale: `scripts/live_daily.py` via cron (es. alle 19:00).",
    )

    _render_daily_algo(state)


def _render_daily_algo(state):
    """Modalità 🤖: esecuzione giornaliera della strategia sul portafoglio reale."""
    # --- SIDEBAR: SETUP CAPITALE E OPZIONI ---
    st.sidebar.header("⚙️ Configurazione")

    # 1. Strategia (dal backtest)
    current_strategy = state.get('strategy') or DEFAULT_STRATEGY
    idx = STRATEGY_ORDER.index(current_strategy) if current_strategy in STRATEGY_ORDER else 0
    strategy_key = st.sidebar.selectbox(
        "🧬 Strategia (validata in backtest)",
        STRATEGY_ORDER,
        index=idx,
        format_func=lambda k: STRATEGIES[k].name,
        help="Scegli la strategia che ha battuto il benchmark nel backtest: "
             "la sua logica verrà applicata ogni giorno al portafoglio reale.",
    )
    if strategy_key != current_strategy:
        # cambio strategia: azzera gli override dei costi ai default della strategia
        state['strategy'] = strategy_key
        state['commission'] = None
        state['max_positions'] = None
        save_algo_state(state)
        st.rerun()
    strat = STRATEGIES[strategy_key]

    with st.sidebar.expander(f"📖 Logica: {strat.name}", expanded=False):
        st.markdown(strat.description)

    # 1b. Costi di default della strategia
    def_commission = strat.defaults.get('COMMISSION', CONFIG['COMMISSION'])
    def_max_pos = strat.defaults.get('MAX_POSITIONS', CONFIG['MAX_POSITIONS'])

    # 2. Capitale liquido (stato autonomo: aggiorna dopo versamenti/prelievi)
    current_cash = state.get('cash', 10000.0)
    new_cash = st.sidebar.number_input(
        "💰 Capitale Liquido (€)",
        min_value=0.0,
        value=float(current_cash),
        step=100.0,
        help="Modifica manualmente la liquidità disponibile (dopo ogni versamento/"
             "prelievo o operazione eseguita su Trade Republic)."
    )
    if new_cash != current_cash:
        state['cash'] = new_cash
        save_algo_state(state)
        st.rerun()

    # 3. Costi & rischio (default = quelli della strategia)
    commission = st.sidebar.number_input(
        "💸 Commissione per operazione (€)", 0.0, 20.0,
        float(state.get('commission') if state.get('commission') is not None else def_commission),
        0.5, help="Commissione del tuo broker (Trade Republic: 1€ per azioni).")
    max_positions = st.sidebar.slider(
        "Posizioni massime", 1, 8,
        int(state.get('max_positions') if state.get('max_positions') is not None else def_max_pos),
        help="Pochi titoli = concentrazione sui migliori segnali e meno commissioni.")

    # 4. Flag Frazionato
    allow_fractional = st.sidebar.checkbox(
        "Ammetti Azioni Frazionate",
        value=False,
        help="Se attivo, compra es. 0.5 azioni. Se disattivo, solo numeri interi."
    )

    # 5. Universo ticker: tutto il DB già scaricato oppure watchlist
    with st.sidebar.expander("🎯 Universo Ticker", expanded=True):
        tickers_list, universe_mode = render_universe_selector(
            default_watchlist=DEFAULT_LIVE_LIST,
            textarea_height=150,
            default_mode=UNIVERSE_WHOLE_DB,
        )

    # Reset Button
    if st.sidebar.button("🗑️ Reset Totale"):
        from etf_metrics.core.trading import _default_state
        state = _default_state()
        state['strategy'] = strategy_key
        save_algo_state(state)
        st.rerun()

    # --- DASHBOARD ---
    cash_display = state['cash']
    n_open = len(state['positions'])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Liquidità (Cash)", f"€{cash_display:,.2f}")
    col2.metric("Posizioni Aperte", n_open)
    col3.metric("Strategia", STRATEGIES[state.get('strategy') or DEFAULT_STRATEGY].name.split(" (")[0])
    col4.metric("Ultimo Aggiornamento", state['last_update'] if state['last_update'] else "Mai")

    if universe_mode == UNIVERSE_WHOLE_DB and tickers_list:
        st.caption(f"📦 Universo: **tutto il DB locale** ({len(tickers_list)} ticker). "
                   "Le posizioni aperte vengono sempre monitorate, anche fuori universo.")
    elif tickers_list:
        st.caption(f"✍️ Universo: **watchlist** ({len(tickers_list)} ticker). "
                   "Le posizioni aperte vengono sempre monitorate, anche fuori watchlist.")

    st.divider()

    # --- ACTION ---
    if st.button("🚀 ESEGUI ANALISI GIORNALIERA", type="primary", width="stretch"):
        pos_tickers = list(state['positions'].keys())
        if not tickers_list and not pos_tickers:
            st.warning("Nessun ticker da analizzare: compila la watchlist o popola il DB "
                       "dalla pagina 📥 Gestione Dati.")
        else:
            # persiste i costi scelti e l'universo (riusato anche dallo script cron)
            state['commission'] = float(commission)
            state['max_positions'] = int(max_positions)
            state['universe_tickers'] = list(tickers_list)
            with st.status("Elaborazione Strategia...", expanded=True) as status:
                # Le posizioni aperte restano sempre monitorate, anche fuori universo
                universe = sorted(set(tickers_list) | set(pos_tickers))
                st.write("Caricamento dati da parquet")
                market_data = get_latest_market_data(universe, allow_download=False)

                if not market_data:
                    status.update(label="Nessun dato disponibile", state="error")
                    st.error("Nessun dato disponibile: i ticker non sono nel DB locale. "
                             "Scarica i dati dalla pagina "
                             "📥 Gestione Dati e riprova.")
                else:
                    new_state, msgs = run_daily_update(
                        state, tickers_list, allow_fractional,
                        market_data=market_data, strategy=strategy_key)
                    save_algo_state(new_state)

                    actions = [m for m in msgs if m.startswith(("🔴", "🟢"))]
                    for m in msgs:
                        if m.startswith(("🔴", "🟢")):
                            st.markdown(f"**{m}**")
                        else:
                            st.write(m)
                    status.update(
                        label=f"Fatto! {len(actions)} operazioni consigliate."
                        if actions else "Fatto! Nessuna operazione da eseguire.",
                        state="complete")
                    st.session_state['algo_updated'] = True

    if st.session_state.get('algo_updated'):
        st.session_state['algo_updated'] = False
        st.rerun()

    # --- TABELLE ---
    st.subheader("💼 Posizioni")
    if state['positions']:
        pos_data = []
        for t, data in state['positions'].items():
            pos_data.append({
                "Ticker": t,
                "Qty": f"{data['qty']:.4f}" if allow_fractional else f"{data['qty']:.0f}",
                "Entry": f"€{data['entry_price']:.2f}",
                "Stop": f"€{data['stop_loss']:.2f}" if data.get('stop_loss') else "—",
                "Data": data['entry_date'],
                "TP1": "✅" if data.get('tp1_taken') else "",
            })
        st.dataframe(pd.DataFrame(pos_data), width="stretch")
        with st.expander("📅 Quando verrà venduta? (condizioni di uscita attuali)", expanded=True):
            for t, data in state['positions'].items():
                st.markdown(f"**{t}** — {data['qty']:g} pz @ €{data['entry_price']:.2f}")
                outlook = data.get('exit_outlook')
                if outlook:
                    st.markdown(outlook)
                else:
                    st.caption("Condizioni non ancora calcolate: esegui l'analisi giornaliera 🚀.")
        st.caption("💡 Aggiorna manualmente la liquidità in sidebar dopo aver eseguito "
                   "le operazioni su Trade Republic (versamenti/prelievi).")
    else:
        st.info("Portafoglio Flat (100% Cash)")

    st.subheader("📜 Log")
    if state['trade_log']:
        st.dataframe(pd.DataFrame(state['trade_log']), width="stretch")
        st.caption("⏰ Per l'esecuzione automatica giornaliera (es. alle 19:00) vedi "
                   "`scripts/live_daily.py` — la nota cron nel README.")
