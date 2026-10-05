# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from etf_metrics.core.algo_live import (
    DEFAULT_STRATEGY, load_algo_state, save_algo_state, run_daily_update,
    get_latest_market_data)
from etf_metrics.core.automated_backtest import CONFIG, STRATEGIES, STRATEGY_ORDER
from etf_metrics.ui.components import render_universe_selector, UNIVERSE_WHOLE_DB

DEFAULT_LIVE_LIST = "NVDA\nAMD\nTSLA\nAAPL\nMSFT\nAMZN\nGOOGL\nMETA\nNFLX\nPLTR\nCOIN"


def render_algo_live_ui():
    st.title("💲Portafoglio Live")

    state = load_algo_state()

    with st.expander("🧠 Come funziona"):
        st.markdown("""
        Il Portafoglio Live **mette in pratica la strategia del backtest** sul tuo
        portafoglio reale (es. Trade Republic):

        1. Scegli la strategia che ha superato il backtest (es. in *Confronto Strategie*)
        2. Lanciala ogni giorno (qui o via cron con `scripts/live_daily.py`, es. alle 19:00)
        3. Esegui su Trade Republic le operazioni consigliate (🔴 vendi / 🟢 compra)

        La logica è **identica al motore di backtest** (stesse regole di ingresso/uscita,
        stop, ribilanciamento, limite acquisti/mese) applicata allo stato reale:
        cassa e posizioni che vedi qui. Le posizioni aperte sono sempre monitorate,
        anche se i loro ticker sono fuori universo.
        """)

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

    # 2. Capitale
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
    def_commission = strat.defaults.get('COMMISSION', CONFIG['COMMISSION'])
    def_max_pos = strat.defaults.get('MAX_POSITIONS', CONFIG['MAX_POSITIONS'])
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
        value=True,
        help="Se attivo, compra es. 0.5 azioni. Se disattivo, solo numeri interi."
    )

    # 5. Universo ticker: tutto il DB già scaricato oppure watchlist
    with st.sidebar.expander("🎯 Universo Ticker", expanded=True):
        tickers_list, universe_mode, allow_download = render_universe_selector(
            default_watchlist=DEFAULT_LIVE_LIST,
            textarea_height=150,
            show_download=True,
        )

    # Reset Button
    if st.sidebar.button("🗑️ Reset Totale"):
        from etf_metrics.core.algo_live import _default_state
        state = _default_state()
        state['strategy'] = strategy_key
        save_algo_state(state)
        st.rerun()

    # --- DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Liquidità (Cash)", f"€{state['cash']:,.2f}")
    col2.metric("Posizioni Aperte", len(state['positions']))
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
        if not tickers_list and not state['positions']:
            st.warning("Nessun ticker da analizzare: compila la watchlist o popola il DB "
                       "dalla pagina 📥 Gestione Dati.")
        else:
            # persiste i costi scelti e l'universo (riusato anche dallo script cron)
            state['commission'] = float(commission)
            state['max_positions'] = int(max_positions)
            state['universe_tickers'] = list(tickers_list)
            with st.status("Elaborazione Strategia...", expanded=True) as status:
                # Le posizioni aperte restano sempre monitorate, anche fuori universo
                universe = sorted(set(tickers_list) | set(state['positions'].keys()))
                st.write(f"📥 Caricamento dati ({len(universe)} ticker, DB locale + delta)...")
                market_data = get_latest_market_data(universe, allow_download=allow_download)

                if not market_data:
                    status.update(label="Nessun dato disponibile", state="error")
                    st.error("Nessun dato disponibile: i ticker non sono nel DB locale e il "
                             "download è fallito o disattivato. Scarica i dati dalla pagina "
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
        st.caption("💡 Aggiorna manualmente la liquidità in sidebar dopo aver eseguito "
                   "le operazioni su Trade Republic (versamenti/prelievi).")
    else:
        st.info("Portafoglio Flat (100% Cash)")

    st.subheader("📜 Log")
    if state['trade_log']:
        st.dataframe(pd.DataFrame(state['trade_log']), width="stretch")
        st.caption("⏰ Per l'esecuzione automatica giornaliera (es. alle 19:00) vedi "
                   "`scripts/live_daily.py` — la nota cron nel README.")
