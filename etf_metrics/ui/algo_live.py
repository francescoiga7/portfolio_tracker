# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from etf_metrics.core.algo_live import load_algo_state, save_algo_state, run_daily_update


def render_algo_live_ui():
    st.title("💲Portafoglio Live")

    state = load_algo_state()

    # --- SIDEBAR: SETUP CAPITALE E OPZIONI ---
    st.sidebar.header("⚙️ Configurazione")

    # 1. Modifica Capitale Dinamica
    current_cash = state.get('cash', 10000.0)
    new_cash = st.sidebar.number_input(
        "💰 Capitale Liquido ($)",
        min_value=0.0,
        value=float(current_cash),
        step=100.0,
        help="Modifica manualmente la liquidità disponibile."
    )

    # Aggiorna lo stato se l'utente cambia il numero
    if new_cash != current_cash:
        state['cash'] = new_cash
        save_algo_state(state)
        st.rerun()

    # 2. Flag Frazionato
    allow_fractional = st.sidebar.checkbox(
        "Ammetti Azioni Frazionate",
        value=True,
        help="Se attivo, compra es. 0.5 azioni. Se disattivo, solo numeri interi (es. NVDA a 1000$ richiede almeno 1000$)."
    )

    # 3. Tickers
    default_tickers = "NVDA\nAMD\nTSLA\nAAPL\nMSFT\nAMZN\nGOOGL\nMETA\nNFLX\nPLTR\nCOIN"
    with st.sidebar.expander("Lista Ticker", expanded=False):
        tickers_input = st.text_area("Ticker", default_tickers, height=150)
    tickers_list = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

    # Reset Button
    if st.sidebar.button("🗑️ Reset Totale"):
        state = {"cash": 10000.0, "positions": {}, "trade_log": [], "last_update": None}
        save_algo_state(state)
        st.rerun()

    # --- DASHBOARD ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Liquidità (Cash)", f"${state['cash']:.2f}")
    col2.metric("Posizioni Aperte", len(state['positions']))
    col3.metric("Ultimo Aggiornamento", state['last_update'] if state['last_update'] else "Mai")

    st.divider()

    # --- ACTION ---
    if st.button("🚀 ESEGUI ANALISI GIORNALIERA", type="primary", width="stretch"):
        with st.status("Elaborazione Strategia...", expanded=True) as status:
            st.write("📥 Analisi Mercato e Regime...")
            new_state, msgs = run_daily_update(state, tickers_list, allow_fractional)
            save_algo_state(new_state)

            for m in msgs:
                if "VENDITA" in m or "ACQUISTO" in m:
                    st.write(f"**{m}**")
                else:
                    st.write(m)
            status.update(label="Fatto!", state="complete")
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
                "Entry": f"${data['entry_price']:.2f}",
                "Stop": f"${data['stop_loss']:.2f}",
                "Date": data['entry_date']
            })
        st.dataframe(pd.DataFrame(pos_data), width="stretch")
    else:
        st.info("Portafoglio Flat (100% Cash)")

    st.subheader("📜 Log")
    if state['trade_log']:
        st.dataframe(pd.DataFrame(state['trade_log']), width="stretch")
