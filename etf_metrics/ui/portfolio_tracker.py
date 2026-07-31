# -*- coding: utf-8 -*-
from datetime import datetime

import pandas as pd
import streamlit as st

from etf_metrics.core.portfolio_tracker import calculate_portfolio_pnl
from etf_metrics.shared.config import PORTFOLIO_FILE
from etf_metrics.ui.components import (
    format_dataframe,
    render_allocation_pie,
    get_portfolio_store,
)


def render_portfolio_tracker_ui():
    store = get_portfolio_store()
    # Persiste eventuali modifiche in memoria al file (safety net al primo avvio).
    store.save_to_json(PORTFOLIO_FILE)

    st.title("💼 Portfolio Tracker")
    st.caption("Analizza le tue posizioni, monitora le performance e gestisci le transazioni.")

    with st.expander("📖 Metodologia e Funzionamento"):
        st.markdown("""
        Questa sezione ti permette di monitorare e analizzare i tuoi portafogli di investimento. Puoi:
        - **Creare nuovi portafogli** o **caricare portafogli esistenti**.
        - **Aggiungere transazioni** di acquisto e vendita per ogni portafoglio.
        - **Visualizzare un riepilogo** con le metriche principali del tuo portafoglio.
        - **Analizzare le posizioni aperte** con dettagli su P&L non realizzato e segnali di trend.
        - **Consultare lo storico delle transazioni** e il cassetto fiscale con il P&L realizzato.
        """)

    st.sidebar.header("⚙️ Impostazioni")
    saved_portfolios = store.names()
    current_portfolio_name = st.session_state.get("current_portfolio_name", "")

    with st.sidebar.expander("Carica o Crea Portafoglio", expanded=not current_portfolio_name):
        action = st.radio("Azione", ["Carica Portafoglio", "Crea Nuovo Portafoglio"], label_visibility="collapsed")
        if action == "Carica Portafoglio":
            if not saved_portfolios:
                st.warning("Nessun portafoglio salvato.")
            else:
                selected = st.selectbox(
                    "Seleziona Portafoglio",
                    saved_portfolios,
                    index=saved_portfolios.index(current_portfolio_name) if current_portfolio_name in saved_portfolios else 0,
                )
                if st.button("Carica"):
                    st.session_state.current_portfolio_name = selected
                    st.rerun()
        else:
            new_name = st.text_input("Nome Nuovo Portafoglio", "Il Mio Portafoglio")
            if st.button("Crea"):
                store.ensure_portfolio(new_name)
                st.session_state.current_portfolio_name = new_name
                st.rerun()

    commission = 1.0
    tax_rate = 26

    if current_portfolio_name:
        st.sidebar.subheader(f"Gestisci: '{current_portfolio_name}'")
        with st.sidebar.expander("➕ Aggiungi Transazione"):
            trans_type = st.radio("Tipo", ["Acquisto", "Vendita"], horizontal=True)
            form_key = "buy_form" if trans_type == "Acquisto" else "sell_form"
            with st.form(form_key):
                isin = st.text_input("ISIN")
                qty = st.number_input("Quantità", min_value=0.000001, step=0.0004, format="%.6f")
                price = st.number_input(f"Prezzo {trans_type} (€)", min_value=0.01, step=0.01, format="%.2f")
                date = st.date_input(f"Data {trans_type}", datetime.now().date())
                is_satellite = st.checkbox("Satellite?", key=f"satellite_flag_{form_key}")
                if st.form_submit_button(f"Registra {trans_type}"):
                    trans_data = {
                        "type": "buy" if trans_type == "Acquisto" else "sell",
                        "isin": isin.strip().upper(),
                        "quantity": qty, "price": price, "date": date,
                        "satellite": is_satellite if trans_type == "Acquisto" else False,
                    }
                    store.add_transaction(current_portfolio_name, trans_data)
                    store.save_to_json(PORTFOLIO_FILE)
                    st.rerun()
        with st.sidebar.expander("⚙️ Impostazioni Fiscali"):
            commission = st.number_input("Commissione per operazione (€)", min_value=0.0, value=1.0, step=0.5, format="%.2f")
            tax_rate = st.slider("Imposta plusvalenze (%)", 0, 100, 26, 1)

    if not current_portfolio_name:
        st.info("👈 Crea o carica un portafoglio dalla barra laterale per iniziare.")
        return

    portfolio_data = store.get(current_portfolio_name)
    if not portfolio_data or not portfolio_data.get("transactions"):
        st.info("Questo portafoglio è vuoto. Aggiungi una transazione per iniziare.")
        return

    with st.spinner("Aggiornamento P&L e Segnali di Trend in corso..."):
        pnl_results = calculate_portfolio_pnl(portfolio_data["transactions"], commission, tax_rate)

    tab1, tab2, tab3 = st.tabs(["🧭 Riepilogo", "📊 Posizioni Dettagliate", "🗃️ Cassetto Fiscale e Storico"])

    with tab1:
        st.header("Riepilogo Generale")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valore Attuale", f"€{pnl_results['current_value']:,.2f}")
        col2.metric("P&L Totale", f"€{pnl_results['total_pnl']:,.2f}", f"{pnl_results['total_pnl_pct']:.2f}%")
        col3.metric("Investito (Aperte)", f"€{pnl_results['net_invested']:,.2f}")
        col4.metric("P&L Realizzato", f"€{pnl_results['realized_pnl']:,.2f}")
        st.markdown("---")
        st.subheader("Allocazione Attuale per ISIN")
        if pnl_results['open_positions']:
            labels = [p['isin'] for p in pnl_results['open_positions']]
            values = [p['current_amount'] for p in pnl_results['open_positions']]
            render_allocation_pie(labels, values, "")
        else:
            st.info("Nessuna posizione aperta per mostrare l'allocazione.")

    with tab2:
        st.header("Analisi delle Posizioni")
        st.subheader("Portafoglio Live: Posizioni Aperte")
        if pnl_results['open_positions']:
            df_open = pd.DataFrame(pnl_results['open_positions'])
            config = {
                'isin': 'ISIN',
                'trend_signal': 'Segnale di Trend',
                'quantity': 'Quantità',
                'avg_buy_price': 'Prezzo Medio Acq. (€)',
                'current_price': 'Prezzo Attuale (€)',
                'invested_amount': 'Investito (€)',
                'current_amount': 'Valore Attuale (€)',
                'unrealized_pnl': 'P&L Non Realizzato (€)',
                'unrealized_pnl_pct': 'P&L Non Realizzato (%)'
            }
            st.dataframe(
                format_dataframe(
                    df_open, config,
                    pnl_cols=['unrealized_pnl', 'unrealized_pnl_pct'],
                    bar_cols=['current_amount'],
                    trend_cols=['trend_signal'],
                ),
                width="stretch",
            )
        else:
            st.info("Nessuna posizione aperta.")

        st.subheader("Storico Complessivo per ISIN")
        if pnl_results['aggregated_positions']:
            df_agg = pd.DataFrame(pnl_results['aggregated_positions'])
            config = {
                'isin': 'ISIN', 'total_bought_qty': 'Tot. Quantità Acq.', 'total_sold_qty': 'Tot. Quantità Vend.',
                'current_qty': 'Quantità Attuale', 'avg_buy_price': 'Prezzo Medio Acq. (€)',
                'realized_pnl': 'P&L Realizzato (€)', 'unrealized_pnl': 'P&L Non Realizzato (€)',
                'total_pnl': 'P&L Totale (€)', 'current_value': 'Valore Attuale (€)'
            }
            st.dataframe(
                format_dataframe(df_agg, config, pnl_cols=['realized_pnl', 'unrealized_pnl', 'total_pnl']),
                width="stretch",
            )
        else:
            st.info("Nessun dato aggregato da mostrare.")

    with tab3:
        st.header("Dettaglio Fiscale e Transazioni")
        st.subheader("Dettaglio Plus/Minusvalenze Realizzate")
        if pnl_results['capital_gains']:
            df_gains = pd.DataFrame(pnl_results['capital_gains'])
            config = {
                'date': 'Data', 'isin': 'ISIN', 'quantity': 'Quantità', 'sale_price': 'Prezzo Vendita (€)',
                'avg_buy_price': 'Prezzo Medio Acq. (€)', 'gross_pnl': 'P&L Lordo (€)',
                'commission': 'Commissioni (€)', 'taxable_amount': 'Imponibile (€)',
                'tax_paid': 'Tasse Pagate (€)', 'net_pnl': 'P&L Netto (€)'
            }
            st.dataframe(
                format_dataframe(df_gains, config, pnl_cols=['gross_pnl', 'taxable_amount', 'tax_paid', 'net_pnl']),
                width="stretch",
            )
        else:
            st.info("Nessuna vendita registrata.")

        with st.expander("📜 Cronologia Completa delle Transazioni"):
            df_trans = pd.DataFrame(portfolio_data["transactions"])
            df_trans_display = df_trans.rename(columns={
                'type': 'Tipo', 'isin': 'ISIN', 'quantity': 'Quantità',
                'price': 'Prezzo (€)', 'date': 'Data'
            })
            st.dataframe(df_trans_display, width="stretch")
