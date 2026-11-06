# -*- coding: utf-8 -*-
from datetime import datetime

import pandas as pd
import streamlit as st

from etf_metrics.core.portfolio_tracker import (
    calculate_portfolio_pnl,
    transaction_tickers,
    resolve_transaction_ticker,
)
from etf_metrics.core.data_updater import update_tickers
from etf_metrics.shared.config import PORTFOLIO_FILE
from etf_metrics.ui.components import (
    format_dataframe,
    render_allocation_pie,
    render_page_help,
    get_portfolio_store,
)


def _render_db_update_report(report: dict) -> None:
    """Esito compatto dell'aggiornamento del DB locale (da Carica / Registra Acquisto)."""
    results = report.get('results') or {}
    skipped = report.get('skipped') or {}
    new_rows = sum(r.get('new_rows', 0) for r in results.values())
    if results:
        msg = (f"🗄️ DB locale aggiornato: {len(results)} ticker sincronizzati "
               f"({new_rows:,} righe nuove)")
        if skipped:
            msg += f", {len(skipped)} saltati"
        st.success(msg + ".")
        with st.expander("Dettaglio aggiornamento DB"):
            rows = [{
                "Ticker": t,
                "Modalità": "⚡ Delta" if r.get('mode') == 'delta' else "🔄 Completo",
                "Righe nuove": r.get('new_rows', 0),
                "Righe totali": r.get('total_rows', 0),
            } for t, r in sorted(results.items())]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    elif skipped:
        reasons = ", ".join(sorted(set(skipped.values())))
        st.info(f"🗄️ DB locale già sincronizzato ({len(skipped)} saltati: {reasons}).")
    if report.get('rate_limited'):
        st.warning("Rate limit Yahoo: il download è stato interrotto. "
                   "I dati già scaricati sono comunque stati salvati nel DB.")


def render_portfolio_tracker_ui():
    """Pagina 💼 Portfolio Tracker: posizioni reali, P&L e cassetto fiscale.

    All'avvio persiste su file eventuali modifiche in memoria (safety net
    al primo avvio). Il bottone 📥 Carica aggiorna il DB locale con i
    ticker del portafoglio: ticker nuovo -> storico completo (max), ticker
    esistente -> delta dall'ultima data nel DB (anche se già aggiornato
    oggi). La registrazione di un acquisto fa lo stesso per il ticker
    dell'operazione (se risolvibile). Versamenti/prelievi: feature rimossa,
    mapping mantenuto solo per visualizzare gli storici già registrati nei
    portafogli esistenti.
    """
    store = get_portfolio_store()
    store.save_to_json(PORTFOLIO_FILE)

    st.title("💼 Portfolio Tracker")
    st.caption("Analizza le tue posizioni, monitora le performance e gestisci le transazioni.")

    render_page_help(
        "**Cosa fa**: registra acquisti/vendite e calcola valore attuale, P&L e cassetto fiscale.",
        "**A cosa serve**: monitorare il portafoglio reale (es. Trade Republic) in tempo reale.",
        "I prezzi sono live da Yahoo; il bottone 📥 **Carica** aggiorna anche lo storico nel DB.",
        "La checkbox 🛰️ **satellite** segnala gli acquisti tattici (usata da 🧭 Satelliti PAC).",
    )

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
                    transactions = store.get(selected).get("transactions", [])
                    tickers = transaction_tickers(transactions)
                    if tickers:
                        with st.spinner(f"Aggiornamento DB: {len(tickers)} ticker..."):
                            st.session_state.db_update_report = update_tickers(
                                tickers, period="max")
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
                isin = st.text_input(
                    "ISIN o Ticker",
                    help=("ISIN (es. IE00BK5BQT80) oppure ticker Yahoo diretto "
                          "(es. VWCE.MI, SAP.DE)."))
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
                    if trans_type == "Acquisto":
                        ticker = resolve_transaction_ticker(isin)
                        if ticker:
                            with st.spinner(f"Aggiornamento DB: {ticker}..."):
                                st.session_state.db_update_report = update_tickers(
                                    [ticker], period="max")
                        else:
                            st.session_state.db_update_report = {
                                'requested': 1, 'results': {},
                                'skipped': {isin.strip().upper():
                                            'ISIN/ticker non risolvibile su Yahoo'},
                                'rate_limited': False, 'elapsed_sec': 0.0}
                    st.rerun()
        with st.sidebar.expander("⚙️ Impostazioni Fiscali"):
            commission = st.number_input("Commissione per operazione (€)", min_value=0.0, value=1.0, step=0.5, format="%.2f")
            tax_rate = st.slider("Imposta plusvalenze (%)", 0, 100, 26, 1)
            carryforward_years = st.slider(
                "Anni recupero minusvalenze", 0, 8, 4,
                help="Art. 68 c.9 TUIR: le minusvalenze compensano le plusvalenze dei "
                     "4 periodi d'imposta successivi (default fiscale italiano). "
                     "0 = compensazione solo nello stesso anno.")

    if not current_portfolio_name:
        st.info("👈 Crea o carica un portafoglio dalla barra laterale per iniziare.")
        return

    report = st.session_state.pop("db_update_report", None)
    if report:
        _render_db_update_report(report)

    portfolio_data = store.get(current_portfolio_name)
    if not portfolio_data or not portfolio_data.get("transactions"):
        st.info("Questo portafoglio è vuoto. Aggiungi una transazione per iniziare.")
        return

    with st.spinner("Aggiornamento P&L in corso..."):
        pnl_results = calculate_portfolio_pnl(portfolio_data["transactions"], commission, tax_rate,
                                              carryforward_years=carryforward_years)

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
        st.subheader("Posizioni Aperte (collegate a ⚡ Trading)")
        if pnl_results['open_positions']:
            df_open = pd.DataFrame(pnl_results['open_positions'])
            config = {
                'isin': 'ISIN',
                'ticker': 'Ticker',
                'name': 'Nome',
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
                ),
                width="stretch",
            )
        else:
            st.info("Nessuna posizione aperta.")

        st.subheader("Storico Complessivo per ISIN")
        if pnl_results['aggregated_positions']:
            df_agg = pd.DataFrame(pnl_results['aggregated_positions'])
            config = {
                'isin': 'ISIN', 'ticker': 'Ticker', 'name': 'Nome',
                'total_bought_qty': 'Tot. Quantità Acq.', 'total_sold_qty': 'Tot. Quantità Vend.',
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
        st.subheader("🧾 Recupero Minusvalenze (art. 68 TUIR)")
        tp = pnl_results.get('tax_position') or {}
        if tp and (tp.get('gains_realized') or tp.get('losses_realized')):
            cf = tp.get('carryforward_years')
            cf_txt = f"{cf} anni" if cf is not None else "illimitato"
            col1, col2, col3 = st.columns(3)
            col1.metric("Tasse (ritenuta per vendita)", f"€{tp['tax_withheld']:,.2f}",
                        help="Somma dell'imposta calcolata su ogni singola vendita in utile, "
                             "senza compensazione (modello alla fonte).")
            col2.metric("Imposta dovuta (con recupero)", f"€{tp['tax_due']:,.2f}",
                        help=f"Imposta con compensazione delle minusvalenze (recupero {cf_txt}): "
                             "quella che emerge dal modello Redditi.")
            col3.metric("Recuperabile in dichiarazione", f"€{tp['tax_recoverable']:,.2f}",
                        delta=f"-{tp['tax_recoverable'] / max(tp['tax_withheld'], 0.01) * 100:.0f}% di tasse",
                        delta_color="inverse",
                        help="Eccesso di ritenuta rispetto all'imposta dovuta: si recupera "
                             "in dichiarazione dei redditi.")
            residual = tp.get('residual_credit') or 0.0
            if residual > 0.005:
                st.success(f"📉 Minusvalenze residue recuperabili: **€{residual:,.2f}** — "
                           "compensano le prossime plusvalenze (tasse ridotte).")
                exp_rows = [{
                    "Scadenza": (f"31/12/{b['expires']}" if b.get('expires') is not None else "mai (illimitato)"),
                    "Importo (€)": round(b['amount'], 2)
                } for b in (tp.get('credit_by_expiry') or [])]
                if exp_rows:
                    st.dataframe(pd.DataFrame(exp_rows), width="stretch", hide_index=True)
                st.caption("Le minusvalenze si usano entro il 4° periodo d'imposta successivo: "
                           "una perdita del 2022 copre fino al 2026, poi decade.")
            else:
                st.info("Nessuna minusvalenza residua: le prossime plusvalenze pagano "
                        f"l'imposta piena ({tp.get('tax_rate', 26)}%).")
            if tp.get('losses_expired', 0) > 0.005:
                st.warning(f"⏳ Minusvalenze perse per scadenza: **€{tp['losses_expired']:,.2f}** "
                           "(non più compensabili).")
            st.caption("Semplificazioni: aliquota unica per strumento (la quota al 12,5% dei "
                       "titoli di Stato non è distinta), commissioni di vendita dedotte dalla "
                       "base imponibile, vendite senza storico d'acquisto escluse.")
        else:
            st.info("Nessuna vendita registrata: niente da compensare.")

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
            type_map = {"buy": "Acquisto", "sell": "Vendita",
                        "deposit": "Versamento", "withdrawal": "Prelievo"}
            df_trans["type"] = df_trans["type"].map(lambda v: type_map.get(v, v))
            df_trans_display = df_trans.rename(columns={
                'type': 'Tipo', 'isin': 'ISIN', 'quantity': 'Quantità',
                'price': 'Prezzo (€)', 'date': 'Data', 'amount': 'Importo (€)',
                'strategy': 'Strategia'
            })
            st.dataframe(df_trans_display, width="stretch")
