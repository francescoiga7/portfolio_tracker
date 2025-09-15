# -*- coding: utf-8 -*-
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import pandas as pd
import streamlit as st

# Plotly opzionale
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from .utils import to_percent_index
from .metrics import compute_metrics_from_series, compute_sharpe_ratio


# Funzioni helper per la UI
def _render_allocation_pie(labels: List[str], values: List[float], title: str, key="alloc_pie"):
    """Pie chart delle allocazioni."""
    if not labels or not values or sum(values) <= 0:
        st.info("Aggiungi almeno una posizione per vedere l'allocazione.")
        return
    if HAS_PLOTLY:
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.35, textinfo="label+percent")]
        )
        fig.update_layout(title_text=title, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True, key=key)


def _display_backtest_results(all_series: Dict[str, pd.Series], rf_ann: float, key_prefix="backtest"):
    """Grafico normalizzato e tabella metriche per il backtester."""
    if not all_series:
        st.error("Nessun dato da visualizzare.")
        return

    period_map = {"1M": 1, "3M": 3, "6M": 6, "YTD": "ytd", "1A": 12, "3A": 36, "5A": 60, "Max": None}
    selected_period_label = st.radio(
        "Seleziona periodo di analisi", list(period_map.keys()), index=len(period_map) - 1, horizontal=True,
        key=f"{key_prefix}_period_radio"
    )
    lookback_months = period_map[selected_period_label]

    end_date = pd.to_datetime('today').normalize()
    if lookback_months == "ytd":
        start_date = pd.to_datetime(f"{end_date.year}-01-01")
    elif lookback_months is not None:
        start_date = end_date - pd.DateOffset(months=lookback_months)
    else:
        start_date = min(s.index.min() for s in all_series.values() if s is not None and not s.empty)

    metrics_list, series_to_plot = [], {}
    for name, series in all_series.items():
        if series is None or series.empty: continue
        filtered = series[series.index >= start_date]
        if filtered.shape[0] < 2: continue

        series_to_plot[name] = filtered
        metrics = compute_metrics_from_series(filtered)
        metrics["sharpe"] = compute_sharpe_ratio(filtered, rf_ann)
        metrics["name"] = name
        metrics_list.append(metrics)

    st.subheader(f"Andamento Portafogli ({selected_period_label})")
    st.line_chart(series_to_plot)

    st.subheader(f"Metriche di Performance ({selected_period_label})")
    if metrics_list:
        df = pd.DataFrame(metrics_list).set_index("name")[["cagr", "vol_ann", "mdd", "sharpe"]]
        df.columns = ["CAGR %", "Volatilità Ann. %", "Max Drawdown %", "Sharpe Ratio"]
        st.dataframe(df.style.format("{:.2f}", na_rep="n.d."), use_container_width=True)


# UI Principale del Tracker
def render_portfolio_tracker_ui():
    from .portfolio_tracker import PortfolioTracker
    PortfolioTracker.init_session_from_json_once(filename="saved_portfolio.json")

    st.title("💼 Portfolio Tracker")
    st.caption("Analisi delle tue posizioni e storico transazioni.")

    # --- Sidebar per gestione portafoglio ---
    st.sidebar.header("📂 I Tuoi Portafogli")

    saved_portfolios = PortfolioTracker.get_saved_portfolio_names()

    # Logica per creare o caricare un portafoglio
    action = st.sidebar.radio("Azione", ["Carica Portafoglio", "Crea Nuovo Portafoglio"])

    current_portfolio_name = st.session_state.get("current_portfolio_name", "")

    if action == "Carica Portafoglio":
        if not saved_portfolios:
            st.sidebar.warning("Nessun portafoglio salvato. Creane uno nuovo.")
            st.session_state.current_portfolio_name = ""
        else:
            selected = st.sidebar.selectbox("Seleziona Portafoglio", saved_portfolios)
            if st.sidebar.button("Carica"):
                st.session_state.current_portfolio_name = selected
                st.rerun()
    else:  # Crea Nuovo
        new_name = st.sidebar.text_input("Nome Nuovo Portafoglio", "Il Mio Portafoglio")
        if st.sidebar.button("Crea"):
            st.session_state.current_portfolio_name = new_name
            st.session_state.setdefault("_pt_portfolios", {})[new_name] = {"name": new_name, "transactions": []}
            st.rerun()

    # Se un portafoglio è caricato, mostra le opzioni di transazione
    if current_portfolio_name:
        st.sidebar.subheader(f"Gestisci '{current_portfolio_name}'")

        with st.sidebar.expander("➕ Aggiungi Acquisto"):
            with st.form("buy_form"):
                buy_isin = st.text_input("ISIN")
                buy_qty = st.number_input("Quantità", min_value=0.00, step=0.1)
                buy_price = st.number_input("Prezzo Acquisto (€)", min_value=0.01, step=0.50, format="%.2f")
                buy_date = st.date_input("Data Acquisto", datetime.now().date())
                if st.form_submit_button("Registra Acquisto"):
                    trans = {"type": "buy", "isin": buy_isin.strip().upper(), "quantity": buy_qty, "price": buy_price,
                             "date": buy_date}
                    PortfolioTracker.add_transaction_to_portfolio(current_portfolio_name, trans)
                    st.rerun()

        with st.sidebar.expander("➖ Registra Vendita"):
            with st.form("sell_form"):
                sell_isin = st.text_input("ISIN ")
                sell_qty = st.number_input("Quantità ", min_value=0.01, step=0.1)
                sell_price = st.number_input("Prezzo Vendita (€)", min_value=0.01, step=0.01, format="%.2f")
                sell_date = st.date_input("Data Vendita", datetime.now().date())
                if st.form_submit_button("Registra Vendita"):
                    trans = {"type": "sell", "isin": sell_isin.strip().upper(), "quantity": sell_qty,
                             "price": sell_price, "date": sell_date}
                    PortfolioTracker.add_transaction_to_portfolio(current_portfolio_name, trans)
                    st.rerun()

        st.sidebar.subheader("Impostazioni Fiscali")
        commission = st.sidebar.number_input("Commissione di vendita (€)", min_value=0.0, value=1.0, step=0.5,
                                             format="%.2f")
        tax_rate = st.sidebar.slider("Imposta sulle plusvalenze (%)", min_value=0, max_value=100, value=26, step=1)

    # --- Pagina principale ---
    if not current_portfolio_name:
        st.info("Crea o carica un portafoglio dalla barra laterale per iniziare.")
        return

    portfolio_data = PortfolioTracker.load_portfolio_from_session(current_portfolio_name)
    if not portfolio_data or not portfolio_data.get("transactions"):
        st.info("Questo portafoglio è vuoto. Aggiungi una transazione di acquisto.")
        return

    # Calcola e mostra risultati P&L
    with st.spinner("Calcolo P&L in corso..."):
        pnl_results = PortfolioTracker.calculate_portfolio_pnl(portfolio_data["transactions"], commission, tax_rate)

    st.subheader("🧭 Riepilogo Portafoglio")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Investito Netto", f"€{pnl_results['net_invested']:,.2f}")
    col2.metric("Valore Attuale", f"€{pnl_results['current_value']:,.2f}")
    col3.metric("P&L Realizzato", f"€{pnl_results['realized_pnl']:,.2f}")
    col4.metric("P&L Totale", f"€{pnl_results['total_pnl']:,.2f}", f"{pnl_results['total_pnl_pct']:.2f}%")

    # Grafico allocazione
    if pnl_results['open_positions']:
        labels = [p['isin'] for p in pnl_results['open_positions']]
        values = [p['current_amount'] for p in pnl_results['open_positions']]
        _render_allocation_pie(labels, values, "Allocazione Attuale (%)")

    # Tabella posizioni aperte
    st.subheader("Portafoglio Live: Posizioni Aperte")
    if pnl_results['open_positions']:
        df_open = pd.DataFrame(pnl_results['open_positions'])
        st.dataframe(df_open[['isin', 'quantity', 'avg_buy_price', 'current_price', 'invested_amount', 'current_amount',
                              'unrealized_pnl', 'unrealized_pnl_pct']].style.format({
            'avg_buy_price': '€{:.2f}', 'current_price': '€{:.2f}',
            'invested_amount': '€{:,.2f}', 'current_amount': '€{:,.2f}',
            'unrealized_pnl': '€{:+,.2f}', 'unrealized_pnl_pct': '{:+.2f}%'
        }), use_container_width=True)
    else:
        st.info("Nessuna posizione aperta.")

    # Tabella aggregata per ISIN
    st.subheader("Analisi per ISIN: La Storia Completa")
    if pnl_results['aggregated_positions']:
        df_agg = pd.DataFrame(pnl_results['aggregated_positions'])
        st.dataframe(df_agg[['isin', 'total_bought_qty', 'total_sold_qty', 'current_qty', 'avg_buy_price',
                             'realized_pnl', 'unrealized_pnl', 'total_pnl', 'current_value']].style.format({
            'avg_buy_price': '€{:.2f}', 'realized_pnl': '€{:+,.2f}',
            'unrealized_pnl': '€{:+,.2f}', 'total_pnl': '€{:+,.2f}',
            'current_value': '€{:,.2f}'
        }), use_container_width=True)
    else:
        st.info("Nessun dato aggregato da mostrare.")

    # Tabella plusvalenze/minusvalenze
    st.subheader("Cassetto Fiscale: Plus e Minusvalenze")
    if pnl_results['capital_gains']:
        df_gains = pd.DataFrame(pnl_results['capital_gains'])
        st.dataframe(df_gains[
            ['date', 'isin', 'quantity', 'sale_price', 'avg_buy_price', 'gross_pnl', 'commission', 'taxable_amount',
             'tax_paid', 'net_pnl']].style.format({
            'sale_price': '€{:.2f}', 'avg_buy_price': '€{:.2f}',
            'gross_pnl': '€{:+,.2f}', 'commission': '€{:.2f}',
            'taxable_amount': '€{:.2f}', 'tax_paid': '€{:.2f}',
            'net_pnl': '€{:+,.2f}'
        }), use_container_width=True)
    else:
        st.info("Nessuna vendita registrata.")

    # Tabella transazioni
    with st.expander("📜 Cronologia Transazioni"):
        df_trans = pd.DataFrame(portfolio_data["transactions"])
        st.dataframe(df_trans, use_container_width=True)