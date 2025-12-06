# -*- coding: utf-8 -*-
from typing import Dict, List
from datetime import datetime
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from etf_metrics.core.metrics import compute_metrics_from_series, compute_sharpe_ratio


def style_pnl_columns(val):
    """Colore verde per valori positivi, rosso per negativi."""
    if isinstance(val, (int, float)):
        color = '#28a745' if val > 0 else '#dc3545' if val < 0 else '#333333'
        return f'color: {color}; font-weight: 600;'
    return ''


def style_trend_signal(val: str) -> str:
    """Applica uno stile colorato alla colonna dei segnali di trend."""
    val_lower = val.lower()
    if "mantieni" in val_lower:
        return 'background-color: #28a745; color: white; font-weight: bold;' # Verde per Mantieni
    elif "monitora" in val_lower:
        return 'background-color: #ffc107; color: black; font-weight: bold;' # Giallo per Monitora
    elif val_lower.startswith("vendi"):
        return 'background-color: #dc3545; color: white; font-weight: bold;' # Rosso per Vendi
    return '' # Default

def format_dataframe(df: pd.DataFrame, column_config: Dict, pnl_cols: List[str] = [], bar_cols: List[str] = [],
                     trend_cols: List[str] = []):
    """Applica formattazione completa a un DataFrame per la visualizzazione."""
    df_display = df.rename(columns=column_config)

    styler = df_display.style

    renamed_pnl_cols = [column_config.get(col) for col in pnl_cols if column_config.get(col) in df_display.columns]
    renamed_bar_cols = [column_config.get(col) for col in bar_cols if column_config.get(col) in df_display.columns]
    renamed_trend_cols = [column_config.get(col) for col in trend_cols if column_config.get(col) in df_display.columns]

    format_dict = {
        name: '€{:,.2f}' for col, name in column_config.items() if '€' in name
    }
    format_dict.update({
        name: '{:+.2f}%' for col, name in column_config.items() if '%' in name
    })
    format_dict.update({
        name: '{:.6f}' for col, name in column_config.items() if 'Quantità' in name
    })
    styler = styler.format(format_dict, na_rep="n.d.")

    if renamed_pnl_cols:
        styler = styler.apply(lambda x: x.map(style_pnl_columns), subset=renamed_pnl_cols)

    if renamed_trend_cols:
        for col_name in renamed_trend_cols:
            styler = styler.map(style_trend_signal, subset=[col_name])

    if renamed_bar_cols:
        for col_name in renamed_bar_cols:
            styler = styler.background_gradient(cmap='viridis_r', subset=[col_name])

    styler = styler.set_properties(**{'text-align': 'right'}).set_properties(
        subset=[column_config.get('isin', 'isin')], **{'text-align': 'left'}
    )

    return styler


def _render_allocation_pie(labels: List[str], values: List[float], title: str, key="alloc_pie"):
    if not labels or not values or sum(values) <= 0:
        st.info("Aggiungi almeno una posizione per vedere l'allocazione.")
        return
    if HAS_PLOTLY:
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo="label+percent", pull=[0.05] * len(labels))]
        )
        fig.update_layout(title_text=title, margin=dict(t=50, b=10, l=10, r=10),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.4))

        st.plotly_chart(
            fig,
            width="stretch",
            key="comparison_chart"
        )

def _display_backtest_results(all_series: Dict[str, pd.Series], rf_ann: float, key_prefix="backtest"):
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
        st.dataframe(df.style.format("{:.2f}", na_rep="n.d."),width="stretch")


def render_portfolio_tracker_ui():
    from etf_metrics.core.portfolio_tracker import PortfolioTracker
    PortfolioTracker.init_session_from_json_once(filename="saved_portfolio.json")

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
    saved_portfolios = PortfolioTracker.get_saved_portfolio_names()
    current_portfolio_name = st.session_state.get("current_portfolio_name", "")
    with st.sidebar.expander("Carica o Crea Portafoglio", expanded=not current_portfolio_name):
        action = st.radio("Azione", ["Carica Portafoglio", "Crea Nuovo Portafoglio"], label_visibility="collapsed")
        if action == "Carica Portafoglio":
            if not saved_portfolios:
                st.warning("Nessun portafoglio salvato.")
            else:
                selected = st.selectbox("Seleziona Portafoglio", saved_portfolios, index=saved_portfolios.index(
                    current_portfolio_name) if current_portfolio_name in saved_portfolios else 0)
                if st.button("Carica"):
                    st.session_state.current_portfolio_name = selected
                    st.rerun()
        else:
            new_name = st.text_input("Nome Nuovo Portafoglio", "Il Mio Portafoglio")
            if st.button("Crea"):
                st.session_state.current_portfolio_name = new_name
                st.session_state.setdefault("_pt_portfolios", {})[new_name] = {"name": new_name, "transactions": []}
                st.rerun()

    if current_portfolio_name:
        st.sidebar.subheader(f"Gestisci: '{current_portfolio_name}'")
        with st.sidebar.expander("➕ Aggiungi Transazione"):
            trans_type = st.radio("Tipo", ["Acquisto", "Vendita"], horizontal=True)
            form_key = "buy_form" if trans_type == "Acquisto" else "sell_form"
            with st.form(form_key):
                isin = st.text_input("ISIN")
                qty = st.number_input("Quantità", min_value=0.000001, step=0.0001, format="%.6f")
                price = st.number_input(f"Prezzo {trans_type} (€)", min_value=0.01, step=0.01, format="%.2f")
                date = st.date_input(f"Data {trans_type}", datetime.now().date())
                is_satellite = st.checkbox("Satellite?", key=f"satellite_flag_{form_key}")
                if st.form_submit_button(f"Registra {trans_type}"):
                    trans_data = {"type": "buy" if trans_type == "Acquisto" else "sell", "isin": isin.strip().upper(),
                                  "quantity": qty, "price": price, "date": date,
                                  "satellite": is_satellite if trans_type == "Acquisto" else False}
                    PortfolioTracker.add_transaction_to_portfolio(current_portfolio_name, trans_data)
                    st.rerun()
        with st.sidebar.expander("⚙️ Impostazioni Fiscali"):
            commission = st.number_input("Commissione per operazione (€)", min_value=0.0, value=1.0, step=0.5,
                                         format="%.2f")
            tax_rate = st.slider("Imposta plusvalenze (%)", 0, 100, 26, 1)

    if not current_portfolio_name:
        st.info("👈 Crea o carica un portafoglio dalla barra laterale per iniziare.")
        return
    portfolio_data = PortfolioTracker.load_portfolio_from_session(current_portfolio_name)
    if not portfolio_data or not portfolio_data.get("transactions"):
        st.info("Questo portafoglio è vuoto. Aggiungi una transazione per iniziare.")
        return
    with st.spinner("Aggiornamento P&L e Segnali di Trend in corso..."):
        pnl_results = PortfolioTracker.calculate_portfolio_pnl(portfolio_data["transactions"], commission, tax_rate)

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
            _render_allocation_pie(labels, values, "")
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
            st.dataframe(format_dataframe(df_open, config,
                                          pnl_cols=['unrealized_pnl', 'unrealized_pnl_pct'],
                                          bar_cols=['current_amount'],
                                          trend_cols=['trend_signal']),
                        width="stretch")
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
            st.dataframe(format_dataframe(df_agg, config,
                                          pnl_cols=['realized_pnl', 'unrealized_pnl', 'total_pnl']),
                        width="stretch")
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
            st.dataframe(format_dataframe(df_gains, config,
                                          pnl_cols=['gross_pnl', 'taxable_amount', 'tax_paid', 'net_pnl']),
                        width="stretch")
        else:
            st.info("Nessuna vendita registrata.")

        with st.expander("📜 Cronologia Completa delle Transazioni"):
            df_trans = pd.DataFrame(portfolio_data["transactions"])
            df_trans_display = df_trans.rename(columns={
                'type': 'Tipo', 'isin': 'ISIN', 'quantity': 'Quantità',
                'price': 'Prezzo (€)', 'date': 'Data'
            })
            st.dataframe(df_trans_display,width="stretch")