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
    st.warning("⚠️ Plotly non disponibile - alcuni grafici saranno disabilitati")

from .utils import to_percent_index
from .metrics import compute_metrics_from_series, compute_sharpe_ratio
from .config import FAMOUS_PORTFOLIOS


def _seed_sidebar_from_saved_portfolio(portfolio_data, prefix="sb_"):
    """Prefilla i campi della sidebar dai dati di un portafoglio salvato."""
    if not portfolio_data or not portfolio_data.get("holdings"):
        return
    st.session_state.setdefault(f"{prefix}name", portfolio_data.get("name", "Il Mio Portafoglio"))
    st.session_state.setdefault(f"{prefix}npos", len(portfolio_data["holdings"]))
    npos = len(portfolio_data["holdings"])
    for i, h in enumerate(portfolio_data["holdings"][:npos]):
        st.session_state[f"{prefix}isin_{i}"] = (h.get("isin") or "").strip().upper()
        st.session_state[f"{prefix}qty_{i}"] = float(h.get("quantity", 0.0))
        st.session_state[f"{prefix}price_{i}"] = float(h.get("purchase_price", 0.0))
        d = h.get("purchase_date")
        if isinstance(d, str):
            try:
                d = pd.to_datetime(d).date()
            except Exception:
                d = datetime.now().date()
        elif not d:
            d = datetime.now().date()
        st.session_state[f"{prefix}date_{i}"] = d


def _build_weights_text_from_holdings_detail(holdings_detail, base="invested"):
    """
    Converte le posizioni del tracker (results['holdings_detail']) in stringa 'ISIN: peso'
    con pesi (%) su: invested/current/quantity.
    """
    if not holdings_detail:
        return ""
    rows, total = [], 0.0
    for h in holdings_detail:
        isin = (h.get("isin") or "").strip().upper()
        if not isin:
            continue
        if base == "current":
            amount = float(h.get("current_amount", 0.0))
        elif base == "quantity":
            amount = float(h.get("quantity", 0.0)) * float(h.get("purchase_price", 0.0))
        else:
            amount = float(h.get("invested_amount", 0.0))
        if amount > 0:
            rows.append((isin, amount))
            total += amount
    if total <= 0:
        return ""
    lines = []
    for isin, amount in rows:
        w = round(100.0 * amount / total, 2)
        lines.append(f"{isin}: {w}")
    return "\n".join(lines)


def _display_backtest_results(all_series: Dict[str, pd.Series], rf_ann: float, key_prefix="backtest"):
    """Grafico normalizzato base 100 e tabella metriche (CAGR, Vol, MDD, Sharpe)"""
    if not all_series:
        st.error("Nessun dato da visualizzare. Controlla gli ISIN e la loro storicità.")
        return
    period_map = {"1M": 1, "3M": 3, "6M": 6, "YTD": "ytd", "1A": 12, "3A": 36, "5A": 60, "Max": None}
    selected_period_label = st.radio(
        "Seleziona periodo di analisi",
        list(period_map.keys()),
        index=len(period_map) - 1,
        horizontal=True,
        key=f"{key_prefix}_period_radio"
    )
    lookback_months = period_map[selected_period_label]
    filtered_series_dict: Dict[str, pd.Series] = {}
    metrics_list: List[Dict] = []

    end_date = pd.to_datetime('today').normalize()
    if lookback_months == "ytd":
        start_date = pd.to_datetime(f"{end_date.year}-01-01")
    elif lookback_months is not None:
        start_date = end_date - pd.DateOffset(months=lookback_months)
    else:
        start_date = min(s.index.min() for s in all_series.values())

    for name, series in all_series.items():
        filtered_series = series[series.index >= start_date]
        if filtered_series.shape[0] < 2:
            continue
        filtered_series_dict[name] = filtered_series
        metrics = compute_metrics_from_series(filtered_series)
        metrics["sharpe"] = compute_sharpe_ratio(filtered_series, rf_ann)
        metrics["name"] = name
        metrics_list.append(metrics)

    st.subheader(f"Andamento Portafogli ({selected_period_label})")
    if HAS_PLOTLY:
        fig = go.Figure()
        for name, series in filtered_series_dict.items():
            norm_series = to_percent_index(series) + 100  # base 100
            fig.add_trace(go.Scatter(x=norm_series.index, y=norm_series.values, mode='lines', name=name))
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")
    else:
        st.line_chart({name: to_percent_index(s) + 100 for name, s in filtered_series_dict.items()})

    st.subheader(f"Metriche di Performance ({selected_period_label})")
    if metrics_list:
        df = pd.DataFrame(metrics_list).set_index("name")
        df_view = df[["cagr", "vol_ann", "mdd", "sharpe"]].rename(columns={
            "cagr": "CAGR %", "vol_ann": "Volatilità Ann. %", "mdd": "Max Drawdown %", "sharpe": "Sharpe Ratio"
        })
        st.dataframe(df_view.style.format("{:.2f}", na_rep="n.d."), use_container_width=True)
    else:
        st.info("Non ci sono abbastanza dati nel periodo selezionato per calcolare le metriche.")


def _render_allocation_pie(labels: List[str], values: List[float], title: str, key="alloc_pie"):
    """Pie chart delle allocazioni. labels: ISIN, values: importi."""
    if not labels or not values or sum(values) <= 0:
        st.info("Aggiungi almeno una posizione per vedere l'allocazione.")
        return
    if HAS_PLOTLY:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.35,
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>Quota: %{percent}<br>Valore: €%{value:,.2f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=title,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.warning("Plotly non disponibile: grafico a torta non mostrato.")


def render_portfolio_tracker_ui():
    from .portfolio_tracker import PortfolioTracker

    # Inizializzazione e caricamento
    PortfolioTracker.init_session_from_json_once(filename="saved_portfolio.json")

    # --- Sidebar ---
    st.sidebar.header("📂 Portafoglio")

    saved_portfolios = PortfolioTracker.get_saved_portfolio_names()
    selected_portfolio = None
    if saved_portfolios:
        selected_portfolio = st.sidebar.selectbox(
            "Carica portafoglio salvato:", ["Nuovo..."] + saved_portfolios, key="sb_saved_select"
        )
    if selected_portfolio != "Nuovo...":
        portfolio_data = PortfolioTracker.load_portfolio_from_session(selected_portfolio)
        if portfolio_data:
            st.session_state.current_portfolio = portfolio_data
            _seed_sidebar_from_saved_portfolio(portfolio_data, prefix="sb_")

    st.sidebar.subheader("📝 Il Tuo Portafoglio")
    portfolio_name = st.sidebar.text_input("Nome Portafoglio:", value=st.session_state.get("sb_name", "Il Mio Portafoglio"), key="sb_name")
    num_positions = st.sidebar.number_input("Numero di posizioni:", min_value=1, max_value=20, value=st.session_state.get("sb_npos", 1), key="sb_npos")

    holdings: List[Dict] = []
    for i in range(int(num_positions)):
        st.sidebar.markdown(f"**Posizione {i+1}:**")
        isin = st.sidebar.text_input(f"ISIN {i+1}:", key=f"sb_isin_{i}", value=st.session_state.get(f"sb_isin_{i}", ""))
        quantity = st.sidebar.number_input(f"Quantità {i+1}:", min_value=0.0, step=0.1, key=f"sb_qty_{i}", value=st.session_state.get(f"sb_qty_{i}", 0.0))
        purchase_price = st.sidebar.number_input(
            f"Prezzo Acquisto € {i+1}:", min_value=0.0, step=0.01, key=f"sb_price_{i}", format="%.2f",
            value=st.session_state.get(f"sb_price_{i}", 0.0)
        )
        purchase_date = st.sidebar.date_input(
            f"Data Acquisto {i+1}:", key=f"sb_date_{i}", value=st.session_state.get(f"sb_date_{i}", datetime.now().date())
        )
        if isin and quantity > 0 and purchase_price > 0:
            holdings.append({
                "isin": isin.strip().upper(),
                "quantity": quantity,
                "purchase_price": purchase_price,
                "purchase_date": purchase_date,
            })

    if st.sidebar.button("💾 Salva Portafoglio", disabled=not holdings, key="sb_save_btn"):
        portfolio_data = {
            "name": portfolio_name,
            "holdings": holdings,
            "created_date": datetime.now().date(),
        }
        PortfolioTracker.save_portfolio_to_session(portfolio_data)
        PortfolioTracker.save_portfolio_to_json(portfolio_data)
        st.sidebar.success(f"✅ Portafoglio '{portfolio_name}' salvato su sessione e file!")

    if st.sidebar.button("📊 Calcola P&L", disabled=not holdings, key="sb_calc_btn"):
        with st.spinner("Calcolo P&L in corso..."):
            pnl_results = PortfolioTracker.calculate_portfolio_pnl(holdings)
            st.session_state.pnl_results = pnl_results
        st.success("Calcolo P&L completato!")

    # --- Pagina principale ---
    st.title("💼 Portfolio Tracker")
    st.caption("Riepilogo e analisi del tuo portafoglio")

    # Riepilogo con torta (allocazione)
    st.subheader("🧭 Riepilogo Portafoglio")
    if hasattr(st.session_state, 'pnl_results') and st.session_state.pnl_results and st.session_state.pnl_results.get('holdings_detail'):
        hd = st.session_state.pnl_results['holdings_detail']
        labels = [h['isin'] for h in hd]
        values = [float(h.get('current_amount', 0.0)) for h in hd]
        _render_allocation_pie(labels, values, "Allocazione attuale (%)", key="alloc_current")
    else:
        if holdings:
            labels = [h['isin'] for h in holdings]
            values = [float(h['quantity']) * float(h['purchase_price']) for h in holdings]
            _render_allocation_pie(labels, values, "Allocazione per Investito (%)", key="alloc_invested")
        else:
            st.info("Definisci il tuo portafoglio nella barra di sinistra per vedere il riepilogo.")

    # Se P&L calcolato: mostra i dettagli
    if hasattr(st.session_state, 'pnl_results'):
        display_pnl_results()


def display_pnl_results():
    """KPI principali, tabella dettagliata, performance vs benchmark e (facoltativo) backtest."""
    from .portfolio_tracker import PortfolioTracker

    results = st.session_state.pnl_results
    st.subheader("📈 Risultati P&L")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Investito", f"€{results['total_invested']:,.2f}")
    with col2:
        st.metric("Valore Attuale", f"€{results['current_value']:,.2f}")
    with col3:
        pnl_color = "normal" if results['total_pnl'] >= 0 else "inverse"
        st.metric("P&L Totale", f"€{results['total_pnl']:,.2f}", delta_color=pnl_color)
    with col4:
        pnl_pct_color = "normal" if results['total_pnl_pct'] >= 0 else "inverse"
        st.metric("P&L %", f"{results['total_pnl_pct']:+.2f}%", delta_color=pnl_pct_color)

    if results.get('holdings_detail'):
        st.subheader("📋 Dettaglio Posizioni")
        df_holdings = pd.DataFrame(results['holdings_detail'])
        df_display = df_holdings[
            ['isin', 'quantity', 'purchase_price', 'current_price',
             'invested_amount', 'current_amount', 'pnl', 'pnl_pct', 'days_held']
        ].copy()
        df_display.columns = [
            'ISIN', 'Quantità', 'Prezzo Acquisto €', 'Prezzo Attuale €',
            'Investito €', 'Valore Attuale €', 'P&L €', 'P&L %', 'Giorni'
        ]
        st.dataframe(
            df_display.style.format({
                'Prezzo Acquisto €': '€{:.2f}',
                'Prezzo Attuale €': '€{:.2f}',
                'Investito €': '€{:,.2f}',
                'Valore Attuale €': '€{:,.2f}',
                'P&L €': '€{:+,.2f}',
                'P&L %': '{:+.2f}%'
            }),
            use_container_width=True
        )

    # Performance vs benchmark
    if results.get('holdings_detail'):
        st.subheader("🏆 Performance vs Benchmark")
        benchmark_results = PortfolioTracker.get_performance_vs_benchmark(
            [
                {
                    'isin': h['isin'], 'quantity': h['quantity'],
                    'purchase_price': h['purchase_price'], 'purchase_date': h['purchase_date']
                }
                for h in results['holdings_detail']
            ]
        )
        if 'error' not in benchmark_results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Performance", f"{benchmark_results.get('portfolio_performance', 0):.2f}%")
            with col2:
                st.metric("S&P 500 Performance", f"{benchmark_results.get('benchmark_performance', 0):.2f}%")
            with col3:
                outperf = benchmark_results.get('outperformance', 0)
                st.metric("Outperformance", f"{outperf:+.2f}%", delta_color="normal" if outperf >= 0 else "inverse")
        else:
            st.warning(benchmark_results.get('error', 'Impossibile calcolare il benchmark.'))

    # Backtest su queste posizioni (expander)
    if results.get('holdings_detail'):
        with st.expander("🔙 Backtest su queste posizioni", expanded=False):
            base_label = st.radio("Base per i pesi", ["Investito", "Valore attuale", "Quantità"], index=0, horizontal=True, key="tracker_bt_base")
            base_map = {"Investito": "invested", "Valore attuale": "current", "Quantità": "quantity"}
            base_key = base_map[base_label]

            strategy = st.radio("Strategia", ["Lump Sum", "PAC"], index=0, horizontal=True, key="tracker_bt_strategy")
            monthly_investment = 0
            if strategy == "PAC":
                monthly_investment = st.number_input("Investimento mensile (€)", min_value=100, max_value=10000, value=1000, step=100, key="tracker_bt_pac_amount")

            try:
                famous_all = list(FAMOUS_PORTFOLIOS.keys())
            except Exception:
                famous_all = []
            famous_selection = st.multiselect(
                "Confronta con portafogli modello",
                famous_all,
                default=(["Classic 60/<40"] if "Classic 60/40" in famous_all else []),
                key="tracker_bt_famous_selection"
            )

            rf_ann_backtest = st.number_input("Risk-free annuo (%) per Sharpe", min_value=-5.0, max_value=10.0, value=1.0, step=0.25, key="tracker_bt_rf_ann")

            if st.button("▶️ Esegui Backtest su questo portafoglio", key="tracker_bt_run_btn"):
                portfolio_text = _build_weights_text_from_holdings_detail(results['holdings_detail'], base=base_key)
                if not portfolio_text:
                    st.error("Impossibile costruire i pesi dalle posizioni correnti.")
                else:
                    from .portfolio_backtester import parse_portfolio_input, get_all_portfolios_for_backtest
                    user_portfolio_def = parse_portfolio_input(portfolio_text)
                    if user_portfolio_def:
                        with st.spinner("Esecuzione backtest..."):
                            all_series = get_all_portfolios_for_backtest(
                                user_portfolio_def,
                                famous_selection,
                                FAMOUS_PORTFOLIOS if famous_all else {},
                                strategy.lower().replace(" ", "_"),
                                monthly_investment
                            )
                            st.session_state.tracker_backtest_results = {'series': all_series, 'rf_ann': rf_ann_backtest}
                        st.success("Backtest completato!")
                    else:
                        st.error("Input generato non valido. Controlla le posizioni.")

    tracker_bt = st.session_state.get('tracker_backtest_results')
    if tracker_bt and tracker_bt.get('series'):
        st.markdown("---")
        st.subheader("📊 Backtest (portafoglio corrente)")
        _display_backtest_results(tracker_bt['series'], tracker_bt['rf_ann'], key_prefix="tracker_bt")