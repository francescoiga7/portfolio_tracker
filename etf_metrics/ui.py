# -*- coding: utf-8 -*-
from typing import Optional, Tuple
from datetime import datetime
import pandas as pd
import streamlit as st
import re

# Import condizionali per plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("⚠️ Plotly non disponibile - i grafici saranno disabilitati")

from .config import DEFAULT_CSV_PATH, PERIODS_ALL, FAMOUS_PORTFOLIOS
from .pipeline import compute_etf_over_periods
from .metrics import compute_metrics_from_series, compute_sharpe_ratio
from .utils import to_percent_index
from .yahoo_client import get_series, resolve_isin_one
from .portfolio_backtester import parse_portfolio_input, get_all_portfolios_for_backtest
from .momentum import calculate_momentum_rankings
from .etf_search_engine import search_etfs_universal, format_results_for_display


# --- GESTIONE SESSION STATE ---
def init_session_state():
    defaults = {
        'data_out': None,
        'portfolio_results': None,
        'momentum_results': None,
        'etf_search_results': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_app():
    init_session_state()
    st.set_page_config(page_title="ETF & Portfolio Analysis", page_icon="📊", layout="wide")

    # --- SIDEBAR ---
    st.sidebar.title("Strumenti di Analisi 📈")
    app_mode = st.sidebar.selectbox("Scegli modalità",
                                    ["Analisi Singolo ETF",
                                     "Verifica Momentum", "Ricerca ETF",
                                     "Portfolio Tracker"])

    if app_mode == "Analisi Singolo ETF":
        render_single_etf_ui()
    elif app_mode == "Verifica Momentum":
        render_momentum_ui()
    elif app_mode == "Ricerca ETF":
        render_etf_search_ui()
    elif app_mode == "Portfolio Tracker":
        render_portfolio_tracker_ui()

# --- MODALITÀ: ANALISI SINGOLO ETF ---
def render_single_etf_ui():
    st.sidebar.header("Impostazioni Analisi")
    isin = st.sidebar.text_input("ISIN", value="IE00BK5BQT80").strip().upper()
    # Periodi sempre tutti - rimosso il selettore
    periods = PERIODS_ALL  # Usa tutti i periodi disponibili
    bench_override = st.sidebar.text_input(
        "Override Benchmark (Ticker Yahoo)",
        "",
        help="Inserisci un ticker Yahoo per sovrascrivere il benchmark. " +
             "Se l'ISIN non è mappato, verrà aggiunto automaticamente alla configurazione."
    ).strip() or None
    rf_ann = st.sidebar.number_input("Risk-free annuo (%)", -5.0, 10.0, 1.0, 0.25)
    compare_input = st.sidebar.text_area("Confronta con (ISIN o Ticker)", "IE00B4L5Y983, SPY")

    st.sidebar.header("Impostazioni Salvataggio")
    do_save_csv = st.sidebar.checkbox("Salva/aggiorna CSV", True)

    if st.sidebar.button("Analizza ETF"):
        if not isin:
            st.warning("Inserisci un ISIN.")
        else:
            with st.spinner("Calcolo in corso…"):
                try:
                    csv_path = DEFAULT_CSV_PATH if do_save_csv else None
                    info_out, rows, frames = _cached_compute(isin, tuple(periods), csv_path, bench_override, rf_ann)
                    st.session_state.data_out = (info_out, rows, frames)
                    st.session_state.etf_params = {'compare_input': compare_input, 'periods': periods}
                except Exception as e:
                    st.error(f"Errore durante l'analisi: {e}")
                    st.session_state.data_out = None

    st.title("Analisi Singolo ETF")
    if st.session_state.data_out:
        display_single_etf_results()
    else:
        st.info("Inserisci un ISIN e premi 'Analizza ETF'.")


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _cached_compute(isin_: str, periods_: Tuple[str, ...], csv_path_: Optional[str], bench_override_: Optional[str],
                    rf_ann_: float):
    return compute_etf_over_periods(isin_, list(periods_), csv_path_, bench_override_, rf_ann_)


def display_single_etf_results():
    info_out, rows, aligned_frames = st.session_state.data_out
    params = st.session_state.etf_params
    periods = params['periods']

    # Header con informazioni estese
    st.subheader("Informazioni ETF")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ISIN", info_out.get("isin", "n.d."))
        st.metric("Ticker Yahoo", info_out.get("yahoo_symbol", "n.d."))
        st.metric("Categoria", info_out.get("category", "n.d."))

    with col2:
        st.metric("TER", info_out.get("ter_pct", "n.d."))
        st.metric("Tracking Difference", info_out.get("td_external", "n.d."))
        st.metric("Dimensione Fondo", info_out.get("fund_size", "n.d."))

    with col3:
        st.metric("Metodo di Replica", info_out.get("replication_method", "n.d."))
        st.metric("Distribuzione", info_out.get("distribution", "n.d."))
        b_sym = info_out.get("benchmark_symbol")
        b_name = info_out.get("benchmark_name")
        st.write(f"**Benchmark:** `{b_sym}` - *{b_name}*")

        # Mostra feedback se è stata aggiunta una nuova mappatura
        if info_out.get("mapping_updated"):
            st.success(f"🎯 {info_out['mapping_updated']}")
            st.info("💡 La prossima volta questo ISIN userà automaticamente il benchmark corretto!")

    # Mostra alternative con tracking difference migliore
    alternative_etfs = info_out.get("alternative_etfs", [])
    if alternative_etfs:
        st.subheader("Alternative con Tracking Difference Migliore")
        alt_df = pd.DataFrame(alternative_etfs)
        st.dataframe(alt_df.style.format({"td": "{:.2f}%"}), use_container_width=True)

    # Crea solo due tab: Grafico e Confronto
    tabs = st.tabs(["📈 Grafico", "🔍 Confronto"])

    # Tab Grafico con filtro periodo
    with tabs[0]:
        # Filtro periodo (simile a quello del confronto)
        period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1A": "1y", "3A": "3y", "5A": "5y", "Max": "max"}
        selected_period_label = st.radio(
            "Seleziona periodo:",
            list(period_map.keys()),
            index=4,  # Default su "1A"
            horizontal=True,
            key="etf_analysis_period_radio"
        )
        selected_period = period_map[selected_period_label]

        # Filtra i dati per il periodo selezionato
        df = pd.DataFrame(rows).fillna(pd.NA)
        filtered_df = df[df['period'] == selected_period] if not df.empty else pd.DataFrame()

        # Mostra metriche del periodo selezionato
        if not filtered_df.empty:
            st.subheader(f"📊 Metriche Performance - {selected_period_label}")

            row_data = filtered_df.iloc[0]

            # Metriche principali in colonne
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_ret = row_data.get('total_return_pct')
                st.metric("Rendimento Totale",
                         f"{total_ret:.2f}%" if pd.notna(total_ret) else "n.d.")

            with col2:
                cagr = row_data.get('cagr_pct')
                st.metric("CAGR",
                         f"{cagr:.2f}%" if pd.notna(cagr) else "n.d.")

            with col3:
                vol = row_data.get('vol_ann_pct')
                st.metric("Volatilità Ann.",
                         f"{vol:.2f}%" if pd.notna(vol) else "n.d.")

            with col4:
                mdd = row_data.get('mdd_pct')
                st.metric("Max Drawdown",
                         f"{mdd:.2f}%" if pd.notna(mdd) else "n.d.")

            # Seconda riga di metriche
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                sharpe = row_data.get('sharpe_ratio')
                st.metric("Sharpe Ratio",
                         f"{sharpe:.2f}" if pd.notna(sharpe) else "n.d.")

            with col6:
                calmar = row_data.get('calmar_ratio')
                st.metric("Calmar Ratio",
                         f"{calmar:.2f}" if pd.notna(calmar) else "n.d.")

            with col7:
                td = row_data.get('tracking_diff_pct')
                st.metric("Tracking Diff.",
                         f"{td:.2f}%" if pd.notna(td) else "n.d.")

            with col8:
                ter = row_data.get('ter_pct')
                if isinstance(ter, str):
                    st.metric("TER", ter)
                else:
                    st.metric("TER", f"{ter:.2f}%" if pd.notna(ter) else "n.d.")

        # Grafico del periodo selezionato
        st.subheader(f"📈 Grafico Performance - {selected_period_label}")

        frame = aligned_frames.get(selected_period)

        # Debug: mostra informazioni sui dati
        if frame is not None:
            st.write(f"🔍 Debug: Frame shape: {frame.shape}, Columns: {list(frame.columns)}")
        else:
            st.write(f"🔍 Debug: Nessun frame per periodo {selected_period}")

        if frame is not None and not frame.empty and len(frame) > 1:
            if HAS_PLOTLY:
                fig = go.Figure()

                for col in frame.columns:
                    try:
                        # Verifica che la serie abbia dati validi
                        series = frame[col].dropna()
                        if len(series) < 2:
                            st.warning(f"Serie {col} ha dati insufficienti ({len(series)} punti)")
                            continue

                        pct = to_percent_index(series)

                        if pct.empty:
                            st.warning(f"Serie percentuale vuota per {col}")
                            continue

                        # Linea principale
                        fig.add_trace(go.Scatter(
                            x=pct.index,
                            y=pct.values,
                            mode='lines',
                            name=col,
                            line=dict(width=2),
                            connectgaps=True
                        ))

                        # Media mobile 50 giorni solo per l'ETF (prima colonna)
                        if len(series) >= 50 and col == frame.columns[0]:
                            ma_50 = series.rolling(window=50).mean()
                            ma_50_pct = to_percent_index(ma_50.dropna())

                            if not ma_50_pct.empty:
                                fig.add_trace(go.Scatter(
                                    x=ma_50_pct.index,
                                    y=ma_50_pct.values,
                                    mode='lines',
                                    name=f'{col} MA50',
                                    line=dict(
                                        width=1.5,
                                        dash='dash',
                                        color='rgba(128,128,128,0.7)'
                                    ),
                                    connectgaps=True
                                ))

                    except Exception as e:
                        st.error(f"Errore nella creazione del grafico per {col}: {e}")
                        continue

                if fig.data:  # Solo se abbiamo tracce valide
                    fig.update_layout(
                        title=f"Performance vs Benchmark - {selected_period_label}",
                        yaxis_title="Performance (%)",
                        xaxis_title="Data",
                        hovermode='x unified',
                        showlegend=True,
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True, key=f"etf_chart_{selected_period}")
                else:
                    st.error("Nessuna traccia valida da visualizzare nel grafico")

            else:
                try:
                    chart_data = frame.apply(to_percent_index)
                    if not chart_data.empty:
                        st.line_chart(chart_data)
                    else:
                        st.warning("Dati del grafico vuoti dopo conversione percentuale")
                except Exception as e:
                    st.error(f"Errore nella creazione del grafico fallback: {e}")

        else:
            if frame is None:
                st.warning(f"Nessun dato di allineamento disponibile per il periodo {selected_period_label}")
            elif frame.empty:
                st.warning(f"Frame vuoto per il periodo {selected_period_label}")
            elif len(frame) <= 1:
                st.warning(f"Dati insufficienti per il grafico (solo {len(frame)} punto/i)")

        # Tabella completa (opzionale, nascosta in un expander)
        with st.expander("📋 Tabella Completa Tutti i Periodi"):
            view_cols = ["period", "total_return_pct", "cagr_pct", "vol_ann_pct", "mdd_pct", "sharpe_ratio", "calmar_ratio"]

            if all(col in df.columns for col in view_cols):
                df_view = df[view_cols].rename(columns={
                    "period": "Periodo", "total_return_pct": "Totale %", "cagr_pct": "CAGR %",
                    "vol_ann_pct": "Vol Ann %", "mdd_pct": "Max DD %", "sharpe_ratio": "Sharpe",
                    "calmar_ratio": "Calmar"
                })

                formatters = {
                    "Totale %": "{:.2f}%", "CAGR %": "{:.2f}%", "Vol Ann %": "{:.2f}%",
                    "Max DD %": "{:.2f}%", "Sharpe": "{:.2f}", "Calmar": "{:.2f}"
                }

                st.dataframe(
                    df_view.style.format(formatters, na_rep="n.d."),
                    use_container_width=True
                )


    # Tab Confronto
    with tabs[1]:
        display_comparison_tab(params['compare_input'])


def display_comparison_tab(compare_input):
    st.subheader("Confronto ETF su diversi periodi")

    # Logica per risolvere ISIN e Ticker
    inputs = [item.strip().upper() for item in re.split(r'[,\n]', compare_input) if item.strip()]
    tickers_to_compare = []
    with st.spinner("Risoluzione ISIN in corso..."):
        for item in inputs:
            if re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', item):  # È un ISIN
                ticker = resolve_isin_one(item)
                if ticker:
                    tickers_to_compare.append(ticker)
                else:
                    st.warning(f"ISIN {item} non trovato, verrà saltato.")
            else:  # È un Ticker
                tickers_to_compare.append(item)

    tickers = list(set(tickers_to_compare))  # Rimuovi duplicati

    if not tickers:
        st.info("Aggiungi ISIN o Ticker validi nella sidebar per confrontarli.")
        return

    period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1A": "1y", "3A": "3y", "5A": "5y", "Max": "max"}
    selected_period_label = st.radio("Seleziona periodo", list(period_map.keys()), index=4, horizontal=True,
                                     key="comparison_period_radio")
    period_yf = period_map[selected_period_label]

    data = {ticker: get_series(ticker, period_yf) for ticker in tickers}

    if HAS_PLOTLY:
        fig = go.Figure()
        for ticker, series in data.items():
            if series is not None and not series.empty:
                pct = to_percent_index(series)
                fig.add_trace(go.Scatter(x=pct.index, y=pct.values, mode='lines', name=ticker))

        fig.update_layout(title=f"Andamento ETF ({selected_period_label})", yaxis_title="% dal primo punto")
        st.plotly_chart(fig, use_container_width=True, key="comparison_chart")
    else:
        # Fallback usando st.line_chart
        chart_data = pd.DataFrame({ticker: to_percent_index(series)
                                  for ticker, series in data.items()
                                  if series is not None and not series.empty})
        if not chart_data.empty:
            st.line_chart(chart_data)

# --- MODALITÀ: RICERCA ETF ---
def render_etf_search_ui():
    """UI Streamlit per ricerca universale ETF UCITS (default periodo 1Y)."""
    st.title("🔎 ETF Search Engine (Universale • UCITS)")
    st.sidebar.header("⚙️ Parametri di ricerca")

    period = st.sidebar.selectbox(
        "Periodo storico",
        options=["1m", "3m", "6m", "1y", "3y", "5y", "10y", "ytd", "max"],
        index=3  # default -> "1y"
    )

    risk_free_pct = st.sidebar.number_input(
        "Tasso risk-free annuo (%)",
        value=3.95,         # ✅ 4.0 = 4% annuo
        min_value=-5.0,
        max_value=20.0,
        step=0.10,
        format="%.2f"
    )

    quotes_per_query = st.sidebar.slider(
        "Risultati per query seed",
        min_value=20, max_value=200, value=100, step=10
    )

    limit_universe = st.sidebar.slider(
        "Limite universo (discovery)",
        min_value=100, max_value=2000, value=600, step=50
    )

    max_results = st.sidebar.slider(
        "ETF da valutare (post discovery)",
        min_value=10, max_value=500, value=150, step=10
    )

    sort_by = st.sidebar.selectbox(
        "Ordina per",
        options=["sharpe ratio", "rendimento cagr", "max drawdown", "ticker"],
        index=0
    )

    ascending = st.sidebar.checkbox("Ordine crescente", value=False)

    if st.sidebar.button("Esegui ricerca universale"):
        st.subheader("Risultati ricerca ETF (UCITS)")
        with st.spinner("Scopro l'universo ETF UCITS e calcolo le metriche..."):
            df = search_etfs_universal(
                period=period,
                risk_free_rate_pct=risk_free_pct,   # ✅ in %
                quotes_per_query=quotes_per_query,
                limit_universe=limit_universe,
                max_results=max_results,
                sort_by=sort_by,
                ascending=ascending,
            )

        if df.empty:
            st.warning("Nessun risultato utile o dati insufficienti per calcolare le metriche.")
        else:
            st.dataframe(format_results_for_display(df), use_container_width=True)


# --- MODALITÀ: VERIFICA MOMENTUM ---
def render_momentum_ui():
    st.sidebar.header("Impostazioni Momentum")
    momentum_isins = st.sidebar.text_area("ISIN da confrontare (uno per riga)",
                                          "IE00BMC38736\nIE000YYE6WK5\nIE00BK5BQT80\nIE00BDDRF700\nIE00BL25JP72\nIE000M7V94E1\nIE00BYPLS672\nIE0007Y8Y157\nIE00BYZK4552\nIE00BJGWQN72\nIE00BYPLS672\nIE00BDVPNG13\nIE00BM67HT60\nLU2023678282\nIE00BYWQWR46\nIE00BYZK4669\nIE00BYZK4776\nIE00BQ70R696\nIE00BYZK4883\nIE000U58J0M1\nIE00BF0M6N54\nIE00BFYN8Y92\nIE00BKTLJC87\nIE00B1FZS467\nIE00BGL86Z12\nIE00BGBN6P67\nIE00BLRPQH31\nIE00BF0M2Z96\nIE00B1FZSF77\nIE00B579F325\nDE000A1EK0G3\nCH0454664001\nGB00BLD4ZL17", height=120)
    lookback = st.sidebar.slider("Periodo di lookback (mesi)", 1, 24, 6, 1)
    rf_ann_momentum = st.sidebar.number_input("Risk-free annuo (%) per Sharpe", -5.0, 10.0, 3.9, 0.25,
                                              key="rf_momentum")

    if st.sidebar.button("Verifica Momentum"):
        isins_list = [isin for isin in momentum_isins.split('\n') if isin.strip()]
        if not isins_list:
            st.warning("Inserisci almeno un ISIN.")
            st.session_state.momentum_results = None
        else:
            st.session_state.momentum_results = calculate_momentum_rankings(isins_list, lookback, rf_ann_momentum)

    st.title("Verifica Momentum Corretto per il Rischio")
    st.caption(f"Classifica per Sharpe Ratio negli ultimi {lookback} mesi.")

    if st.session_state.momentum_results:
        results = st.session_state.momentum_results
        if results:
            best_performer = results[0]
            st.success(
                f"🥇 **Asset con miglior Sharpe Ratio:** **{best_performer['ticker']}** ({best_performer['isin']}) con Sharpe di **{best_performer['sharpe_ratio']:.2f}**.")

            st.subheader("Classifica Completa")
            df = pd.DataFrame(results).rename(columns={
                "isin": "ISIN", "ticker": "Ticker", "cagr_pct": "CAGR %",
                "mdd_pct": "Max DD %", "sharpe_ratio": "Sharpe Ratio"
            }).drop(columns=["start_date", "end_date"])

            formatters = {"CAGR %": "{:.2f}%", "Max DD %": "{:.2f}%", "Sharpe Ratio": "{:.2f}"}
            st.dataframe(df.style.format(formatters, na_rep="n.d."), use_container_width=True)
        else:
            st.error("Impossibile calcolare il momentum. Controlla gli ISIN inseriti.")
    else:
        st.info("Inserisci gli ISIN e premi 'Verifica Momentum'.")

# --- MODALITÀ: PORTFOLIO TRACKER ---

def _seed_sidebar_from_saved_portfolio(portfolio_data, prefix="sb_"):
    """
    Prefilla i campi della sidebar dai dati di un portafoglio salvato.
    Usa session_state per impostare i valori dei widget prima che vengano creati.
    """
    if not portfolio_data or not portfolio_data.get("holdings"):
        return
    st.session_state.setdefault(f"{prefix}name", portfolio_data.get("name", "Il Mio Portafoglio"))
    st.session_state.setdefault(f"{prefix}npos", len(portfolio_data["holdings"]))
    npos = len(portfolio_data["holdings"])
    for i, h in enumerate(portfolio_data["holdings"][:npos]):
        st.session_state[f"{prefix}isin_{i}"] = (h.get("isin") or "").strip().upper()
        st.session_state[f"{prefix}qty_{i}"] = float(h.get("quantity", 0.0))
        st.session_state[f"{prefix}price_{i}"] = float(h.get("purchase_price", 0.0))
        # Date: Streamlit accetta date; se è stringa, prova a convertirla
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
    con pesi (%) che sommano a ~100, calcolati su:
      - base="invested"  -> invested_amount
      - base="current"   -> current_amount
      - base="quantity"  -> quantity * purchase_price
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
        else:  # default "invested"
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


def _display_backtest_results(all_series, rf_ann, key_prefix="backtest"):
    """
    Mostra grafico normalizzato base 100 e tabella metriche (CAGR, Vol, MDD, Sharpe)
    a partire da un dizionario {nome_portafoglio: pandas.Series con index datetime}.
    """
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

    filtered_series_dict = {}
    metrics_list = []

    end_date = pd.to_datetime('today').normalize()
    if lookback_months == "ytd":
        start_date = pd.to_datetime(f"{end_date.year}-01-01")
    elif lookback_months is not None:
        start_date = end_date - pd.DateOffset(months=lookback_months)
    else:
        start_date = min(s.index.min() for s in all_series.values())

    for name, series in all_series.items():
        # Serie attesa: pandas.Series con index datetime e valori cumulativi (NAV/equity line)
        filtered_series = series[series.index >= start_date]
        if filtered_series.shape[0] < 2:
            continue
        filtered_series_dict[name] = filtered_series

        metrics = compute_metrics_from_series(filtered_series)
        metrics["sharpe"] = compute_sharpe_ratio(filtered_series, rf_ann)
        metrics["name"] = name
        metrics_list.append(metrics)

    st.subheader(f"Andamento Portafogli ({selected_period_label})")
    fig = go.Figure()
    for name, series in filtered_series_dict.items():
        norm_series = to_percent_index(series) + 100  # normalizza con base 100
        fig.add_trace(go.Scatter(x=norm_series.index, y=norm_series.values, mode='lines', name=name))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

    st.subheader(f"Metriche di Performance ({selected_period_label})")
    if metrics_list:
        df = pd.DataFrame(metrics_list).set_index("name")
        df_view = df[["cagr", "vol_ann", "mdd", "sharpe"]].rename(columns={
            "cagr": "CAGR %", "vol_ann": "Volatilità Ann. %", "mdd": "Max Drawdown %", "sharpe": "Sharpe Ratio"
        })
        st.dataframe(df_view.style.format("{:.2f}", na_rep="n.d."), use_container_width=True)
    else:
        st.info("Non ci sono abbastanza dati nel periodo selezionato per calcolare le metriche.")


def _render_allocation_pie(labels, values, title, key="alloc_pie"):
    """
    Pie chart delle allocazioni. labels: ISIN, values: importi.
    """
    if not labels or not values or sum(values) <= 0:
        st.info("Aggiungi almeno una posizione per vedere l'allocazione.")
        return
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


# ---------------------------------------------------
# TRACKER: UI con input in sidebar e riepilogo in pagina
# ---------------------------------------------------

def render_portfolio_tracker_ui():
    """
    Barra di sinistra: caricamento portafoglio, input posizioni, azioni Salva/Calcola.
    Pagina principale: riepilogo (grafico a torta), KPI, tabella, benchmark, backtest integrato.
    """
    from .portfolio_tracker import PortfolioTracker  # come nel tuo progetto

    # ---------------- Sidebar ----------------
    st.sidebar.header("📂 Portafoglio")

    # Carica portafogli salvati (in sidebar)
    saved_portfolios = PortfolioTracker.get_saved_portfolio_names()
    selected_portfolio = None
    if saved_portfolios:
        selected_portfolio = st.sidebar.selectbox(
            "Carica portafoglio salvato:",
            ["Nuovo..."] + saved_portfolios,
            key="sb_saved_select"
        )
        if selected_portfolio != "Nuovo...":
            portfolio_data = PortfolioTracker.load_portfolio_from_session(selected_portfolio)
            if portfolio_data:
                st.session_state.current_portfolio = portfolio_data
                # Prefill dei campi sidebar
                _seed_sidebar_from_saved_portfolio(portfolio_data, prefix="sb_")

    st.sidebar.subheader("📝 Il Tuo Portafoglio")

    portfolio_name = st.sidebar.text_input(
        "Nome Portafoglio:", value=st.session_state.get("sb_name", "Il Mio Portafoglio"), key="sb_name"
    )
    num_positions = st.sidebar.number_input(
        "Numero di posizioni:", min_value=1, max_value=20,
        value=st.session_state.get("sb_npos", 1), key="sb_npos"
    )

    holdings = []
    for i in range(int(num_positions)):
        st.sidebar.markdown(f"**Posizione {i+1}:**")
        isin = st.sidebar.text_input(f"ISIN {i+1}:", key=f"sb_isin_{i}",
                                     value=st.session_state.get(f"sb_isin_{i}", ""))
        quantity = st.sidebar.number_input(f"Quantità {i+1}:", min_value=0.0, step=0.1, key=f"sb_qty_{i}",
                                           value=st.session_state.get(f"sb_qty_{i}", 0.0))
        purchase_price = st.sidebar.number_input(f"Prezzo Acquisto € {i+1}:", min_value=0.0, step=0.01,
                                                 key=f"sb_price_{i}",
                                                 value=st.session_state.get(f"sb_price_{i}", 0.0))
        purchase_date = st.sidebar.date_input(f"Data Acquisto {i+1}:",
                                              key=f"sb_date_{i}",
                                              value=st.session_state.get(f"sb_date_{i}", datetime.now().date()))
        if isin and quantity > 0 and purchase_price > 0:
            holdings.append({
                'isin': isin.strip().upper(),
                'quantity': quantity,
                'purchase_price': purchase_price,
                'purchase_date': purchase_date
            })

    # Azioni (in sidebar)
    if st.sidebar.button("💾 Salva Portafoglio", disabled=not holdings, key="sb_save_btn"):
        portfolio_data = {
            'name': portfolio_name,
            'holdings': holdings,
            'created_date': datetime.now().date()
        }
        PortfolioTracker.save_portfolio_to_session(portfolio_data)
        st.sidebar.success(f"✅ Portafoglio '{portfolio_name}' salvato!")

    if st.sidebar.button("📊 Calcola P&L", disabled=not holdings, key="sb_calc_btn"):
        with st.spinner("Calcolo P&L in corso..."):
            pnl_results = PortfolioTracker.calculate_portfolio_pnl(holdings)
            st.session_state.pnl_results = pnl_results
            st.success("Calcolo P&L completato!")

    # ---------------- Pagina principale ----------------
    st.title("💼 Portfolio Tracker")
    st.caption("Riepilogo e analisi del tuo portafoglio")

    # RIEPILOGO (grafico a torta)
    st.subheader("🧭 Riepilogo Portafoglio")
    if hasattr(st.session_state, 'pnl_results') and st.session_state.pnl_results and st.session_state.pnl_results.get('holdings_detail'):
        # Se P&L calcolato, usa il valore attuale per l'allocazione
        hd = st.session_state.pnl_results['holdings_detail']
        labels = [h['isin'] for h in hd]
        values = [float(h.get('current_amount', 0.0)) for h in hd]
        _render_allocation_pie(labels, values, "Allocazione attuale (%)", key="alloc_current")
    else:
        # Altrimenti mostra allocazione per Investito sulla base degli input correnti
        if holdings:
            labels = [h['isin'] for h in holdings]
            values = [float(h['quantity']) * float(h['purchase_price']) for h in holdings]
            _render_allocation_pie(labels, values, "Allocazione per Investito (%)", key="alloc_invested")
        else:
            st.info("Definisci il tuo portafoglio nella barra di sinistra per vedere il riepilogo.")

    # Se ci sono risultati P&L, mostra le sezioni di dettaglio e backtest
    if hasattr(st.session_state, 'pnl_results'):
        display_pnl_results()


def display_pnl_results():
    """
    Mostra:
    - KPI principali (Investito, Valore Attuale, P&L, P&L%)
    - Tabella dettagliata (formattata)
    - Performance vs Benchmark
    - Expander: Backtest su queste posizioni (integra il backtest in fondo alla pagina)
    - Output Backtest a fondo pagina (se eseguito)
    """
    from .portfolio_tracker import PortfolioTracker  # se serve

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

    # ------------------------------
    # TABELLA DETTAGLIATA (con le tue formattazioni)
    # ------------------------------
    if results.get('holdings_detail'):
        st.subheader("📋 Dettaglio Posizioni")

        df_holdings = pd.DataFrame(results['holdings_detail'])
        df_display = df_holdings[['isin', 'quantity', 'purchase_price', 'current_price',
                                  'invested_amount', 'current_amount', 'pnl', 'pnl_pct', 'days_held']].copy()

        df_display.columns = [
            'ISIN', 'Quantità', 'Prezzo Acquisto €', 'Prezzo Attuale €',
            'Investito €', 'Valore Attuale €', 'P&L €', 'P&L %', 'Giorni'
        ]

        # Formattazione e colorazione P&L
        st.dataframe(
            df_display.style.format({
                'Prezzo Acquisto €': '€{:.2f}',
                'Prezzo Attuale €': '€{:.2f}',
                'Investito €': '€{:,.2f}',
                'Valore Attuale €': '€{:,.2f}',
                'P&L €': '€{:+,.2f}',
                'P&L %': '{:+.2f}%'
            }).applymap(
                lambda x: 'color: green' if isinstance(x, str) and '+' in x
                else 'color: red' if isinstance(x, str) and (x.startswith('€-') or x.startswith('-'))
                else '',
                subset=['P&L €', 'P&L %']
            ),
            use_container_width=True
        )

    # ------------------------------
    # PERFORMANCE VS BENCHMARK
    # ------------------------------
    if results.get('holdings_detail'):
        st.subheader("🏆 Performance vs Benchmark")
        benchmark_results = PortfolioTracker.get_performance_vs_benchmark(
            [{'isin': h['isin'], 'quantity': h['quantity'],
              'purchase_price': h['purchase_price'], 'purchase_date': h['purchase_date']}
             for h in results['holdings_detail']]
        )

        if 'error' not in benchmark_results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Performance",
                          f"{benchmark_results.get('portfolio_performance', 0):.2f}%")
            with col2:
                st.metric("S&P 500 Performance",
                          f"{benchmark_results.get('benchmark_performance', 0):.2f}%")
            with col3:
                outperf = benchmark_results.get('outperformance', 0)
                st.metric("Outperformance", f"{outperf:+.2f}%",
                          delta_color="normal" if outperf >= 0 else "inverse")
        else:
            st.warning(benchmark_results.get('error', 'Impossibile calcolare il benchmark.'))

    # ------------------------------
    # BACKTEST SU QUESTE POSIZIONI (EXPANDER)
    # ------------------------------
    if results.get('holdings_detail'):
        with st.expander("🔙 Backtest su queste posizioni", expanded=False):
            # Base pesi
            base_label = st.radio(
                "Base per i pesi",
                ["Investito", "Valore attuale", "Quantità"],
                index=0,  # imposta a 1 se vuoi default "Valore attuale"
                horizontal=True,
                key="tracker_bt_base"
            )
            base_map = {"Investito": "invested", "Valore attuale": "current", "Quantità": "quantity"}
            base_key = base_map[base_label]

            # Strategia
            strategy = st.radio("Strategia", ["Lump Sum", "PAC"], index=0, horizontal=True, key="tracker_bt_strategy")
            monthly_investment = 0
            if strategy == "PAC":
                monthly_investment = st.number_input(
                    "Investimento mensile (€)", min_value=100, max_value=10000, value=1000, step=100,
                    key="tracker_bt_pac_amount"
                )

            # Confronto & Risk-free
            try:
                famous_all = list(FAMOUS_PORTFOLIOS.keys())  # deve essere disponibile in UI.py
            except Exception:
                famous_all = []

            famous_selection = st.multiselect(
                "Confronta con portafogli modello",
                famous_all,
                default=(["Classic 60/40"] if "Classic 60/40" in famous_all else []),
                key="tracker_bt_famous_selection"
            )
            rf_ann_backtest = st.number_input(
                "Risk-free annuo (%) per Sharpe", min_value=-5.0, max_value=10.0, value=1.0, step=0.25,
                key="tracker_bt_rf_ann"
            )

            # Esegui backtest
            if st.button("▶️ Esegui Backtest su questo portafoglio", key="tracker_bt_run_btn"):
                portfolio_text = _build_weights_text_from_holdings_detail(results['holdings_detail'], base=base_key)
                if not portfolio_text:
                    st.error("Impossibile costruire i pesi dalle posizioni correnti.")
                else:
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
                            st.session_state.tracker_backtest_results = {
                                'series': all_series,
                                'rf_ann': rf_ann_backtest
                            }
                            st.success("Backtest completato!")
                    else:
                        st.error("Input generato non valido. Controlla le posizioni.")

    # ------------------------------
    # OUTPUT BACKTEST (fondo pagina)
    # ------------------------------
    tracker_bt = st.session_state.get('tracker_backtest_results')
    if tracker_bt and tracker_bt.get('series'):
        st.markdown("---")
        st.subheader("📊 Backtest (portafoglio corrente)")
        _display_backtest_results(tracker_bt['series'], tracker_bt['rf_ann'], key_prefix="tracker_bt")

# --- ENTRY POINT ---
if __name__ == '__main__':
    run_app()
