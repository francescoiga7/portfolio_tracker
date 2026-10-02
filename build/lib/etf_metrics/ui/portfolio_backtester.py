# -*- coding: utf-8 -*-
import streamlit as st

from etf_metrics.core.portfolio_backtester import (
    parse_portfolio_input,
    get_all_portfolios_for_backtest,
    compute_weights_from_transactions,
)
from etf_metrics.shared.config import FAMOUS_PORTFOLIOS
from etf_metrics.ui.components import display_backtest_results, get_portfolio_store


def get_weights_from_portfolio(portfolio_name: str) -> str:
    """Carica un portafoglio e calcola i pesi percentuali dalle posizioni aperte."""
    portfolio_data = get_portfolio_store().get(portfolio_name)
    transactions = portfolio_data.get("transactions") if portfolio_data else None
    if not transactions:
        return ""
    return compute_weights_from_transactions(transactions)


def render_portfolio_backtester_ui():
    st.title("⏳ Portfolio Backtester")
    st.caption("Simula e confronta le performance di portafogli personalizzati nel tempo.")

    with st.expander("📖 Metodologia e Funzionamento"):
        st.markdown("""
        Questa sezione ti permette di simulare e confrontare le performance di portafogli personalizzati nel tempo. Puoi:
        - **Scegliere una strategia di investimento**: "Lump Sum (PIC)" o "PAC".
        - **Definire il tuo portafoglio**: manualmente o caricandolo da un portafoglio esistente.
        - **Impostare la frequenza di ribilanciamento**: "Mai" o "Annuale".
        - **Confrontare il tuo portafoglio** con alcuni portafogli modello.
        - **Eseguire il backtest** e visualizzare i risultati.
        """)

    st.sidebar.header("⚙️ Impostazioni")

    strategy = st.sidebar.selectbox("Strategia di Investimento", ["Lump Sum (PIC)", "PAC"])

    initial_investment = 0
    monthly_investment = 0

    if strategy == "Lump Sum (PIC)":
        initial_investment = st.sidebar.number_input("Investimento Iniziale (€)", min_value=100, max_value=1000000,
                                                     value=10000, step=100)
    elif strategy == "PAC":
        monthly_investment = st.sidebar.number_input("Investimento Mensile (€)", min_value=50, max_value=10000,
                                                     value=500, step=50)

    st.sidebar.subheader("Definizione Portafoglio")
    portfolio_source = st.sidebar.radio("Scegli come definire il portafoglio", ["Manuale", "Carica da Portafoglio Esistente"])

    portfolio_text_content = ""
    is_disabled = False

    if portfolio_source == "Carica da Portafoglio Esistente":
        saved_portfolios = get_portfolio_store().names()
        if not saved_portfolios:
            st.sidebar.warning("Nessun portafoglio salvato. Creane uno nel Portfolio Tracker.")
        else:
            selected_portfolio = st.sidebar.selectbox("Seleziona un Portafoglio", saved_portfolios)
            if selected_portfolio:
                portfolio_text_content = get_weights_from_portfolio(selected_portfolio)
                is_disabled = True

    portfolio_text = st.sidebar.text_area(
        "Asset e Pesi (ISIN: Peso%)",
        value=portfolio_text_content if portfolio_text_content else "IE00BK5BQT80: 80\nIE00BDBRDM35: 20",
        height=150,
        help="Inserisci un ISIN o Ticker per riga, seguito da ':' e dal peso percentuale (es. 'VWCE.MI: 80').",
        disabled=is_disabled,
    )

    rebalancing = st.sidebar.selectbox("Frequenza di Ribilanciamento", ["Mai", "Annuale"])

    try:
        famous_all = list(FAMOUS_PORTFOLIOS.keys())
    except Exception:
        famous_all = []

    famous_selection = st.sidebar.multiselect(
        "Confronta con portafogli modello",
        famous_all,
        default=(["Classic 60/40"] if "Classic 60/40" in famous_all else [])
    )

    rf_ann_backtest = st.sidebar.number_input("Risk-free annuo (%)", min_value=-5.0, max_value=10.0,
                                              value=3.95, step=0.25)

    if st.sidebar.button("▶️ Esegui Backtest"):
        user_portfolio_def = parse_portfolio_input(portfolio_text)
        if not user_portfolio_def:
            st.error("La definizione del portafoglio non è valida. Controlla il formato.")
        else:
            with st.spinner("Esecuzione backtest in corso..."):
                all_series = get_all_portfolios_for_backtest(
                    user_portfolio_def,
                    famous_selection,
                    FAMOUS_PORTFOLIOS if famous_all else {},
                    initial_investment=initial_investment,
                    strategy=strategy.lower().replace(" ", "_"),
                    monthly_investment=monthly_investment,
                    rebalancing=rebalancing.lower()
                )
                st.session_state.backtest_results = {'series': all_series, 'rf_ann': rf_ann_backtest}
            st.success("Backtest completato!")

    if st.session_state.get('backtest_results') and st.session_state.backtest_results.get('series'):
        st.subheader("📊 Risultati del Backtest")
        display_backtest_results(
            st.session_state.backtest_results['series'],
            st.session_state.backtest_results['rf_ann'],
            key_prefix="main_bt"
        )
    else:
        st.info("👈 Definisci un portafoglio e i parametri nella sidebar, poi premi 'Esegui Backtest'.")
