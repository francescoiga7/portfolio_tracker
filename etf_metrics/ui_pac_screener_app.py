# -*- coding: utf-8 -*-
import streamlit as st
from datetime import date
from .pac_screener import fetch_screener_data, process_screener_rankings, get_market_regime
import pandas as pd


def render_pac_screener_ui():
    """Renderizza la UI per il Screener Tattico con logica di caching avanzata."""
    st.title("🎯 Screener ETF Tattico")
    st.caption("Scopri ETP con potenziale di breakout quotati sulle principali borse europee.")

    # Inizializza il log di debug nella sessione se non esiste
    if 'debug_log_loading' not in st.session_state:
        st.session_state.debug_log_loading = []
    if 'debug_log_processing' not in st.session_state:
        st.session_state.debug_log_processing = []

    with st.expander("📖 Leggi la Metodologia"):
        st.markdown("""
           Questo screener è ottimizzato per la parte **satellite** di un portafoglio e utilizza un approccio multi-fattore per identificare ETP con alto potenziale a breve-medio termine.

           **1. Contesto di Mercato (Filtro VIX):** Lo screener opera solo in regimi di mercato favorevoli al rischio (VIX < 20).
           **2. Filtro di Liquidità:** Vengono considerati solo ETP con un volume medio giornaliero scambiato superiore alla soglia impostata.
           **3. Filtro di Momentum Assoluto:** Vengono considerati solo ETP con performance a 6 e 12 mesi positiva.
           **4. Punteggio Composito:** Basato su Momentum, Qualità del Trend (Calmar Ratio), Compressione di Volatilità e Bassa Volatilità.

           *Questo screener non costituisce una raccomandazione di investimento.*
           """)

    st.sidebar.header("⚙️ Parametri Screener")

    # --- Sezione 1: Caricamento Dati ---
    with st.sidebar.expander("1. Carica Universo Dati", expanded=True):

        source_mode = st.radio("Modalità di Ricerca", ["Scoperta Automatica Universo", "Inserisci ISIN Specifici"])

        specific_isins_input = ""
        if source_mode == "Inserisci ISIN Specifici":
            specific_isins_input = st.text_area("Elenco ISIN (uno per riga)", height=150)

        if st.button("Carica/Aggiorna Dati"):
            st.session_state.pac_screener_raw_data = None
            st.session_state.debug_log_loading = []
            st.session_state.debug_log_processing = []

            specific_isins = None
            if source_mode == "Inserisci ISIN Specifici" and specific_isins_input.strip():
                specific_isins = [isin.strip().upper() for isin in specific_isins_input.split('\n') if isin.strip()]

            log_list_for_fetching = []
            st.session_state.pac_screener_raw_data = fetch_screener_data(
                log_area=log_list_for_fetching,
                specific_isins=specific_isins
            )
            st.session_state.debug_log_loading.extend(log_list_for_fetching)

    # --- Sezione 2: Analisi Interattiva ---
    if 'pac_screener_raw_data' in st.session_state and st.session_state.pac_screener_raw_data:
        st.sidebar.header("2. Filtri e Time Travel")

        min_avg_value = st.sidebar.number_input(
            "Volume minimo scambiato (€)", 0, 1000000, 100000, 50000,
            help="Volume medio giornaliero minimo (in €)."
        )

        enable_time_travel = st.sidebar.checkbox("Abilita Time Travel")
        as_of_date = st.sidebar.date_input(
            "Posizionati alla data del:",
            date.today(),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
            disabled=not enable_time_travel
        )
        analysis_date = as_of_date if enable_time_travel else date.today()

        market_info = get_market_regime(as_of_date=analysis_date)
        vix = market_info.get("vix")
        regime = market_info.get("regime")

        if regime != "Favorevole al Rischio":
            vix_str = f"{vix:.2f}" if vix is not None else "N/D"
            st.sidebar.error(f"Regime di Mercato al {analysis_date.strftime('%d/%m/%Y')}: {regime} (VIX: {vix_str})")
            st.sidebar.warning(
                "In questa data il mercato era in alta volatilità. Lo screener è disattivato per prudenza.")
            st.warning(
                f"Alla data del {analysis_date.strftime('%d/%m/%Y')} il mercato era in modalità Avversa al Rischio. Lo screener non produce risultati per prudenza.")
            return

        st.sidebar.success(f"Regime di Mercato al {analysis_date.strftime('%d/%m/%Y')}: Favorevole (VIX: {vix:.2f})")

        if enable_time_travel:
            st.info(
                f"Modalità Time Travel: L'analisi riflette i dati disponibili al **{analysis_date.strftime('%d/%m/%Y')}**.")

        log_list_for_processing = []
        results = process_screener_rankings(
            st.session_state.pac_screener_raw_data,
            min_avg_value,
            analysis_date,
            log_list_for_processing
        )
        st.session_state.debug_log_processing = log_list_for_processing

        if not results.empty:
            top_etf = results.iloc[0]
            st.success(
                f"🏆 Candidato Migliore al {analysis_date.strftime('%d/%m/%Y')}: **{top_etf['name']} ({top_etf['ticker']})**")
            st.subheader("Classifica Completa")
            df_display = results.rename(columns={
                "ticker": "Ticker", "isin": "ISIN", "name": "Nome", "final_score": "Punteggio Finale",
                "quality_score": "Score Qualità", "momentum_score": "Score Momentum",
                "breakout_score": "Score Breakout", "low_vol_score": "Score Low Volatility",
                "calmar_ratio": "Calmar Ratio (12M)", "roc_6m": "Rendimento 6M",
                "volatility_6m": "Volatilità 6M", "avg_value_eur": "Valore Scambiato (€)"
            })
            st.dataframe(
                df_display.style.format({
                    "Punteggio Finale": "{:.2f}", "Score Qualità": "{:.2f}", "Score Momentum": "{:.2f}",
                    "Score Breakout": "{:.2f}", "Score Low Volatility": "{:.2f}",
                    "Calmar Ratio (12M)": "{:.2f}", "Rendimento 6M": "{:+.2%}",
                    "Volatilità 6M": "{:.2%}", "Valore Scambiato (€)": "€{:,.0f}"
                }).background_gradient(cmap='Greens', subset=['Punteggio Finale', 'Score Qualità', 'Score Momentum'])
                .background_gradient(cmap='Reds_r', subset=['Score Breakout', 'Score Low Volatility']),
               width="stretch"
            )
        else:
            st.warning(
                "Nessun ETP ha soddisfatto i criteri alla data selezionata. "
                "Prova ad allargare i filtri (es. abbassare il volume minimo)."
            )

        with st.expander("🔍 Dettagli del Processo di Screening (Debug Log)"):
            st.markdown("##### Log del Caricamento Dati")
            st.markdown("\n".join(st.session_state.debug_log_loading))

            st.markdown("##### Log del Calcolo Metriche e Filtri")
            st.markdown("\n".join(st.session_state.debug_log_processing))

            log_df_data = [log for log in st.session_state.debug_log_processing if isinstance(log, list)]
            if log_df_data:
                log_df = pd.DataFrame(log_df_data, columns=['Ticker', 'Esito'])
                st.dataframe(log_df,width="stretch")

    else:
        st.info("Imposta la modalità e premi 'Carica/Aggiorna Dati Universo ETF' per iniziare.")