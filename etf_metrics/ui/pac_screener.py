# -*- coding: utf-8 -*-
import streamlit as st
from datetime import date
# Importa le nuove funzioni
from etf_metrics.core.pac_screener import fetch_screener_data, calculate_all_metrics, filter_and_rank_metrics, \
    get_market_regime
import pandas as pd

# PESI "BREAKOUT FOCUSED"
ALGORITHM_WEIGHTS = {
    "Momentum Score": 0.40,
    "Trend Quality Score (Calmar)": 0.05,
    "Breakout Score (Volatilità Compressa)": 0.50,
    "Low Volatility Score": 0.05
}


def render_pac_screener_ui():
    """Renderizza la UI per il Screener Tattico con logica di caching avanzata."""
    st.title("🎯 Screener ETF")
    st.caption("Scopri ETP con potenziale di breakout quotati sulle principali borse europee.")

    if 'debug_log_loading' not in st.session_state:
        st.session_state.debug_log_loading = []
    if 'debug_log_processing' not in st.session_state:
        st.session_state.debug_log_processing = []

    with st.expander("📖 Metodologia e Analisi Quantitativa (Approccio 'Breakout Focused')"):
        # (Markdown invariato)
        st.markdown(f"""
              Questo screener è progettato per la parte **satellite (tattica)** di un portafoglio. Ora è sintonizzato per una strategia **"Breakout Focused"**, dando priorità agli asset che mostrano una forte compressione di volatilità pronta per un movimento esplosivo.

              ---

              ### 1. Filtri Preliminari (I "Guardrail")

              Prima di qualsiasi ranking, applichiamo filtri di sicurezza per evitare trappole:
              * **Contesto di Mercato (VIX + Trend SP500):** Si opera solo in contesti "Risk-On" (VIX basso, SP500 > MA200).
              * **Filtro di Trend (Singolo ETF):** Si considerano solo ETF sopra la loro MA200.
              * **Filtro di Continuazione Trend:** Si considerano solo ETF vicini ai loro massimi a 52 settimane.
              * **Filtro di Liquidità:** Si escludono asset illiquidi.

              ### 2. Analisi Quantitative e Fattori di Ranking

              L'algoritmo ora dà priorità massima al potenziale di breakout.

              | Fattore di Punteggio | Metrica Quantitativa | Importanza nell'Algoritmo |
              | :--- | :--- | :--- |
              | **Breakout/Compressione** | Volatilità di Breve Termine vs. Lungo Termine | **(Peso Massimo)** - Il driver primario. Cerchiamo la "molla" più carica. |
              | **Momentum Assoluto** | Performance a 6 e 12 Mesi (ROC - Rate of Change) | **(Peso Alto)** - Il motore che conferma la direzione del breakout. |
              | **Momentum Assoluto** | Performance a 6 e 12 Mesi (ROC - Rate of Change) | **(Peso Alto)** - Il motore che conferma la direzione del breakout. |
              | **Qualità del Trend** | Calmar Ratio (12 Mesi) | **(Peso Minimo)** - Usato solo come "tie-breaker" (spareggio). |
              | **Bassa Volatilità Relativa** | Volatilità a 6 Mesi | **(Peso Minimo)** - Rilevanza minima. |

              ---

              ### 3. Punteggio Composito e Pesi

              Il **Punteggio Finale** è una media pesata dei singoli punteggi. 

              * **Formula:** $Punteggio \: Finale = \sum (Score_Fattore \times Peso_Fattore)$

              I pesi attuali assegnati sono:
              * **Breakout Score:** {ALGORITHM_WEIGHTS['Breakout Score (Volatilità Compressa)'] * 100:.0f}% (Priorità Massima)
              * **Momentum Score:** {ALGORITHM_WEIGHTS['Momentum Score'] * 100:.0f}% (Priorità Alta)
              * **Trend Quality Score (Calmar):** {ALGORITHM_WEIGHTS['Trend Quality Score (Calmar)'] * 100:.0f}% (Peso Minimo)
              * **Low Volatility Score:** {ALGORITHM_WEIGHTS['Low Volatility Score'] * 100:.0f}% (Peso Minimo)
              """)

    st.sidebar.header("⚙️ Impostazioni")

    with st.sidebar.expander("1. Carica Universo Dati", expanded=True):
        source_mode = st.radio("Modalità di Ricerca", ["Scoperta Automatica Universo", "Inserisci ISIN Specifici"])

        specific_isins_input = ""
        if source_mode == "Inserisci ISIN Specifici":
            specific_isins_input = st.text_area("Elenco ISIN (uno per riga)", height=150)

        if st.button("Carica/Aggiorna Dati"):
            # Pulisce *tutte* le cache correlate quando ricarichiamo i dati
            st.cache_data.clear()
            st.session_state.pac_screener_raw_data = None
            st.session_state.debug_log_loading = []
            st.session_state.debug_log_processing = []

            specific_isins = None
            if source_mode == "Inserisci ISIN Specifici" and specific_isins_input.strip():
                specific_isins = [isin.strip().upper() for isin in specific_isins_input.split('\n') if isin.strip()]

            log_list_for_fetching = []
            # Fase 1: Fetching
            st.session_state.pac_screener_raw_data = fetch_screener_data(
                log_area=log_list_for_fetching,
                specific_isins=specific_isins
            )
            st.session_state.debug_log_loading.extend(log_list_for_fetching)
            # Forza un rerun per passare alla fase 2
            st.rerun()

    if 'pac_screener_raw_data' in st.session_state and st.session_state.pac_screener_raw_data:
        st.sidebar.header("2. Filtri e Time Travel")

        min_avg_value = st.sidebar.number_input(
            "Volume minimo scambiato (€)", 0, 1000000, 100000, 50000,
            help="Volume medio giornaliero minimo (in €)."
        )

        min_prox = st.sidebar.slider(
            "Prossimità minima ai Massimi 52W",
            min_value=0.50, max_value=1.0, value=0.85, step=0.01, format="%.0f%%",
            help="Filtro 'Trend Continuation'. Richiede che l'ETF sia entro questa percentuale del suo massimo a 52 settimane (es. 85% = max -15% drawdown)."
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
        market_trend = market_info.get("market_trend")
        date_str = analysis_date.strftime('%d/%m/%Y')

        # Controlli di sicurezza (invariati)
        if regime != "Favorevole al Rischio":
            vix_str = f"{vix:.2f}" if vix is not None else "N/D"
            st.sidebar.error(f"Regime: {regime} (VIX: {vix_str})")
            st.sidebar.warning(
                f"In questa data il mercato era in alta volatilità. Lo screener è disattivato per prudenza.")
            st.warning(
                f"Alla data del {date_str} il mercato era in modalità Avversa al Rischio. Lo screener non produce risultati per prudenza.")
            return

        if market_trend == "Ribassista":
            st.sidebar.error(f"Trend Mercato: {market_trend} (SP500 < MA200)")
            st.sidebar.warning(
                f"Il mercato generale (S&P 500) è in trend ribassista. Lo screener è disattivato per prudenza.")
            st.warning(
                f"Alla data del {date_str} il mercato generale era in trend ribassista. Lo screener non produce risultati per prudenza.")
            return

        st.sidebar.success(f"Regime: {regime} (VIX: {vix:.2f})\n\nTrend: {market_trend} (SP500 > MA200)")

        if enable_time_travel:
            st.info(f"Modalità Time Travel: L'analisi riflette i dati disponibili al **{date_str}**.")

        # === LOGICA DI ESECUZIONE MODIFICATA ===

        # Fase 2: Calcolo (LENTO, ma CACHATO)
        # Chiama la funzione senza il log. I log verranno restituiti.
        metrics_df, calc_log = calculate_all_metrics(
            st.session_state.pac_screener_raw_data,
            analysis_date
        )

        # Lista di log unificata per questa esecuzione
        # Inizia con i log della funzione cachata
        current_run_log = list(calc_log)

        # Fase 3: Filtraggio (VELOCE)
        # Passa la lista 'current_run_log' per aggiungere i log di filtraggio
        results = filter_and_rank_metrics(
            metrics_df,
            min_avg_value,
            min_prox,
            current_run_log
        )

        # Mostra i risultati (invariato)
        if not results.empty:
            top_etf = results.iloc[0]
            st.success(
                f"🏆 Candidato Migliore al {date_str}: **{top_etf['name']} ({top_etf['ticker']})**")
            st.subheader("Classifica Completa")
            df_display = results.rename(columns={
                "ticker": "Ticker", "isin": "ISIN", "name": "Nome", "final_score": "Punteggio Finale",
                "quality_score": "Score Qualità", "momentum_score": "Score Momentum",
                "breakout_score": "Score Breakout", "low_vol_score": "Score Low Volatility",
                "calmar_ratio": "Calmar Ratio (12M)", "roc_6m": "Rendimento 6M",
                "volatility_6m": "Volatilità 6M", "avg_value_eur": "Valore Scambiato (€)",
                "proximity_to_high": "Prox. Massimi 52W"
            })
            st.dataframe(
                df_display.style.format({
                    "Punteggio Finale": "{:.2f}", "Score Qualità": "{:.2f}", "Score Momentum": "{:.2f}",
                    "Score Breakout": "{:.2f}", "Score Low Volatility": "{:.2f}",
                    "Calmar Ratio (12M)": "{:.2f}", "Rendimento 6M": "{:+.2%}",
                    "Volatilità 6M": "{:.2%}", "Valore Scambiato (€)": "€{:,.0f}",
                    "Prox. Massimi 52W": "{:.1%}"
                }).background_gradient(cmap='Greens', subset=['Punteggio Finale', 'Score Breakout', 'Score Momentum',
                                                              'Prox. Massimi 52W'])
                .background_gradient(cmap='Reds_r', subset=['Score Qualità', 'Score Low Volatility']),
                width="stretch"
            )
        else:
            st.warning(
                "Nessun ETP ha soddisfatto i criteri alla data selezionata (es. Liquidità, Trend MA200, Momentum > 0, Vicinanza Massimi). "
                "Prova ad allargare i filtri (es. abbassare la Prossimità Massimi)."
            )

        with st.expander("🔍 Dettagli del Processo di Screening (Debug Log)"):
            st.markdown("##### Log del Caricamento Dati")
            st.markdown("\n".join(st.session_state.debug_log_loading))

            st.markdown("##### Log del Calcolo Metriche e Filtri (Questa Esecuzione)")
            st.markdown("\n".join(current_run_log))  # Mostra i log dell'esecuzione corrente

            st.markdown("##### Dettaglio Calcolo Metriche (dalla cache)")
            # Legge il log di dettaglio salvato dalla funzione cachata
            log_df_data = [log for log in st.session_state.debug_log_processing if isinstance(log, list)]
            if log_df_data:
                log_df = pd.DataFrame(log_df_data, columns=['Ticker', 'Esito'])
                st.dataframe(log_df, width="stretch")

    else:
        st.info("👈 Imposta la modalità e premi 'Carica/Aggiorna Dati Universo ETF' per iniziare.")