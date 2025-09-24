# -*- coding: utf-8 -*-
import streamlit as st
from datetime import date
from .pac_screener import screen_for_tactical_etfs, get_market_regime


def render_pac_screener_ui():
    """Renderizza la UI per il Screener Tattico con funzionalità Time Travel integrata."""
    st.title("🎯 Screener Tattico Avanzato (Europa)")
    st.caption("Scopri ETP con potenziale di breakout quotati sulle principali borse europee.")

    with st.expander("📖 Leggi la Metodologia"):
        st.markdown("""
           Questo screener è ottimizzato per la parte **satellite** di un portafoglio e utilizza un approccio multi-fattore per identificare ETP con alto potenziale a breve-medio termine.

           **1. Contesto di Mercato (Filtro VIX):** Lo screener opera solo in regimi di mercato favorevoli al rischio (VIX < 20). In caso di alta volatilità, consiglia cautela.
           **2. Filtro di Liquidità:** Vengono considerati solo ETP con un volume medio giornaliero scambiato superiore alla soglia impostata.
           **3. Punteggio Composito:** Basato su Qualità del Trend (Calmar Ratio), Compressione di Volatilità e Prossimità ai Massimi di 52 settimane.

           *Questo screener non costituisce una raccomandazione di investimento.*
           """)

    st.sidebar.header("⚙️ Parametri Screener")

    with st.sidebar.expander("⏳ Time Travel (Opzionale)"):
        enable_time_travel = st.checkbox("Abilita Time Travel")
        as_of_date = st.date_input(
            "Posizionati alla data del:",
            date.today(),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
            disabled=not enable_time_travel
        )

    analysis_date = as_of_date if enable_time_travel else date.today()
    st.sidebar.markdown("---")

    min_avg_value = st.sidebar.number_input(
        "Volume minimo scambiato (€)", 0, 1000000, 100000, 50000,
        help="Volume medio giornaliero minimo (in €)."
    )

    market_info = get_market_regime(as_of_date=analysis_date)
    vix = market_info.get("vix")
    regime = market_info.get("regime")

    if regime == "Favorevole al Rischio":
        st.sidebar.success(f"Regime di Mercato al {analysis_date.strftime('%d/%m/%Y')}: Favorevole (VIX: {vix:.2f})")
        if st.sidebar.button("▶️ Avvia Screener Tattico"):
            with st.spinner(f"Analisi del mercato europeo al {analysis_date.strftime('%d/%m/%Y')} in corso..."):
                results_df = screen_for_tactical_etfs(
                    discovery_limit=5000,
                    min_avg_value=min_avg_value,
                    as_of_date=analysis_date  # Passa la data di analisi
                )
                st.session_state.pac_screener_results = results_df
    else:
        vix_str = f"{vix:.2f}" if vix is not None else "N/D"
        st.sidebar.error(f"Regime di Mercato al {analysis_date.strftime('%d/%m/%Y')}: {regime} (VIX: {vix_str})")
        st.sidebar.warning("In questa data il mercato era in alta volatilità. Lo screener è disattivato per prudenza.")

    if enable_time_travel:
        st.info(
            f"Modalità Time Travel: L'analisi riflette i dati disponibili al **{analysis_date.strftime('%d/%m/%Y')}**.")

    results = st.session_state.get('pac_screener_results')
    if results is not None:
        if not results.empty:
            top_etf = results.iloc[0]
            st.success(
                f"🏆 Candidato Migliore al {analysis_date.strftime('%d/%m/%Y')}: **{top_etf['name']} ({top_etf['ticker']})**")
            st.subheader("Classifica Completa")
            df_display = results.rename(columns={
                "ticker": "Ticker", "isin": "ISIN", "name": "Nome", "final_score": "Punteggio Finale",
                "quality_score": "Score Qualità Trend", "squeeze_score": "Score Compressione",
                "proximity_score": "Score Prossimità Massimi", "calmar_ratio": "Calmar Ratio (6M)",
                "avg_value_eur": "Valore Scambiato (€)"
            })
            st.dataframe(
                df_display.style.format({
                    "Punteggio Finale": "{:.2f}", "Score Qualità Trend": "{:.2f}", "Score Compressione": "{:.2f}",
                    "Score Prossimità Massimi": "{:.2f}", "Calmar Ratio (6M)": "{:.2f}",
                    "Valore Scambiato (€)": "€{:,.0f}"
                }).background_gradient(cmap='Greens', subset=['Punteggio Finale', 'Score Qualità Trend'])
                .background_gradient(cmap='Reds', subset=['Score Compressione'])
                .background_gradient(cmap='Blues', subset=['Score Prossimità Massimi']),
                width="stretch"
            )
        elif 'pac_screener_results' in st.session_state:
            st.warning(
                "Nessun ETP ha soddisfatto i criteri. Prova ad allargare i filtri (es. abbassare il volume minimo).\n\n"
                "**Nota:** Durante il weekend, i dati sui volumi potrebbero essere incompleti o assenti, portando a zero risultati. "
                "Per risultati ottimali, si consiglia di eseguire lo screener durante i giorni feriali."
            )
        else:
            st.warning("Nessun ETP ha soddisfatto i criteri alla data selezionata.")
    else:
        st.info("Imposta i parametri e premi 'Avvia Screener Tattico' per iniziare.")