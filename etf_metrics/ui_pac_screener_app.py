# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import re
from .pac_screener import screen_for_tactical_etfs, get_market_regime


def render_pac_screener_ui():
    """Renderizza la UI per il nuovo Screener Tattico Avanzato."""
    st.title("🎯 Screener Tattico Avanzato")
    st.caption("Filtra per borsa e scopri ETF con potenziale di breakout.")

    with st.expander("📖 Leggi la Metodologia"):
        st.markdown("""
           Questo screener è ottimizzato per la parte **satellite** di un portafoglio e utilizza un approccio multi-fattore per identificare ETF con alto potenziale a breve-medio termine.

           **1. Contesto di Mercato (Filtro VIX):** Lo screener opera solo in regimi di mercato favorevoli al rischio (VIX < 20). In caso di alta volatilità, consiglia cautela.

           **2. Filtro di Liquidità:** Vengono considerati solo ETF con un volume medio giornaliero scambiato superiore alla soglia impostata, per garantire la negoziabilità.

           **3. Punteggio Composito:**
           - **Qualità del Trend (50%):** Misurato con il **Calmar Ratio** (Rendimento/Max Drawdown) per premiare trend stabili e con crolli contenuti.
           - **Compressione di Volatilità (30%):** Identifica ETF in una fase di consolidamento, che spesso precede un movimento di prezzo esplosivo.
           - **Prossimità ai Massimi (20%):** Premia gli ETF che si trovano vicini ai loro massimi di 52 settimane, pronti a rompere resistenze chiave.

           *Questo screener non costituisce una raccomandazione di investimento. L'analisi walk-forward è suggerita per una validazione più robusta.*
           """)

    st.sidebar.header("⚙️ Parametri Screener")

    # --- NUOVO: Filtro per Borse Valori ---
    EXCHANGE_MAP = {
        "Borsa Italiana (Milano)": ".MI",
        "London Stock Exchange": ".L",
        "Deutsche Börse (XETRA)": ".DE",
        "Euronext Amsterdam": ".AS"
    }
    selected_exchanges_names = st.sidebar.multiselect(
        "Filtra per Borse Valori",
        options=list(EXCHANGE_MAP.keys()),
    )
    selected_exchanges_suffixes = [EXCHANGE_MAP[name] for name in selected_exchanges_names]

    search_terms_input = st.sidebar.text_area(
        "Oppure analizza ISIN/Ticker Specifici",
        "",
        help="Lascia vuoto per scoprire i migliori dall'intero mercato in base ai filtri sopra."
    )
    st.sidebar.markdown("---")

    discovery_limit = st.sidebar.slider(
        "Universo ETF da scoprire", 200, 2000, 750, 50,
        help="Numero massimo di ETF da scoprire prima di applicare i filtri."
    )
    min_avg_value = st.sidebar.number_input(
        "Volume minimo scambiato (€)", 0, 1000000, 100000, 50000,
        help="Volume medio giornaliero minimo (in €)."
    )
    st.sidebar.caption("⚠️ Valori alti di 'Universo ETF' possono aumentare i tempi di analisi.")

    market_info = get_market_regime()
    vix = market_info.get("vix")
    regime = market_info.get("regime")

    if regime == "Favorevole al Rischio":
        st.sidebar.success(f"Regime di Mercato: Favorevole (VIX: {vix:.2f})")
        if st.sidebar.button("▶️ Avvia Screener Tattico"):
            search_terms = [term.strip().upper() for term in re.split(r'[,\n]', search_terms_input) if term.strip()]

            with st.spinner("Analisi in corso..."):
                results_df = screen_for_tactical_etfs(
                    discovery_limit=discovery_limit,
                    min_avg_value=min_avg_value,
                    specific_isins=search_terms or None,
                    selected_exchanges=selected_exchanges_suffixes  # Passa il filtro
                )
                st.session_state.pac_screener_results = results_df

    else:
        st.sidebar.error(f"Regime di Mercato: {regime} (VIX: {vix:.2f if vix else 'N/D'})")
        st.sidebar.warning("Il mercato è in una fase di alta volatilità. Lo screener è disattivato per prudenza.")

    results = st.session_state.get('pac_screener_results')
    if results is not None:
        if not results.empty:
            top_etf = results.iloc[0]
            st.success(
                f"🏆 **Candidato Tattico Migliore: {top_etf['name']} ({top_etf['ticker']})** con un punteggio di **{top_etf['final_score']:.2f}**")
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
                use_container_width=True
            )
        elif 'pac_screener_results' in st.session_state:
            st.warning(
                "Nessun ETF ha soddisfatto i criteri. Prova ad allargare i filtri (es. abbassare il volume minimo).\n\n"
                "**Nota:** Durante il weekend, i dati sui volumi potrebbero essere incompleti o assenti, portando a zero risultati. "
                "Per risultati ottimali, si consiglia di eseguire lo screener durante i giorni feriali."
            )
    else:
        st.info("Imposta i parametri nella barra laterale e premi 'Avvia Screener Tattico' per iniziare.")