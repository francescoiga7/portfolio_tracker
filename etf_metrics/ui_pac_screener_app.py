# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from .pac_screener import screen_for_breakout_etfs


def render_pac_screener_ui():
    """Renderizza la UI per il nuovo Screener "Breakout Potential"."""
    st.title("🚀 Screener Potenziale Breakout")
    st.caption("Identifica gli ETF pronti per un potenziale movimento esplosivo di prezzo.")

    with st.expander("📖 Leggi la Metodologia"):
        st.markdown("""
        Questo strumento è progettato per la parte **satellite** di un portafoglio e cerca ETF con alto potenziale di crescita a breve-medio termine. La classifica si basa su un **punteggio composito**:
        - **Compressione di Volatilità (50%)**: Identifica ETF in una fase di consolidamento e bassa volatilità, che spesso precede un movimento di prezzo esplosivo.
        - **Forza Relativa (30%)**: Premia gli ETF che hanno iniziato a sovraperformare il mercato azionario globale, indicando un forte interesse d'acquisto.
        - **Prossimità al Breakout (20%)**: Premia gli ETF che si trovano vicini ai loro massimi di 52 settimane, pronti a rompere importanti livelli di resistenza.
        """)

    st.sidebar.header("⚙️ Parametri Screener")
    discovery_limit = st.sidebar.slider(
        "Universo ETF da analizzare",
        min_value=200, max_value=2000, value=750, step=50,
        help="Numero massimo di ETF da scoprire. Valori più alti richiedono più tempo."
    )
    # AVVISO DI SICUREZZA
    st.sidebar.caption(
        "⚠️ Valori alti (>1000) aumentano i tempi di analisi e il rischio di errori temporanei dal fornitore di dati.")

    if st.sidebar.button("▶️ Cerca Opportunità di Breakout"):
        with st.spinner(
                f"Ricerca di candidati al breakout in corso... L'operazione potrebbe richiedere alcuni minuti."):
            results_df = screen_for_breakout_etfs(discovery_limit)
            st.session_state.pac_screener_results = results_df

    results = st.session_state.get('pac_screener_results')
    if results is not None and not results.empty:
        top_etf = results.iloc[0]
        st.success(
            f"🏆 **ETF con maggior potenziale di breakout: {top_etf['name']} ({top_etf['ticker']})** con un punteggio finale di **{top_etf['final_score']:.2f}**")

        st.subheader("Classifica Completa")

        df_display = results.rename(columns={
            "ticker": "Ticker", "name": "Nome", "final_score": "Punteggio Breakout",
            "squeeze_score": "Score Compressione", "rs_score": "Score Forza Relativa",
            "proximity_score": "Score Prossimità Massimi",
            "price": "Prezzo Attuale (€)", "high_52w": "Massimo 52 Settimane (€)"
        })

        st.dataframe(
            df_display.style
            .format({
                "Punteggio Breakout": "{:.2f}", "Score Compressione": "{:.2f}",
                "Score Forza Relativa": "{:.2f}", "Score Prossimità Massimi": "{:.2f}",
                "Prezzo Attuale (€)": "€{:,.2f}", "Massimo 52 Settimane (€)": "€{:,.2f}"
            })
            .background_gradient(cmap='Greens', subset=['Punteggio Breakout'])
            .background_gradient(cmap='Reds', subset=['Score Compressione'])
            .background_gradient(cmap='Blues', subset=['Score Forza Relativa', 'Score Prossimità Massimi']),
            use_container_width=True
        )
    else:
        st.info("Imposta i parametri nella barra laterale e premi 'Cerca Opportunità di Breakout' per iniziare.")