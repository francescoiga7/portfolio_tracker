# -*- coding: utf-8 -*-
import streamlit as st
from datetime import date
import pandas as pd

from etf_metrics.core.pac_screener import (
    fetch_screener_data,
    calculate_all_metrics,
    filter_and_rank_metrics,
    get_market_regime,
)
from etf_metrics.core.data_manager import MarketDataManager

ALGORITHM_WEIGHTS = {
    "Momentum Score": 0.40,
    "Trend Quality Score (Calmar)": 0.05,
    "Breakout Score (Volatilità Compressa)": 0.50,
    "Low Volatility Score": 0.05
}


@st.cache_data(show_spinner=False, ttl=60 * 15)
def _cached_market_regime(as_of_date):
    return get_market_regime(as_of_date=as_of_date)


@st.cache_data(show_spinner="Caricamento e Gestione Dati (DB Cache)...", ttl=60 * 30)
def _cached_fetch(specific_isins):
    sis = list(specific_isins) if specific_isins else None
    return fetch_screener_data(specific_isins=sis)


@st.cache_data(show_spinner="Calcolo metriche (Engine: Polars)...", ttl=60 * 15)
def _cached_metrics(fetched_data, as_of_date):
    return calculate_all_metrics(fetched_data, as_of_date)


def render_pac_screener_ui():
    st.title("🎯 Screener ETF")
    st.caption("Scopri ETP con potenziale di breakout quotati sulle principali borse europee.")

    if 'debug_log_loading' not in st.session_state:
        st.session_state.debug_log_loading = []

    with st.expander("📖 Metodologia e Analisi Quantitativa (Approccio 'Breakout Focused')"):
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
              | **Breakout/Compressione** | Volatilità di Breve Termine vs. Lungo Termine | **(Peso Massimo)** |
              | **Momentum Assoluto** | Performance a 6 e 12 Mesi (ROC) | **(Peso Alto)** |
              | **Qualità del Trend** | Calmar Ratio (12 Mesi) | **(Peso Minimo)** |
              | **Bassa Volatilità Relativa** | Volatilità a 6 Mesi | **(Peso Minimo)** |

              ---

              ### 3. Punteggio Composito e Pesi

              Il **Punteggio Finale** è una media pesata dei singoli punteggi. 

              * **Formula:** $Punteggio \\: Finale = \\sum (Score_Fattore \\times Peso_Fattore)$

              I pesi attuali assegnati sono:
              * **Breakout Score:** {ALGORITHM_WEIGHTS['Breakout Score (Volatilità Compressa)'] * 100:.0f}%
              * **Momentum Score:** {ALGORITHM_WEIGHTS['Momentum Score'] * 100:.0f}%
              * **Trend Quality Score (Calmar):** {ALGORITHM_WEIGHTS['Trend Quality Score (Calmar)'] * 100:.0f}%
              * **Low Volatility Score:** {ALGORITHM_WEIGHTS['Low Volatility Score'] * 100:.0f}%
              """)

    st.sidebar.header("⚙️ Impostazioni")
    with st.sidebar.expander("1. Carica Universo Dati", expanded=True):
        source_mode = st.radio("Modalità di Ricerca", ["Scoperta Automatica Universo", "Inserisci ISIN Specifici"])
        specific_isins_input = ""
        if source_mode == "Inserisci ISIN Specifici":
            specific_isins_input = st.text_area("Elenco ISIN (uno per riga)", height=150)
        if st.button("Carica/Aggiorna Dati"):
            st.cache_data.clear()
            st.session_state.pac_screener_raw_data = None
            st.session_state.debug_log_loading = []
            specific_isins_tuple = None
            if source_mode == "Inserisci ISIN Specifici" and specific_isins_input.strip():
                specific_isins_tuple = tuple(
                    isin.strip().upper() for isin in specific_isins_input.split('\n') if isin.strip()
                )
            data, fetch_logs = _cached_fetch(specific_isins_tuple)
            st.session_state.pac_screener_raw_data = data
            st.session_state.debug_log_loading = list(fetch_logs)
            st.rerun()

    with st.sidebar.expander("🚫 Ticker senza dati (cache negativa)", expanded=False):
        _dm = MarketDataManager()
        _failed_df = _dm.get_failed_tickers_info()
        if _failed_df is None or _failed_df.empty:
            st.caption("Nessun ticker in blacklist: tutti gli strumenti finora scaricati hanno dati.")
        else:
            st.caption(f"**{len(_failed_df)}** ticker 'no data found' vengono saltati ai download "
                       "(nessuna chiamata di rete). Vengono rimossi dalla blacklist se in futuro "
                       "producono dati.")
            st.dataframe(_failed_df, height=220, width="stretch", hide_index=True)
            if st.button("🧹 Reset blacklist", key="reset_failed_tickers"):
                removed = _dm.clear_failed_tickers()
                st.success(f"Rimossi {removed} ticker dalla blacklist: verranno ritentati al prossimo download.")
                st.rerun()

    if 'pac_screener_raw_data' in st.session_state and st.session_state.pac_screener_raw_data:
        st.sidebar.header("2. Filtri e Time Travel")
        min_avg_value = st.sidebar.number_input("Volume minimo scambiato (€)", 0, 1000000, 100000, 50000)
        min_prox = st.sidebar.slider("Prossimità minima ai Massimi 52W", 0.50, 1.0, 0.85, 0.01, format="%.0f%%")
        enable_time_travel = st.sidebar.checkbox("Abilita Time Travel")
        as_of_date = st.sidebar.date_input("Posizionati alla data del:", date.today(), min_value=date(2020, 1, 1), max_value=date.today(), disabled=not enable_time_travel)
        analysis_date = as_of_date if enable_time_travel else date.today()

        market_info = _cached_market_regime(analysis_date)
        vix = market_info.get("vix")
        regime = market_info.get("regime")
        market_trend = market_info.get("market_trend")
        date_str = analysis_date.strftime('%d/%m/%Y')

        if regime != "Favorevole al Rischio" or market_trend == "Ribassista":
            st.warning(f"Alla data del {date_str} il mercato era in modalità Avversa o Trend Ribassista. Lo screener è disattivato.")
            return

        st.sidebar.success(f"Regime: {regime} (VIX: {vix:.2f})\n\nTrend: {market_trend}")

        metrics_df, calc_log = _cached_metrics(st.session_state.pac_screener_raw_data, analysis_date)
        current_run_log = list(calc_log)
        results = filter_and_rank_metrics(metrics_df, min_avg_value, min_prox, current_run_log)

        if not results.empty:
            top_etf = results.iloc[0]
            st.success(f"🏆 Candidato Migliore al {date_str}: **{top_etf['name']} ({top_etf['ticker']})**")
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
                }).background_gradient(cmap='Greens', subset=['Punteggio Finale', 'Score Breakout', 'Score Momentum', 'Prox. Massimi 52W'])
                .background_gradient(cmap='Reds_r', subset=['Score Qualità', 'Score Low Volatility']),
                width="stretch"
            )
        else:
            st.warning("Nessun ETP ha soddisfatto i criteri alla data selezionata.")

        with st.expander("🔍 Debug Log"):
            st.markdown("##### Log Caricamento")
            st.markdown("\n".join(st.session_state.debug_log_loading))
            st.markdown("##### Log Calcolo")
            st.markdown("\n".join(current_run_log))
    else:
        st.info("👈 Imposta la modalità e premi 'Carica/Aggiorna Dati' per iniziare.")
