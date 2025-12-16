# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from etf_metrics.core.robustness import (
    run_monte_carlo_permutation_test,
    run_detrended_analysis,
    run_walk_forward_analysis
)
from etf_metrics.clients.yahoo_client import get_series


def render_robustness_ui():
    st.title("🛡️ Robustness Testing")
    st.caption("Validazione scientifica della strategia' (SMA130 + Momentum).")

    with st.expander("📚 Guida ai Test di Robustezza"):
        st.markdown("""
        Questi test verificano se i risultati della strategia sono dovuti all'abilità (Alpha) o alla fortuna/trend.

        1. **Analisi Detrended (Validità)**: Rimuove il trend rialzista di fondo dal mercato. Se la strategia guadagna anche quando il mercato è piatto (somma zero), significa che le regole di ingresso/uscita hanno un valore predittivo reale.
        2. **Monte Carlo Permutation (Significatività)**: Mescola casualmente l'ordine dei rendimenti giornalieri. Crea universi alternativi per vedere se il risultato reale è statisticamente superiore al caso.
        3. **Walk-Forward (Stabilità)**: Applica la strategia su finestre temporali successive (rolling) per verificare la costanza dei risultati nel tempo.
        """)

    st.sidebar.header("⚙️ Setup Test")

    # Default tickers aggiornati per la strategia momentum
    default_tickers = "NVDA\nAMD\nTSLA\nAAPL\nMSFT\nAMZN\nGOOGL\nMETA\nNFLX"
    tickers_input = st.sidebar.text_area("Ticker per Test", default_tickers, height=150)
    tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

    start_date = st.sidebar.date_input("Data Inizio", pd.to_datetime("2018-01-01"))
    initial_cap = st.sidebar.number_input("Capitale Iniziale", 1000, 1000000, 10000)

    tab1, tab2, tab3 = st.tabs(["📉 Detrended Analysis", "🎲 Monte Carlo", "🚶 Walk-Forward"])

    # --- TAB 1: DETRENDED ANALYSIS ---
    with tab1:
        st.subheader("Test su Dati Detrended (Zero Drift)")
        st.info("Questo test rimuove il 'Vento a favore'. La strategia batte il mercato anche se il mercato non sale?")

        if st.button("Avvia Detrended Test"):
            if not tickers:
                st.warning("Inserisci almeno un ticker.")
            else:
                with st.spinner("Creazione universi 'Zero-Mean' e backtest..."):
                    trades, final, detrended_data = run_detrended_analysis(tickers, str(start_date), initial_cap)

                # Grafico di verifica (Originale vs Detrended) per il primo ticker
                if detrended_data and len(tickers) > 0:
                    ticker_debug = tickers[0]
                    if ticker_debug in detrended_data:
                        st.write(f"### 🔍 Verifica Visiva: {ticker_debug} (Reale vs Detrended)")
                        try:
                            orig_series = get_series(ticker_debug, period="10y")
                            if orig_series is not None:
                                detr_series = detrended_data[ticker_debug]['Close']
                                common_idx = orig_series.index.intersection(detr_series.index)

                                # Normalizza a 100 per confronto
                                df_chart = pd.DataFrame({
                                    "Originale (Con Trend)": orig_series.loc[common_idx],
                                    "Detrended (Senza Trend)": detr_series.loc[common_idx]
                                })
                                df_chart = (df_chart / df_chart.iloc[0]) * 100
                                st.line_chart(df_chart)
                        except Exception as e:
                            st.warning(f"Impossibile generare grafico debug: {e}")

                ret = ((final - initial_cap) / initial_cap) * 100
                color = "green" if ret > 0 else "red"

                st.markdown(f"### Risultato PnL Detrended: :{color}[{ret:.2f}%]")

                if ret > 0:
                    st.success(
                        "✅ **PASSATO:** La strategia genera profitti anche senza trend di mercato! (Forte validità)")
                else:
                    st.error(
                        "⚠️ **FALLITO:** La strategia perde su dati detrended. Il profitto dipende quasi interamente dal trend rialzista del mercato.")

                if not trades.empty:
                    st.write("Giornale Operazioni (Simulazione Detrended):")
                    st.dataframe(trades)
                else:
                    st.warning(
                        "Nessun trade generato sui dati detrended (i segnali SMA130/Momentum potrebbero essere spariti).")

    # --- TAB 2: MONTE CARLO ---
    with tab2:
        st.subheader("Permutation Test (Significatività)")
        st.info("Confronta il risultato reale contro N universi casuali.")

        n_sims = st.slider("Numero Simulazioni", 10, 200, 50)

        if st.button("Avvia Monte Carlo"):
            if not tickers:
                st.warning("Inserisci almeno un ticker.")
            else:
                progress_text = "Simulazione in corso. Attendi..."
                my_bar = st.progress(0, text=progress_text)

                # Wrapper per aggiornare la barra se necessario, ma qui chiamiamo la funzione core diretta
                # Nota: run_monte_carlo_permutation_test è sincrona e potrebbe impiegare tempo
                with st.spinner(f"Esecuzione di {n_sims} backtest su dati permutati..."):
                    res = run_monte_carlo_permutation_test(tickers, str(start_date), n_sims, initial_cap)

                my_bar.empty()

                real_ret = res['real_return'] * 100
                p_val = res['p_value']
                sim_rets = [r * 100 for r in res['monte_carlo_returns']]

                col1, col2, col3 = st.columns(3)
                col1.metric("Rendimento Reale", f"{real_ret:.2f}%")
                col2.metric("P-Value", f"{p_val:.4f}",
                            help="Probabilità che il risultato sia casuale. < 0.05 è ottimo.")
                col3.metric("Simulazioni", n_sims)

                if p_val < 0.05:
                    st.success(
                        f"🎉 **Statisticamente Significativo!** (P-Value {p_val:.3f} < 0.05). La strategia batte il caso.")
                elif p_val < 0.10:
                    st.warning(
                        f"⚠️ **Marginale.** (P-Value {p_val:.3f}). C'è qualche evidenza di abilità, ma non conclusiva.")
                else:
                    st.error(
                        f"❌ **Non Significativo.** (P-Value {p_val:.3f}). Il risultato rientra nella varianza casuale.")

                if sim_rets:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(sim_rets, bins=20, alpha=0.7, color='gray', label='Random Runs (Luck)')
                    ax.axvline(real_ret, color='red', linestyle='dashed', linewidth=2, label='Tua Strategia (Skill?)')
                    ax.set_title("Distribuzione Rendimenti: Strategia vs Caso")
                    ax.set_xlabel("Rendimento Totale %")
                    ax.legend()
                    st.pyplot(fig)

    # --- TAB 3: WALK FORWARD ---
    with tab3:
        st.subheader("Walk-Forward Stability")
        st.info("Verifica la costanza dei risultati su finestre temporali scorrevoli (es. ogni 6 mesi).")

        test_window = st.slider("Durata Finestra Test (Mesi)", 3, 24, 6)

        if st.button("Avvia Walk-Forward"):
            with st.spinner("Analisi finestre temporali..."):
                wf_df = run_walk_forward_analysis(tickers, initial_cap, test_months=test_window)

            if not wf_df.empty:
                st.write(f"### Risultati per finestre di {test_window} mesi")

                # Metriche aggregate
                positive_windows = len(wf_df[wf_df['PnL'] > 0])
                total_windows = len(wf_df)
                win_rate = (positive_windows / total_windows) * 100 if total_windows > 0 else 0
                avg_roi = wf_df['ROI %'].mean()

                c1, c2, c3 = st.columns(3)
                c1.metric("Finestre Positive", f"{positive_windows}/{total_windows}")
                c2.metric("Win Rate Temporale", f"{win_rate:.1f}%")
                c3.metric("ROI Medio per Finestra", f"{avg_roi:.2f}%")

                # Grafico
                st.bar_chart(wf_df.set_index("Window End")['ROI %'])

                st.dataframe(wf_df.style.format({"PnL": "{:.2f}", "ROI %": "{:.2f}%"}))
            else:
                st.warning("Dati insufficienti per generare finestre Walk-Forward.")