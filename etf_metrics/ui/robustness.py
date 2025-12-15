# # -*- coding: utf-8 -*-
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from etf_metrics.core.robustness import (
#     run_monte_carlo_permutation_test,
#     run_detrended_analysis,
#     run_walk_forward_analysis
# )
# from etf_metrics.clients.yahoo_client import get_series
#
#
# def render_robustness_ui():
#     st.title("🛡️ Robustness Testing (Metodo Aronson)")
#     st.caption("Test scientifici per validare la significatività statistica della strategia.")
#
#     with st.expander("📚 Che cosa sono questi test?"):
#         st.markdown("""
#         Questi strumenti si basano sui principi di **"Evidence-Based Technical Analysis"** di David Aronson.
#
#         1. **Analisi Detrended**: Rimuove il trend di fondo del mercato. Se la strategia guadagna anche su dati a "somma zero", ha un vero potere predittivo (Alpha) e non sta solo cavalcando un mercato rialzista.
#         2. **Monte Carlo Permutation Test**: Mescola casualmente i rendimenti giornalieri. Crea universi paralleli per vedere se i risultati sono dovuti alla fortuna.
#         3. **Walk-Forward Analysis**: Verifica la stabilità della strategia su finestre temporali successive.
#         """)
#
#     st.sidebar.header("⚙️ Setup Test")
#
#     tickers_input = st.sidebar.text_area("Ticker per Test", "NVDA", height=70)
#     tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
#
#     start_date = st.sidebar.date_input("Data Inizio", pd.to_datetime("2021-01-01"))
#     initial_cap = st.sidebar.number_input("Capitale Iniziale", 1000, 100000, 10000)
#
#     tab1, tab2, tab3 = st.tabs(["📉 Detrended Analysis", "🎲 Monte Carlo", "🚶 Walk-Forward"])
#
#     with tab1:
#         st.subheader("Test su Dati Detrended (Zero Drift)")
#         st.markdown("Verifica se la strategia funziona rimuovendo il trend di fondo.")
#
#         if st.button("Avvia Detrended Test"):
#             if not tickers:
#                 st.warning("Inserisci almeno un ticker.")
#             else:
#                 with st.spinner("Elaborazione dati sintetici e ricalcolo indicatori..."):
#                     trades, final, detrended_data = run_detrended_analysis(tickers, str(start_date), initial_cap)
#
#                 if detrended_data and tickers[0] in detrended_data:
#                     st.write("### 🔍 Verifica Visiva: Reale vs Detrended")
#                     ticker_to_plot = tickers[0]
#
#                     try:
#                         orig_series = get_series(ticker_to_plot, period="5y")
#                         if orig_series is not None:
#                             detr_series = detrended_data[ticker_to_plot]['Close']
#                             common_idx = orig_series.index.intersection(detr_series.index)
#
#                             chart_df = pd.DataFrame({
#                                 "Originale (Con Trend)": orig_series.loc[common_idx],
#                                 "Detrended (Senza Trend)": detr_series.loc[common_idx]
#                             })
#
#                             chart_df = (chart_df / chart_df.iloc[0]) * 100
#                             st.line_chart(chart_df)
#                     except Exception as e:
#                         st.warning(f"Impossibile generare grafico di debug: {e}")
#
#                 ret = ((final - initial_cap) / initial_cap) * 100
#                 color = "green" if ret > 0 else "red"
#                 st.markdown(f"### Risultato PnL Detrended: :{color}[{ret:.2f}%]")
#
#                 if ret > 0:
#                     st.success("✅ La strategia genera profitti anche senza trend di mercato! (Forte validità)")
#                 else:
#                     st.warning(
#                         "⚠️ La strategia perde o è in pareggio su dati detrended. Dipende fortemente dal trend rialzista.")
#
#                 if not trades.empty:
#                     st.dataframe(trades)
#                 else:
#                     st.info(
#                         "Nessun trade generato. La rimozione del trend potrebbe aver appiattito l'ADX o i segnali di breakout.")
#
#     with tab2:
#         st.subheader("Permutation Test (Significatività)")
#         n_sims = st.slider("Numero Simulazioni", 10, 100, 20)
#
#         if st.button("Avvia Monte Carlo"):
#             if not tickers:
#                 st.warning("Inserisci almeno un ticker.")
#             else:
#                 with st.spinner(f"Esecuzione di {n_sims} backtest su dati permutati..."):
#                     res = run_monte_carlo_permutation_test(tickers, str(start_date), n_sims, initial_cap)
#
#                 real_ret = res['real_return'] * 100
#                 p_val = res['p_value']
#
#                 col1, col2 = st.columns(2)
#                 col1.metric("Rendimento Reale", f"{real_ret:.2f}%")
#                 col2.metric("P-Value", f"{p_val:.4f}", help="P-Value < 0.05 indica significatività statistica.")
#
#                 if p_val < 0.05:
#                     st.success(f"🎉 **Statisticamente Significativo!** La strategia batte il caso.")
#                 else:
#                     st.error(f"❌ **Non Significativo.** Il risultato rientra nella varianza casuale.")
#
#                 if res['monte_carlo_returns']:
#                     fig, ax = plt.subplots()
#                     ax.hist([r * 100 for r in res['monte_carlo_returns']], bins=15, alpha=0.7, label='Random Runs')
#                     ax.axvline(real_ret, color='red', linestyle='dashed', linewidth=2, label='Tua Strategia')
#                     ax.set_title("Distribuzione Rendimenti Random vs Reale")
#                     ax.legend()
#                     st.pyplot(fig)
#
#     with tab3:
#         st.subheader("Walk-Forward Stability")
#         if st.button("Avvia Walk-Forward"):
#             with st.spinner("Analisi finestre temporali..."):
#                 wf_df = run_walk_forward_analysis(tickers, initial_cap)
#
#             if not wf_df.empty:
#                 st.dataframe(wf_df)
#                 st.bar_chart(wf_df.set_index("Window End")['PnL'])
#
#                 win_windows = len(wf_df[wf_df['PnL'] > 0])
#                 tot_windows = len(wf_df)
#                 if tot_windows > 0:
#                     st.metric("Finestre Profittevoli", f"{win_windows}/{tot_windows} ({win_windows / tot_windows:.0%})")
#             else:
#                 st.warning("Dati insufficienti per Walk-Forward.")