# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import date
from etf_metrics.clients.yahoo_client import get_series
from etf_metrics.core.trading import analyze_ticker
from etf_metrics.core.automated_backtest import run_market_aware_backtest

DEFAULT_TRADING_LIST = """NVDA
TSLA
AMD
COIN
MARA
PLTR
META
AMZN
NFLX
QQQ
TQQQ
SQQQ"""


def render_trading_ui():
    st.title("💹 Trading")
    st.caption("Algoritmo quantitativo per Swing Trading e Breakout di Volatilità.")

    with st.expander("🧠 Logica Operativa e Backtest"):
        st.markdown("""
        Questa sezione identifica setup di **Breakout** e **Reversion** basati su volatilità e volumi.

        **Funzionalità Time Travel:**
        Attivando il Backtest nella sidebar, puoi "tornare indietro nel tempo".
        L'algoritmo vedrà SOLO i dati disponibili fino a quella data. È utile per rispondere alla domanda:
        *"Se avessi usato questo algoritmo il 15 ottobre scorso, mi avrebbe dato il segnale giusto?"*

        **Legenda Segnali:**
        * 🟢 **LONG_BREAKOUT:** Prezzo rompe la Banda di Bollinger superiore con volumi.
        * 🔵 **LONG_DIP:** Trend rialzista ma prezzo in ritracciamento (RSI basso).
        * 🔴 **SHORT_BREAKDOWN:** Rottura violenta al ribasso dei supporti.
        """)

    st.sidebar.header("⚙️ Radar Settings")

    st.sidebar.subheader("⏳ Macchina del Tempo")
    enable_time_travel = st.sidebar.checkbox("Abilita Backtest Storico", value=False)

    analysis_date = date.today()
    if enable_time_travel:
        analysis_date = st.sidebar.date_input(
            "Analizza come se fosse il:",
            date.today(),
            min_value=date(2020, 1, 1),
            max_value=date.today()
        )
        st.sidebar.warning(f"⚠️ Analisi congelata al: {analysis_date.strftime('%d/%m/%Y')}")
    else:
        st.sidebar.caption("Analisi in tempo reale (Dati odierni)")

    tickers_input = st.sidebar.text_area("Watchlist (Ticker Yahoo)", DEFAULT_TRADING_LIST, height=200)

    if st.sidebar.button("🔥 Scansiona Mercato"):
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        cutoff_date = pd.to_datetime(analysis_date)

        for i, ticker in enumerate(tickers):
            status_text.text(f"Analisi {ticker} al {analysis_date}...")

            full_df = get_series(ticker, period="5y", as_dataframe=True)

            if full_df is not None and not full_df.empty:
                df_slice = full_df[full_df.index <= cutoff_date].copy()

                if len(df_slice) > 200:
                    signal_data = analyze_ticker(ticker, df_slice)

                    if signal_data and signal_data['signal'] != "NEUTRAL":
                        results.append(signal_data)
                else:
                    pass

            progress_bar.progress((i + 1) / len(tickers))

        status_text.empty()
        progress_bar.empty()

        date_label = analysis_date.strftime('%d/%m/%Y')
        if enable_time_travel:
            st.subheader(f"📅 Risultati Storici al {date_label}")
        else:
            st.subheader(f"📅 Segnali Live ({date_label})")

        if not results:
            st.info(f"Nessun segnale operativo trovato alla data {date_label}.")
            if enable_time_travel:
                st.caption(
                    "Suggerimento: Prova a cambiare data. I segnali di breakout sono rari e durano pochi giorni.")
            return

        st.success(f"Trovate {len(results)} opportunità operative!")

        results.sort(key=lambda x: x['confidence'], reverse=True)

        for res in results:
            signal_color = "green" if "LONG" in res['signal'] else "red"

            with st.container():
                st.markdown(f"### {res['ticker']} : :{signal_color}[{res['signal']}]")
                st.caption(f"Prezzo alla data {date_label}: **{res['price']:.2f}** | Confidence: {res['confidence']}/5")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prezzo Ingresso", f"{res['price']:.2f}")

                sl_delta = res['stop_loss'] - res['price']
                tp_delta = res['take_profit'] - res['price']

                c2.metric("Stop Loss (ATR)", f"{res['stop_loss']:.2f}", delta=f"{sl_delta:.2f}", delta_color="inverse")
                c3.metric("Take Profit", f"{res['take_profit']:.2f}", delta=f"{tp_delta:.2f}", delta_color="normal")

                rsi_val = res['indicators']['RSI']
                rsi_state = "Ipercomprato" if rsi_val > 70 else "Ipervenduto" if rsi_val < 30 else "Neutrale"
                c4.metric("RSI (14)", f"{rsi_val:.1f}", delta=rsi_state, delta_color="off")

                with st.expander("Dettagli Strategici"):
                    st.write("**Motivazioni del Segnale:**")
                    for reason in res['reasons']:
                        st.markdown(f"- {reason}")

                    st.markdown("---")
                    st.markdown(f"**Dati Tecnici al {date_label}:**")
                    st.markdown(f"- *Bande Bollinger Width:* {res['indicators']['BB_Width']:.4f} (Compressione)")
                    st.markdown(f"- *Volume Relativo:* {res['indicators']['Vol_Rel']:.1f}x media")

                st.markdown("---")

    else:
        if enable_time_travel:
            st.info(f"👈 Imposta la data storica ({analysis_date.strftime('%d/%m/%Y')}) e premi 'Scansiona'.")
        else:
            st.info("👈 Inserisci i ticker e premi 'Scansiona Mercato' per l'analisi live.")

        st.markdown("---")
        st.subheader("🤖 Backtest Strategia")
        st.caption(
            "Simula l'acquisto sui segnali. Puoi scegliere tra Trailing Stop dinamico o uscita fissa a Target.")

        default_backtest_list = "NVDA\nTSLA\nAMD\nCOIN\nMARA\nPLTR\nMETA\nAMZN\nNFLX\nQQQ\nTQQQ\nSQQQ"
        backtest_tickers_txt = st.text_area("Ticker per Backtest (dal 2021)", default_backtest_list, height=100)

        # NUOVO FLAG
        use_tp_only = st.checkbox("🎯 Usa Strategia 'Solo Take Profit'",
                                  help="Se attivo, ignora il Trailing Stop. Vende SOLTANTO se il prezzo tocca il Take Profit (Entry + 4*ATR). Più rischioso ma evita stop prematuri.")

        if st.button("Esegui Backtest"):
            t_list = [t.strip().upper() for t in backtest_tickers_txt.split('\n') if t.strip()]

            strategy_name = "Target Fisso (Take Profit)" if use_tp_only else "Trailing Stop Dinamico"
            with st.spinner(f"Simulazione con strategia {strategy_name} in corso..."):
                # Passiamo il flag alla funzione
                df_trades, final_cap = run_market_aware_backtest(t_list, start_date="2021-01-01",
                                                                 use_tp_only=use_tp_only)

            if not df_trades.empty:
                total_return = ((final_cap - 1000) / 1000) * 100

                c1, c2, c3 = st.columns(3)
                c1.metric("Capitale Finale", f"€{final_cap:,.2f}")
                c2.metric("Rendimento", f"{total_return:.2f}%", delta_color="normal")

                closed_trades = df_trades[df_trades['Action'] == 'SELL (100%)']
                if not closed_trades.empty:
                    wins = len(closed_trades[closed_trades['PnL_Eur'] > 0])
                    total_closed = len(closed_trades)
                    win_rate = (wins / total_closed) * 100
                    c3.metric("Win Rate", f"{win_rate:.1f}% ({wins}/{total_closed})")

                st.subheader("Giornale delle Operazioni")

                def style_trades(row):
                    action = row['Action']
                    if "BUY" in action: return ['background-color: #000000'] * len(row)
                    if "SELL" in action:
                        if "PROFIT" in str(row['Reason']) or "TARGET" in str(row['Reason']): return [
                            'background-color: #f0fff4; color: green'] * len(row)
                        return ['background-color: #fff5f5; color: red'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    df_trades.style.apply(style_trades, axis=1)
                    .format({"Price": "{:.2f}", "PnL_Eur": "{:+.2f}", "PnL_Pct": "{:+.2f}%", "Capital": "€{:,.0f}"}),
                    use_container_width=True
                )
            else:
                st.warning("Nessun trade generato nel periodo.")