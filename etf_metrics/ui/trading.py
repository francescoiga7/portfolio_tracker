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

    with st.expander("🧠 Logica Operativa"):
        st.markdown("""
        Questa strategia cerca il compromesso perfetto tra trend di lungo periodo e reattività.

        **Regole Scanner & Backtest:**
        1.  **Filtro Trend (On/Off):** **SMA 130 Giorni**.
            * Se Prezzo > SMA 130 -> Mercato Rialzista (Si cercano acquisti).
            * Se Prezzo < SMA 130 -> Mercato Ribassista (Si Vende/Flat).

        2.  **Motore di Ranking (Selezione):** **Momentum a 6 Mesi**.
            * Tra i titoli sopra la SMA 130, si scelgono quelli con il Momentum (Performance a 126gg) più alto.
            * Ingresso solo se Momentum > 0.

        **In Sintesi:** Compriamo i titoli più forti, ma solo se sono in un trend strutturalmente sano. Usciamo immediatamente se il trend si rompe (chiusura sotto SMA 130).
        """)

    st.sidebar.header("⚙️ Configurazione")

    mode = st.sidebar.radio(
        "Modalità Operativa",
        ["📡 Scanner Segnali", "🤖 Automated Backtest"],
        help="Scanner live o simulazione storica della strategia."
    )

    st.sidebar.divider()

    tickers_input = st.sidebar.text_area("Watchlist (Ticker Yahoo)", DEFAULT_TRADING_LIST, height=200)
    tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

    analysis_date = date.today()
    start_date_backtest = date(2021, 1, 1)

    if mode == "📡 Scanner Segnali":
        st.sidebar.subheader("⏳ Time Travel")
        enable_time_travel = st.sidebar.checkbox("Abilita Analisi Storica", value=False)
        if enable_time_travel:
            analysis_date = st.sidebar.date_input(
                "Data Analisi:",
                date.today(),
                min_value=date(2020, 1, 1),
                max_value=date.today()
            )
            st.sidebar.warning(f"Analisi congelata al: {analysis_date.strftime('%d/%m/%Y')}")
        else:
            st.sidebar.caption("Analisi su dati Odierni.")

    elif mode == "🤖 Automated Backtest":
        st.sidebar.subheader("⏳ Periodo Simulazione")
        start_date_backtest = st.sidebar.date_input("Data Inizio:", pd.to_datetime("2018-01-01"))
        st.sidebar.info("Simulazione portafoglio (max 4 posizioni, ribilanciamento mensile).")


    if mode == "📡 Scanner Segnali":
        st.subheader("📡 Scanner Strategia")

        if st.button("🔥 Scansiona Watchlist"):
            if not tickers:
                st.warning("Inserisci almeno un ticker nella sidebar.")
                return

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            cutoff_date = pd.to_datetime(analysis_date)

            for i, ticker in enumerate(tickers):
                status_text.text(f"Analisi {ticker}...")
                full_df = get_series(ticker, period="2y", as_dataframe=True)

                if full_df is not None and not full_df.empty:
                    df_slice = full_df[full_df.index <= cutoff_date].copy()
                    if len(df_slice) > 135:
                        signal_data = analyze_ticker(ticker, df_slice)
                        if signal_data:
                            results.append(signal_data)

                progress_bar.progress((i + 1) / len(tickers))

            status_text.empty()
            progress_bar.empty()

            date_label = analysis_date.strftime('%d/%m/%Y')
            st.markdown(f"### Risultati al **{date_label}**")

            if not results:
                st.info(f"Nessun risultato disponibile.")
                return

            results.sort(key=lambda x: x['confidence'], reverse=True)

            st.success(f"Analizzati {len(results)} titoli. Ordinati per Momentum.")

            summary_data = []
            for res in results:
                inds = res['indicators']
                summary_data.append({
                    "Ticker": res['ticker'],
                    "Segnale": res['signal'],
                    "Prezzo": res['price'],
                    "SMA 130": inds.get('SMA_130'),
                    "Momentum 6M": inds.get('Momentum_6M')
                })

            for res in results:
                inds = res['indicators']
                mom_val = inds.get('Momentum_6M', 0)
                sma_val = inds.get('SMA_130', 0)
                price = res['price']

                sig_type = res['raw_signal']
                if sig_type == "ENTRY":
                    box_color = "green"
                    icon = "🟢"
                elif sig_type == "EXIT":
                    box_color = "red"
                    icon = "🔴"
                else:
                    box_color = "orange"
                    icon = "🟠"

                with st.container():
                    st.markdown(f"#### {icon} **{res['ticker']}**: :{box_color}[{res['signal']}]")

                    c1, c2, c3 = st.columns(3)

                    dist_sma = (price / sma_val - 1) * 100 if sma_val else 0
                    c1.metric(
                        "Prezzo vs SMA 130",
                        f"{price:.2f}",
                        f"{dist_sma:+.1f}% (vs {sma_val:.2f})",
                        delta_color="normal" if price > sma_val else "inverse"
                    )

                    c2.metric(
                        "Momentum 6 Mesi",
                        f"{mom_val:.2%}",
                        "Fattore Ranking",
                        delta_color="normal" if mom_val > 0 else "inverse"
                    )

                    c3.metric("ATR (Volatilità)", f"{inds.get('ATR', 0):.2f}")

                    st.caption(f"**Analisi:** {res['reason']}")
                    st.divider()

    elif mode == "🤖 Automated Backtest":
        st.subheader("🤖 Simulazione Portafoglio Algoritmico")

        if st.button("▶️ Avvia Simulazione"):
            if not tickers:
                st.warning("Inserisci almeno un ticker nella sidebar.")
                return

            with st.spinner(f"Simulazione in corso dal {start_date_backtest}..."):
                df_trades, final_cap = run_market_aware_backtest(tickers, start_date=str(start_date_backtest))

            if not df_trades.empty:
                initial = 10000
                total_return = ((final_cap - initial) / initial) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Capitale Finale", f"€{final_cap:,.2f}")
                col2.metric("Rendimento Totale", f"{total_return:.2f}%",
                            delta_color="normal" if total_return > 0 else "inverse")

                realized_trades = df_trades[df_trades['Action'] == 'SELL']
                if not realized_trades.empty:
                    wins = len(realized_trades[realized_trades['PnL_Net'] > 0])
                    tot = len(realized_trades)
                    win_rate = (wins / tot) * 100
                    col3.metric("Win Rate", f"{win_rate:.1f}% ({wins}/{tot})")

                st.subheader("Giornale delle Operazioni")

                def style_backtest_rows(row):
                    act = row['Action']
                    if act == 'BUY':
                        return ['background-color: #202020'] * len(row)
                    if act == 'SELL':
                        if "TREND BREAK" in str(row['Reason']):
                            return ['background-color: #fff5f5; color: #c53030'] * len(row)
                        return ['background-color: #006600'] * len(row)
                    return [''] * len(row)

                df_display = df_trades.style.apply(style_backtest_rows, axis=1).format({
                    "Price": "{:.2f}",
                    "PnL_Net": "{:+.2f}",
                    "Capital": "€{:,.0f}",
                    "Qty": "{:.4f}"
                })

                st.dataframe(df_display, width="stretch")

                st.subheader("Curva del Capitale")
                equity_curve = df_trades.drop_duplicates(subset=['Date'], keep='last').set_index('Date')[
                    'Capital']
                st.line_chart(equity_curve)

            else:
                st.warning(
                    "Nessun trade generato nel periodo. Verifica che la data di inizio non sia troppo recente o che i dati siano disponibili.")