# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
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


def _calculate_detailed_metrics(df_trades: pd.DataFrame, initial_capital: float):
    """
    Calcola metriche avanzate usando la Total_Equity (NAV) invece del Cash.
    """
    if df_trades.empty:
        return None

    df_trades['Date'] = pd.to_datetime(df_trades['Date'])
    start_date = df_trades['Date'].min()
    end_date = df_trades['Date'].max()

    if start_date == end_date:
        return None

    all_dates = pd.date_range(start_date, end_date, freq='B')
    equity_df = pd.DataFrame(index=all_dates)

    # SELEZIONE COLONNA CORRETTA PER IL CALCOLO
    # Se esiste 'Total_Equity' (nuovo codice), usa quella. Altrimenti fallback su 'Capital'.
    val_col = 'Total_Equity' if 'Total_Equity' in df_trades.columns else 'Capital'

    equity_df['Value'] = np.nan
    equity_df.iloc[0, 0] = initial_capital  # Start value

    # Aggiorniamo il valore nei giorni in cui ci sono stati trade
    daily_val_update = df_trades.groupby('Date')[val_col].last()
    equity_df.loc[daily_val_update.index, 'Value'] = daily_val_update

    # Forward Fill: nei giorni senza trade, il valore rimane costante
    # (Nota: è una approssimazione, idealmente si ricalcolerebbe ogni giorno mark-to-market,
    # ma il backtester registra solo i trade. Comunque evita il crollo a zero del cash).
    equity_df['Value'] = equity_df['Value'].ffill()

    # 3. Calcolo Ritorni Giornalieri
    equity_df['Daily_Ret'] = equity_df['Value'].pct_change().fillna(0)

    # --- CALCOLO KPI ---
    final_cap = equity_df['Value'].iloc[-1]

    days = (end_date - start_date).days
    years = days / 365.25
    cagr = ((final_cap / initial_capital) ** (1 / years)) - 1 if years > 0 else 0

    volatility = equity_df['Daily_Ret'].std() * np.sqrt(252)

    risk_free_rate = 0.03
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0

    negative_returns = equity_df[equity_df['Daily_Ret'] < 0]['Daily_Ret']
    downside_dev = negative_returns.std() * np.sqrt(252)
    sortino = (cagr - risk_free_rate) / downside_dev if downside_dev > 0 else 0

    cumulative_returns = (1 + equity_df['Daily_Ret']).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    sells = df_trades[df_trades['Action'] == 'SELL']
    gross_profit = sells[sells['PnL_Net'] > 0]['PnL_Net'].sum()
    gross_loss = abs(sells[sells['PnL_Net'] < 0]['PnL_Net'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": max_drawdown,
        "Profit_Factor": profit_factor,
        "Equity_Curve": equity_df['Value']
    }


def render_trading_ui():
    st.title("💹 Trading & Backtest")

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

        **In Sintesi:** Compriamo i titoli più forti, ma solo se sono in un trend strutturalmente sano. Usciamo immediatamente se il trend si rompe.
        """)

    st.sidebar.header("⚙️ Configurazione")

    mode = st.sidebar.radio(
        "Modalità Operativa",
        ["📡 Scanner Segnali", "🤖 Automated Backtest"],
        help="Scanner live o simulazione storica della strategia."
    )

    st.sidebar.divider()

    initial_capital = st.sidebar.number_input(
        "💰 Capitale Iniziale (€)",
        min_value=100.0,
        value=10000.0,
        step=500.0,
        help="Capitale di partenza per la simulazione."
    )

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

    # --- LOGICA DISPLAY ---

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
                    st.divider()

    elif mode == "🤖 Automated Backtest":
        st.subheader("🤖 Simulazione Portafoglio Algoritmico")
        st.caption("Validazione Strategia: Rischio, Rendimento e Drawdown.")

        if st.button("▶️ Avvia Simulazione Completa"):
            if not tickers:
                st.warning("Inserisci almeno un ticker nella sidebar.")
                return

            with st.spinner(f"Simulazione dal {start_date_backtest} con capitale €{initial_capital:,.2f}..."):
                # Esecuzione Core Backtest
                df_trades, final_cap = run_market_aware_backtest(
                    tickers,
                    start_date=str(start_date_backtest),
                    initial_capital=initial_capital
                )

            if not df_trades.empty:
                # Calcolo Metriche Avanzate (Sharpe, MDD, etc.)
                metrics = _calculate_detailed_metrics(df_trades, initial_capital)

                total_return_pct = ((final_cap - initial_capital) / initial_capital) * 100

                # --- SEZIONE 1: KPI PRINCIPALI ---
                st.markdown("### 📊 Performance Report")

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Capitale Finale", f"€{final_cap:,.0f}")
                kpi2.metric("Rendimento Totale", f"{total_return_pct:+.2f}%",
                            delta_color="normal" if total_return_pct > 0 else "inverse")

                if metrics:
                    kpi3.metric("CAGR (Annuo)", f"{metrics['CAGR'] * 100:.2f}%")
                    kpi4.metric("Max Drawdown", f"{metrics['Max_Drawdown'] * 100:.2f}%",
                                help="Massima perdita dal picco. Più è basso (vicino a 0), meglio è.",
                                delta_color="inverse")

                st.divider()

                # --- SEZIONE 2: RISCHIO E STATISTICHE (Punti 1, 2, 5) ---
                st.markdown("### 🛡️ Analisi Rischio & Statistiche (Validation Checklist)")

                if metrics:
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

                    sharpe = metrics['Sharpe']
                    col_r1.metric("Sharpe Ratio", f"{sharpe:.2f}",
                                  help="> 1.0 Buono, > 2.0 Ottimo. Misura il rendimento per unità di rischio.")

                    col_r2.metric("Sortino Ratio", f"{metrics['Sortino']:.2f}",
                                  help="Simile allo Sharpe, ma considera solo la volatilità negativa (i crolli).")

                    pf = metrics['Profit_Factor']
                    col_r3.metric("Profit Factor", f"{pf:.2f}",
                                  help="Rapporto tra vincite lorde e perdite lorde. > 1.5 è solido.")

                    col_r4.metric("Volatilità", f"{metrics['Volatility'] * 100:.1f}%")

                # Statistiche Trade (Punto 5)
                realized = df_trades[df_trades['Action'] == 'SELL']
                if not realized.empty:
                    wins = len(realized[realized['PnL_Net'] > 0])
                    losses = len(realized[realized['PnL_Net'] <= 0])
                    total = len(realized)
                    win_rate = (wins / total) * 100

                    st.markdown(f"""
                    **Statistiche Operative (Significatività):**
                    - Totale Operazioni: **{total}**
                    - ✅ Vincite: **{wins}**
                    - ❌ Perdite: **{losses}**
                    - 🎯 Win Rate: **{win_rate:.1f}%**
                    """)

                    if total < 30:
                        st.warning(
                            "⚠️ Attenzione: Numero di operazioni basso (<30). I risultati statistici potrebbero non essere affidabili.")

                st.divider()

                # --- SEZIONE 3: GRAFICI ---
                tab_chart2, tab_chart1 = st.tabs(["📋 Lista Operazioni", "📈 Curva Capitale"])

                with tab_chart1:
                    if metrics and metrics.get('Equity_Curve') is not None:
                        st.line_chart(metrics['Equity_Curve'])
                    else:
                        simple_curve = df_trades.drop_duplicates(subset=['Date'], keep='last').set_index('Date')[
                            'Capital']
                        st.line_chart(simple_curve)

                with tab_chart2:
                    def style_backtest_rows(row):
                        act = row['Action']
                        if act == 'BUY': return ['background-color: #202020'] * len(row)
                        if act == 'SELL':
                            if "TREND BREAK" in str(row['Reason']): return ['background-color: #3d0000'] * len(row)
                            if row['PnL_Net'] > 0: return ['background-color: #003300'] * len(row)
                            return ['background-color: #3d0000'] * len(row)
                        return [''] * len(row)

                    df_display = df_trades.style.apply(style_backtest_rows, axis=1).format({
                        "Price": "{:.2f}", "PnL_Net": "{:+.2f}", "Capital": "€{:,.0f}", "Total_Equity": "€{:,.0f}",
                        "Qty": "{:.4f}"
                    })
                    st.dataframe(df_display, width="stretch")

            else:
                st.warning(
                    "Nessun trade generato nel periodo. Prova ad ampliare l'universo ticker o cambiare la data di inizio.")