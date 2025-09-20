# -*- coding: utf-8 -*-
from typing import List
import pandas as pd
import streamlit as st

# Importa entrambe le nuove funzioni
from .momentum import fetch_momentum_data, process_momentum_rankings


def render_momentum_ui():
    st.sidebar.header("Impostazioni Momentum")
    momentum_isins = st.sidebar.text_area(
        "ISIN da confrontare (uno per riga)",
        "XEON\nIE00B3VTMJ91\nLU1650487413\nLU1287023185\nIE00BH04GL39\nIE00B14X4S71\n"
        "LU1407887329\nIE00BDFK1573\nIE00B1FZS798\nLU1407890976\nLU0378818131\nIE00BDBRDM35\n"
        "XGIN\nIE00B0M62X26\nIE00B1FZSC47\nIE00B66F4759\nIE00B4PY7Y77\nIE00BF8HV600\nIE00B2NPKV68\n"
        "IE00B9M6RS56\nIE00B4L5Y983\nIWDE\nIE00BK5BQT80\nIE00B5BMR087\nIE00BFMXXD54\nIE00B3ZW0K18\n"
        "LU1079841273\nIE00BSPLC413\nIE00B53SZB19\nLU1829221024\nNDXH\nIE00B6YX5D40\nCSSX5E\n"
        "LU0252633754\nCSMIB\nFR0010010827\nXXSC\nLU0779800910\nIE00BDDRF700\nIE00BL25JP72\n"
        "IE000M7V94E1\nIE00BYPLS672\nIE0007Y8Y157\nIE00BYZK4552\nIE00BJGWQN72\nIE00BYPLS672\n"
        "IE00BDVPNG13\nIE00BM67HT60\nLU2023678282\nIE00BYWQWR46\nIE00BYZK4669\nIE00BYZK4776\n"
        "IE00BQ70R696\nIE00BYZK4883\nIE000U58J0M1\nIE00BF0M6N54\nIE00BFYN8Y92\nIE00BKTLJC87\n"
        "IE00B1FZS467\nIE00BGL86Z12\nIE00BGBN6P67\nIE00BLRPQH31\nIE00BF0M2Z96\nIE00B1FZSF77\n"
        "IE00B579F325\nDE000A1EK0G3\nCH0454664001\nBTCE\nVNGA20\nVNGA40\nVNGA60\nVNGA80\n"
        "NL0009272764\nNL0009272772\nNL0009272780\nXQUI\nIE000YYE6WK5\nIE00BK5BC891",
        height=120,
    )

    # Il pulsante ora serve solo per caricare i dati la prima volta
    if st.sidebar.button("Carica Dati ETF"):
        isins_list: List[str] = [isin for isin in momentum_isins.split('\n') if isin.strip()]
        if not isins_list:
            st.warning("Inserisci almeno un ISIN.")
            st.session_state.momentum_raw_data = None
        else:
            # Salva i dati grezzi nella sessione
            st.session_state.momentum_raw_data = fetch_momentum_data(isins_list)

    st.title("Verifica Momentum Corretto per il Rischio")

    # Controlli che ora funzionano dinamicamente
    if 'momentum_raw_data' in st.session_state and st.session_state.momentum_raw_data:
        st.sidebar.header("Filtri di Analisi")
        lookback = st.sidebar.slider("Periodo di lookback (mesi)", 1, 24, 6, 1)
        rf_ann_momentum = st.sidebar.number_input(
            "Risk-free annuo (%) per Sharpe", -5.0, 10.0, 3.9, 0.25, key="rf_momentum"
        )

        st.caption(f"Classifica per Sharpe Ratio negli ultimi {lookback} mesi, con segnale operativo attuale.")

        # Elabora i dati grezzi con il lookback corrente
        results = process_momentum_rankings(st.session_state.momentum_raw_data, lookback, rf_ann_momentum)

        if results:
            best_performer = results[0]
            st.success(
                f"🥇 **Miglior Sharpe Ratio ({lookback} mesi):** **{best_performer['ticker']}** "
                f"({best_performer['isin']}) con Sharpe di **{best_performer['sharpe_ratio']:.2f}**."
            )
            st.subheader("Classifica Completa")
            df = pd.DataFrame(results).rename(columns={
                "isin": "ISIN", "ticker": "Ticker", "cagr_pct": "CAGR %",
                "mdd_pct": "Max DD %", "sharpe_ratio": "Sharpe Ratio", "signal": "Segnale Operativo"
            })
            df = df[["ISIN", "Ticker", "Sharpe Ratio", "Segnale Operativo", "CAGR %", "Max DD %"]]

            formatters = {"CAGR %": "{:.2f}%", "Max DD %": "{:.2f}%", "Sharpe Ratio": "{:.2f}"}

            def style_signal(val):
                color_map = {"Compra": "#28a745", "Vendi": "#dc3545", "Mantieni": "#ffc107"}
                for signal, color in color_map.items():
                    if signal in str(val):
                        return f'background-color: {color}; color: {"white" if signal != "Mantieni" else "black"}; font-weight: bold;'
                return ''

            st.dataframe(
                df.style.format(formatters, na_rep="n.d.").applymap(style_signal, subset=["Segnale Operativo"]),
                width="stretch"
            )
        else:
            st.warning(f"Nessun ETF con dati sufficienti per il lookback di {lookback} mesi.")
    else:
        st.info("Inserisci gli ISIN e premi 'Carica Dati ETF' per iniziare.")