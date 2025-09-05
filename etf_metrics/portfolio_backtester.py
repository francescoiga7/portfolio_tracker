# -*- coding: utf-8 -*-
import re
from typing import Dict, Optional, List
import pandas as pd
import streamlit as st
from .yahoo_client import resolve_isin_one, get_series


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_portfolio_series(portfolio_def: Dict[str, float], period: str = "max") -> Optional[pd.Series]:
    """
    Costruisce la serie storica di un portafoglio basato sui suoi componenti e pesi.
    Risolve gli ISIN se necessario.
    """
    all_series = {}

    for asset, weight in portfolio_def.items():
        ticker = asset
        # Se l'asset è un ISIN, prova a risolverlo
        if re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', asset.upper()):
            resolved_ticker = resolve_isin_one(asset.upper())
            if not resolved_ticker:
                st.warning(f"Impossibile risolvere l'ISIN {asset}, verrà saltato.")
                continue
            ticker = resolved_ticker

        series = get_series(ticker, period)
        if series is not None and not series.empty:
            all_series[ticker] = series
        else:
            st.warning(f"Nessun dato storico trovato per {ticker} ({asset}), verrà saltato.")

    if not all_series:
        return None

    # Allinea tutte le serie su un indice di date comune
    df = pd.DataFrame(all_series).dropna()
    if df.shape[0] < 2:
        st.error("Dati storici insufficienti per costruire il portafoglio dopo l'allineamento.")
        return None

    # Normalizza le serie (rebase a 1) e applica i pesi
    norm_df = df / df.iloc[0]

    # Assicurati che i pesi corrispondano alle serie effettivamente scaricate
    weights_series = pd.Series(
        {ticker: portfolio_def[asset] for asset, ticker in zip(portfolio_def.keys(), all_series.keys())})
    weights_series /= weights_series.sum()  # Normalizza i pesi nel caso qualche asset sia stato saltato

    portfolio_val = (norm_df * weights_series).sum(axis=1)
    portfolio_val.name = "Portfolio"

    return portfolio_val


def simulate_pac_investment(portfolio_def: Dict[str, float], monthly_investment: float, period: str = "max") -> \
Optional[pd.Series]:
    """
    Simula un Piano di Accumulo Capitale (PAC) per un portafoglio.
    """
    all_series = {}
    asset_to_ticker = {}  # Mappatura da asset originale a ticker risolto

    for asset, weight in portfolio_def.items():
        original_asset = asset
        ticker = asset

        # Se l'asset è un ISIN, prova a risolverlo
        if re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', asset.upper()):
            resolved_ticker = resolve_isin_one(asset.upper())
            if not resolved_ticker:
                st.warning(f"Impossibile risolvere l'ISIN {asset}, verrà saltato.")
                continue
            ticker = resolved_ticker
            asset_to_ticker[original_asset] = ticker

        series = get_series(ticker, period)
        if series is not None and not series.empty:
            all_series[original_asset] = series  # Usa l'asset originale come chiave
        else:
            st.warning(f"Nessun dato storico trovato per {ticker} ({asset}), verrà saltato.")

    if not all_series:
        return None

    # Allinea tutte le serie su un indice di date comune
    df = pd.DataFrame(all_series).dropna()
    if df.shape[0] < 2:
        st.error("Dati storici insufficienti per costruire il portafoglio dopo l'allineamento.")
        return None

    # Calcola l'investimento mensile per ogni asset
    monthly_investments = {}
    start_date = df.index[0]
    end_date = df.index[-1]

    # Genera tutte le date mensili nel periodo
    all_dates = pd.date_range(start=start_date, end=end_date, freq='MS')

    for asset, series in all_series.items():
        # Calcola l'investimento per questo asset
        asset_investment = monthly_investment * portfolio_def[asset]

        # Crea una serie con gli acquisti mensili
        purchases = pd.Series(0.0, index=df.index)
        for date in all_dates:
            if date in df.index:
                # Trova il prezzo alla data di acquisto
                price = series.loc[date]
                # Calcola quante quote acquistare
                shares = asset_investment / price
                purchases.loc[date] += shares

        # Calcola il numero cumulativo di quote
        cumulative_shares = purchases.cumsum()

        # Calcola il valore del investimento in questo asset
        monthly_investments[asset] = cumulative_shares * series

    # Combina tutti gli asset
    portfolio_val = pd.DataFrame(monthly_investments).sum(axis=1)
    portfolio_val.name = "Portfolio PAC"

    return portfolio_val

def parse_portfolio_input(text_input: str) -> Optional[Dict[str, float]]:
    """
    Estrae la definizione del portafoglio da un input testuale.
    Formato atteso: ISIN:PESO% (es. IE00BK5BQT80: 80)
    """
    portfolio = {}
    lines = text_input.strip().split('\n')
    total_weight = 0

    for line in lines:
        if not line.strip():
            continue

        parts = [p.strip() for p in line.split(':')]
        if len(parts) != 2:
            st.error(f"Riga mal formattata: '{line}'. Usare il formato 'ISIN: PESO'.")
            return None

        isin, weight_str = parts
        isin = isin.upper()

        if not re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', isin):
            st.error(f"ISIN non valido: '{isin}'.")
            return None

        try:
            weight = float(weight_str.replace('%', ''))
            if weight <= 0:
                raise ValueError
            portfolio[isin] = weight
            total_weight += weight
        except ValueError:
            st.error(f"Peso non valido per {isin}: '{weight_str}'. Inserire un numero positivo.")
            return None

    if not portfolio:
        st.info("Inserisci almeno un ISIN con un peso per definire il portafoglio.")
        return None

    if abs(total_weight - 100.0) > 0.01:
        st.warning(f"La somma dei pesi è {total_weight:.2f}%, non 100%. I pesi verranno normalizzati.")

    # Normalizza i pesi a 1
    return {k: v / total_weight for k, v in portfolio.items()}


def get_all_portfolios_for_backtest(user_portfolio_def: Dict, famous_portfolios_to_compare: List[str], config: Dict,
                                    strategy: str = "lump_sum", monthly_investment: float = 1000) -> Dict[
    str, pd.Series]:
    """Prepara le serie storiche per tutti i portafogli da confrontare."""

    all_series_dict = {}

    # 1. Calcola il portafoglio dell'utente
    if strategy == "lump_sum":
        user_series = get_portfolio_series(user_portfolio_def)
    else:  # PAC
        user_series = simulate_pac_investment(user_portfolio_def, monthly_investment)

    if user_series is not None:
        all_series_dict["Il Tuo Portafoglio"] = user_series

    # 2. Calcola i portafogli famosi selezionati (sempre lump-sum per i modelli)
    for name in famous_portfolios_to_compare:
        if name in config:
            famous_series = get_portfolio_series(config[name])
            if famous_series is not None:
                all_series_dict[name] = famous_series

    return all_series_dict