# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional, List

import pandas as pd
import streamlit as st

from .yahoo_client import resolve_isin_one, get_series
from .base_client import DataValidator

logger = logging.getLogger(__name__)


def _load_series_and_mapping(portfolio_def: Dict[str, float], period: str):
    """
    Carica le serie per gli asset (ticker o ISIN) e ritorna:
    - df allineato (indice comune) con colonne=ticker
    - mappa asset_originale -> ticker
    """
    all_series: Dict[str, pd.Series] = {}
    asset_to_ticker: Dict[str, str] = {}

    for asset, _ in portfolio_def.items():
        ticker = resolve_isin_one(asset.upper()) if DataValidator.validate_isin(asset) else asset
        if not ticker:
            logger.warning("Impossibile risolvere ISIN %s; asset saltato.", asset)
            continue
        s = get_series(ticker, period)
        if s is None or s.empty:
            logger.warning("Nessun dato storico per %s (%s); asset saltato.", ticker, asset)
            continue
        all_series[ticker] = s
        asset_to_ticker[asset] = ticker

    if not all_series:
        return None, {}

    # Allinea tutte le serie su un indice di date comune
    df = pd.DataFrame(all_series).dropna()
    if df.shape[0] < 2:
        logger.error("Dati storici insufficienti per costruire il portafoglio dopo l'allineamento.")
        return None, {}

    return df, asset_to_ticker
@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_portfolio_series(portfolio_def: Dict[str, float], period: str = "max") -> Optional[pd.Series]:
    """
    Costruisce la serie storica di un portafoglio basato sui suoi componenti e pesi.
    Risolve gli ISIN se necessario. Pesi correttamente allineati agli asset scaricati.
    """
    df, asset_to_ticker = _load_series_and_mapping(portfolio_def, period)
    if df is None:
        return None

    # Normalizza le serie (rebase a 1)
    norm_df = df / df.iloc[0]

    # Costruisci i pesi SOLO per gli asset effettivamente inclusi
    weights_dict = {asset_to_ticker[a]: portfolio_def[a] for a in asset_to_ticker.keys()}
    weights_series = pd.Series(weights_dict, index=norm_df.columns, dtype=float).fillna(0.0)

    # Normalizza i pesi (nel caso qualche asset sia stato escluso)
    if weights_series.sum() > 0:
        weights_series = weights_series / weights_series.sum()
    else:
        return None

    portfolio_val = (norm_df * weights_series).sum(axis=1)
    portfolio_val.name = "Portfolio"
    return portfolio_val


def simulate_pac_investment(
    portfolio_def: Dict[str, float],
    monthly_investment: float,
    period: str = "max",
) -> Optional[pd.Series]:
    """
    Simula un Piano di Accumulo (PAC).
    Gli acquisti avvengono a inizio mese (MS) o al primo giorno di negoziazione successivo (bfill).
    """
    df, asset_to_ticker = _load_series_and_mapping(portfolio_def, period)
    if df is None:
        return None

    monthly_investments: Dict[str, pd.Series] = {}
    start_date = df.index[0]
    end_date = df.index[-1]
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    for asset, ticker in asset_to_ticker.items():
        series = df[ticker]  # già allineata all'indice comune
        asset_investment = monthly_investment * float(portfolio_def[asset])

        # Prezzi di acquisto: data MS o primo giorno utile successivo
        purchase_prices = series.reindex(monthly_dates, method="bfill").dropna()
        if purchase_prices.empty:
            logger.warning("Nessun prezzo mensile disponibile per %s; asset saltato.", ticker)
            continue

        # Quote acquistate ogni mese
        purchases_shares = asset_investment / purchase_prices

        # Serie degli acquisti mappata sull'intero indice di df (tutto il periodo)
        purchases = pd.Series(0.0, index=df.index)
        purchases.loc[purchase_prices.index] = purchases_shares.values

        # Quote cumulate e valore dell'investimento per asset
        cumulative_shares = purchases.cumsum()
        monthly_investments[asset] = cumulative_shares * series

    if not monthly_investments:
        return None

    portfolio_val = pd.DataFrame(monthly_investments).sum(axis=1)
    portfolio_val.name = "Portfolio PAC"
    return portfolio_val


def parse_portfolio_input(text_input: str) -> Optional[Dict[str, float]]:
    """
    Estrae la definizione del portafoglio da un input testuale.
    Formato: ISIN:PESO% (es. IE00BK5BQT80: 80)
    """
    portfolio: Dict[str, float] = {}
    lines = text_input.strip().split('\n')
    total_weight = 0.0

    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(':')]
        if len(parts) != 2:
            st.error(f"Riga mal formattata: '{line}'. Usare il formato 'ISIN: PESO'.")
            return None

        isin, weight_str = parts
        isin = isin.upper()
        if not DataValidator.validate_isin(isin):
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


def get_all_portfolios_for_backtest(
    user_portfolio_def: Dict,
    famous_portfolios_to_compare: List[str],
    config: Dict,
    strategy: str = "lump_sum",
    monthly_investment: float = 1000,
) -> Dict[str, pd.Series]:
    """Prepara le serie storiche per tutti i portafogli da confrontare."""
    all_series_dict: Dict[str, pd.Series] = {}

    # 1) Portafoglio utente
    if strategy == "lump_sum":
        user_series = get_portfolio_series(user_portfolio_def)
    else:  # PAC
        user_series = simulate_pac_investment(user_portfolio_def, monthly_investment)

    if user_series is not None:
        all_series_dict["Il Tuo Portafoglio"] = user_series

    # 2) Portafogli “famosi” (sempre lump-sum)
    for name in famous_portfolios_to_compare:
        if name in config:
            famous_series = get_portfolio_series(config[name])
            if famous_series is not None:
                all_series_dict[name] = famous_series

    return all_series_dict