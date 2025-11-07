# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional, List
import pandas as pd

from etf_metrics.clients.yahoo_client import resolve_isin_one, get_series
from etf_metrics.clients.base_client import DataValidator

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

    df = pd.DataFrame(all_series).dropna()
    if df.shape[0] < 2:
        logger.error("Dati storici insufficienti per costruire il portafoglio dopo l'allineamento.")
        return None, {}

    return df, asset_to_ticker


def _rebalance_portfolio(prices_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Applica una strategia di ribilanciamento annuale.
    """
    capitals = pd.Series(0.0, index=prices_df.index)
    capitals.iloc[0] = 1.0

    last_rebalance_year = prices_df.index[0].year

    returns = prices_df.pct_change().fillna(0)

    current_weights = weights.copy()

    for i in range(1, len(prices_df)):
        date = prices_df.index[i]

        portfolio_return = (returns.iloc[i] * current_weights).sum()
        capitals.iloc[i] = capitals.iloc[i - 1] * (1 + portfolio_return)

        new_weights = current_weights * (1 + returns.iloc[i])
        current_weights = new_weights / new_weights.sum()

        if date.year > last_rebalance_year:
            current_weights = weights.copy()
            last_rebalance_year = date.year

    return capitals


def get_portfolio_series(portfolio_def: Dict[str, float], period: str = "max", rebalancing: str = 'mai') -> Optional[
    pd.Series]:
    """
    Costruisce la serie storica di un portafoglio (Lump Sum).
    Supporta il ribilanciamento annuale.
    """
    df, asset_to_ticker = _load_series_and_mapping(portfolio_def, period)
    if df is None:
        return None

    weights_dict = {asset_to_ticker[a]: portfolio_def[a] for a in asset_to_ticker.keys()}
    weights_series = pd.Series(weights_dict, index=df.columns, dtype=float).fillna(0.0)

    if weights_series.sum() <= 0:
        return None
    weights_series /= weights_series.sum()

    if rebalancing == 'annuale':
        portfolio_val = _rebalance_portfolio(df, weights_series)
    else:
        norm_df = df / df.iloc[0]
        portfolio_val = (norm_df * weights_series).sum(axis=1)

    portfolio_val.name = "Portfolio"
    return portfolio_val


def simulate_pac_investment(
        portfolio_def: Dict[str, float],
        monthly_investment: float,
        period: str = "max",
        start_date: Optional[pd.Timestamp] = None
) -> Optional[pd.Series]:
    """
    Simula un Piano di Accumulo (PAC).
    """
    df, asset_to_ticker = _load_series_and_mapping(portfolio_def, period)
    if df is None:
        return None

    weights_dict = {asset_to_ticker[a]: portfolio_def[a] for a in asset_to_ticker.keys()}
    weights_series = pd.Series(weights_dict, index=df.columns, dtype=float).fillna(0.0)
    if weights_series.sum() <= 0: return None
    weights_series /= weights_series.sum()

    shares = pd.DataFrame(0.0, index=df.index, columns=df.columns)

    simulation_start_date = df.index.min()
    if start_date is not None:
        simulation_start_date = max(simulation_start_date, pd.to_datetime(start_date))

    monthly_dates = pd.date_range(start=simulation_start_date, end=df.index.max(), freq="MS")

    purchase_indices = df.index.searchsorted(monthly_dates, side='left')

    valid_indices = [idx for idx in purchase_indices if idx < len(df.index)]
    actual_purchase_dates = df.index[valid_indices].unique()

    for purchase_date in actual_purchase_dates:
        prices_on_day = df.loc[purchase_date]
        alloc = monthly_investment * weights_series

        shares.loc[purchase_date] += alloc.div(prices_on_day).fillna(0)

    cumulative_shares = shares.cumsum()
    portfolio_values = (cumulative_shares * df).sum(axis=1)

    portfolio_values = portfolio_values[portfolio_values.index >= simulation_start_date]

    portfolio_values.name = "Portfolio PAC"
    return portfolio_values


def parse_portfolio_input(text_input: str) -> Optional[Dict[str, float]]:
    """
    Estrae la definizione del portafoglio da un input testuale.
    """
    portfolio: Dict[str, float] = {}
    lines = text_input.strip().split('\n')
    total_weight = 0.0

    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(':')]
        if len(parts) != 2:
            logger.error("Riga mal formattata: '%s'. Formato atteso 'ISIN: PESO'.", line)
            return None

        asset, weight_str = parts
        asset = asset.upper()

        try:
            weight = float(weight_str.replace('%', ''))
            if weight <= 0:
                raise ValueError
            portfolio[asset] = weight
            total_weight += weight
        except ValueError:
            logger.error("Peso non valido per %s: '%s'.", asset, weight_str)
            return None

    if not portfolio:
        return None

    return {k: v / total_weight for k, v in portfolio.items()}


def get_all_portfolios_for_backtest(
        user_portfolio_def: Dict,
        famous_portfolios_to_compare: List[str],
        config: Dict,
        initial_investment: float = 10000,
        strategy: str = "lump_sum",
        monthly_investment: float = 500,
        rebalancing: str = 'mai'
) -> Dict[str, pd.Series]:
    """
    Prepara le serie storiche per tutti i portafogli da confrontare,
    allineando la data di inizio a quella del portafoglio utente.
    """
    all_series_dict: Dict[str, pd.Series] = {}
    user_series = None
    start_date = None

    if strategy == "lump_sum_(pic)":
        user_series_norm = get_portfolio_series(user_portfolio_def, rebalancing=rebalancing)
        if user_series_norm is not None and not user_series_norm.empty:
            user_series = user_series_norm * initial_investment
            all_series_dict["Il Tuo Portafoglio"] = user_series
            start_date = user_series.index.min()
    else:
        user_series = simulate_pac_investment(user_portfolio_def, monthly_investment)
        if user_series is not None and not user_series.empty:
            all_series_dict["Il Tuo Portafoglio (PAC)"] = user_series
            start_date = user_series.index.min()

    for name in famous_portfolios_to_compare:
        if name in config:
            famous_series_full = None
            series_to_add = None
            series_name = name

            if strategy == "lump_sum_(pic)":
                famous_series_full = get_portfolio_series(config[name], rebalancing=rebalancing)

                if famous_series_full is not None and not famous_series_full.empty:
                    famous_series_aligned = famous_series_full
                    if start_date:
                        famous_series_aligned = famous_series_full[famous_series_full.index >= start_date]

                    if not famous_series_aligned.empty:
                        renormalized_series = (famous_series_aligned / famous_series_aligned.iloc[0])
                        series_to_add = renormalized_series * initial_investment

            else:
                series_name = f"{name} (PAC)"
                famous_series_full = simulate_pac_investment(
                    config[name],
                    monthly_investment,
                    start_date=start_date
                )

                if famous_series_full is not None and not famous_series_full.empty:
                    series_to_add = famous_series_full

            if series_to_add is not None and not series_to_add.empty:
                all_series_dict[series_name] = series_to_add

    return all_series_dict