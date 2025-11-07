# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional, Iterable, Tuple
import pandas as pd
import streamlit as st
from datetime import date
import numpy as np

from etf_metrics.clients.yahoo_client import get_series, get_info, resolve_isin_one
from .etf_search_engine import discover_universe, get_unique_preferred_tickers
from etf_metrics.shared.config import DEFAULT_SEED_QUERIES
from .metrics import compute_metrics_from_series

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False, ttl=60 * 15)
def get_market_regime(as_of_date: Optional[date] = None) -> Dict:
    """
    Controlla sia la volatilità (VIX) sia il trend del mercato generale (S&P 500).
    (Funzione invariata)
    """
    try:
        end_date = pd.to_datetime(as_of_date) if as_of_date else pd.to_datetime(date.today())

        # 1. Controllo VIX
        vix_series = get_series("^VIX", period="1y")
        current_vix = None
        regime = "Sconosciuto"

        if vix_series is not None and not vix_series.empty:
            series_vix_as_of = vix_series[vix_series.index <= end_date]
            if not series_vix_as_of.empty:
                current_vix = series_vix_as_of.iloc[-1]
                regime = "Avverso al Rischio" if current_vix > 20 else "Favorevole al Rischio"
            else:
                regime = "Dati VIX non disponibili per la data"

        # 2. Controllo Trend S&P 500
        gspc_series = get_series("^GSPC", period="2y")
        market_trend = "Sconosciuto"

        if gspc_series is not None and not gspc_series.empty:
            series_gspc_as_of = gspc_series[gspc_series.index <= end_date]
            if len(series_gspc_as_of) > 200:
                sma200_gspc = series_gspc_as_of.rolling(window=200).mean().iloc[-1]
                current_price_gspc = series_gspc_as_of.iloc[-1]
                if pd.notna(sma200_gspc) and pd.notna(current_price_gspc):
                    market_trend = "Rialzista" if current_price_gspc > sma200_gspc else "Ribassista"
            else:
                market_trend = "Dati S&P 500 insuff."

        return {"vix": current_vix, "regime": regime, "market_trend": market_trend}

    except Exception as e:
        logger.error(f"Errore in get_market_regime: {e}")
        return {"vix": None, "regime": "Sconosciuto", "market_trend": "Sconosciuto"}


@st.cache_data(show_spinner="Caricamento dati storici dell'universo...", ttl=60 * 30)
def fetch_screener_data(
        log_area: List[str],
        specific_isins: Optional[List[str]] = None,
        instrument_types: List[str] = ["ETF", "ETP", "ETN"],
        queries: Iterable[str] = DEFAULT_SEED_QUERIES,
        quotes_per_query: int = 200
) -> List[Dict]:
    """
    Fase 1 (Lenta): Scopre o riceve un universo di strumenti e scarica la loro intera serie storica.
    (Funzione invariata)
    """
    unique_tickers = []
    if specific_isins:
        log_area.append(f"**1. Modalità Manuale: {len(specific_isins)} strumenti forniti.**")
        with st.spinner("Risoluzione ISIN/Ticker..."):
            resolved_tickers = []
            for item in specific_isins:
                if len(item) == 12:
                    ticker = resolve_isin_one(item)
                    if ticker:
                        resolved_tickers.append(ticker)
                else:
                    resolved_tickers.append(item)
            unique_tickers = list(filter(None, resolved_tickers))
        log_area.append(f"- Trovati {len(unique_tickers)} ticker validi.")
    else:
        log_area.append(f"**1. Discovery Automatica per tipi: {', '.join(instrument_types)}...**")
        raw_tickers = discover_universe(queries, quotes_per_query=quotes_per_query, instrument_types=instrument_types)
        log_area.append(f"- Trovati {len(raw_tickers)} ticker grezzi.")

        unique_tickers = get_unique_preferred_tickers(raw_tickers)
        log_area.append(f"- **Universo finale da analizzare: {len(unique_tickers)} ticker unici.**")

    fetched_data = []
    progress_bar = st.progress(0, text=f"Download dati per {len(unique_tickers)} strumenti...")
    for i, ticker in enumerate(unique_tickers):
        series_full = get_series(ticker, period="5y", as_dataframe=True)
        if series_full is not None and not series_full.empty:
            fetched_data.append({"ticker": ticker, "series": series_full})
        progress_bar.progress((i + 1) / len(unique_tickers), text=f"Download: {ticker}")
    progress_bar.empty()
    log_area.append(f"- **Download completato per {len(fetched_data)} strumenti con dati storici sufficienti.**")
    return fetched_data


import warnings


def _calculate_metrics_from_series(ticker: str, series_full: pd.DataFrame, as_of_date: date, log_entry: List[str]) -> \
        Optional[Dict]:
    """
    Calcola le metriche per un singolo ETF a partire dalla sua serie storica completa.
    (Funzione invariata)
    """
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            if 'Close' not in series_full.columns:
                log_entry.append("ERRORE: La colonna 'Close' non è presente nei dati.")
                return None

            close_prices_full = series_full['Close']

            end_date = pd.to_datetime(as_of_date)
            series = close_prices_full[close_prices_full.index <= end_date]

            if series.empty or len(series) < 252:
                log_entry.append("Dati storici insufficienti (< 252 giorni) alla data selezionata.")
                return None

            info = get_info(ticker=ticker) or {}
            avg_volume_3m = info.get("averageDailyVolume3Month")
            avg_value_eur = (avg_volume_3m * series.iloc[-1]) if avg_volume_3m is not None else 0

            sma200 = series.rolling(window=200).mean().iloc[-1]
            current_price = series.iloc[-1]
            is_above_ma200 = (current_price > sma200) if pd.notna(sma200) and pd.notna(current_price) else False

            series_12m = series[series.index >= (end_date - pd.DateOffset(months=12))]
            if len(series_12m) < 250:
                log_entry.append("Storico inferiore a 12 mesi.")
                return None

            series_6m = series_12m[series_12m.index >= (end_date - pd.DateOffset(months=6))]
            series_3m = series_6m[series_6m.index >= (end_date - pd.DateOffset(months=3))]

            metrics_12m = compute_metrics_from_series(series_12m)
            cagr, mdd = metrics_12m.get("cagr"), metrics_12m.get("mdd")
            calmar_ratio = -cagr / mdd if cagr is not None and mdd is not None and mdd != 0 else None

            roc_12m = (series_12m.iloc[-1] / series_12m.iloc[0] - 1)
            roc_6m = (series_6m.iloc[-1] / series_6m.iloc[0] - 1)
            roc_3m = (series_3m.iloc[-1] / series_3m.iloc[0] - 1)

            high_52w = series_12m.max()
            proximity_to_high = series.iloc[-1] / high_52w if high_52w > 0 else None

            sma20 = series.rolling(window=20).mean()
            std20 = series.rolling(window=20).std()
            bollinger_width = (4 * std20) / sma20
            min_bw_6m = bollinger_width[bollinger_width.index >= (end_date - pd.DateOffset(months=6))].min()
            volatility_compression = bollinger_width.iloc[-1] / min_bw_6m if min_bw_6m > 0 else None

            returns_6m = series_6m.pct_change().dropna()
            volatility_6m = returns_6m.std(ddof=1) * np.sqrt(252) if len(returns_6m) >= 2 else None

            if caught_warnings:
                for warn in caught_warnings:
                    log_entry.append(f"AVVISO: {warn.message}")
                    print(f"DEBUG - Avviso per il ticker {ticker}: {warn.message}")

            log_entry.append("OK")
            return {
                "ticker": ticker, "isin": get_info(ticker=ticker).get("isin"), "name": info.get("longName", ticker),
                "avg_value_eur": avg_value_eur,
                "is_above_ma200": is_above_ma200,
                "proximity_to_high": proximity_to_high,
                "calmar_ratio": calmar_ratio, "roc_12m": roc_12m,
                "roc_6m": roc_6m, "roc_3m": roc_3m,
                "volatility_compression": volatility_compression, "volatility_6m": volatility_6m,
            }

    except Exception as e:
        log_entry.append(f"ERRORE per il ticker **{ticker}**: {e}")
        print(f"DEBUG - Errore bloccante per il ticker {ticker}: {e}")
        return None

@st.cache_data(show_spinner="Calcolo metriche per la data selezionata...", ttl=60 * 15)
def calculate_all_metrics(fetched_data: List[Dict], as_of_date: date) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fase 2 (Lenta, ma Messa in Cache): Calcola le metriche per tutti gli ETF.
    Questa funzione viene eseguita solo una volta per ogni data.
    """
    log_area = []
    log_area.append(f"\n**3. Esecuzione Calcolo Metriche (in cache) alla data {as_of_date.strftime('%d/%m/%Y')}...**")

    all_metrics = []

    processing_log_details = []

    for data in fetched_data:
        log_entry = [data['ticker']]
        metrics = _calculate_metrics_from_series(data['ticker'], data['series'], as_of_date, log_entry)
        if metrics:
            all_metrics.append(metrics)
        processing_log_details.append(log_entry)

    if not all_metrics:
        log_area.append("- Nessuna metrica calcolabile trovata.")
        return pd.DataFrame(), log_area

    df = pd.DataFrame(all_metrics)
    log_area.append(f"- {len(df)} ETF con metriche calcolabili (dati grezzi).")

    essential_cols = ['calmar_ratio', 'roc_12m', 'roc_6m', 'roc_3m', 'proximity_to_high', 'volatility_compression',
                      'volatility_6m', 'is_above_ma200']
    df = df.dropna(subset=essential_cols)
    log_area.append(f"- {len(df)} ETF rimasti dopo pulizia dati incompleti.")

    df['avg_value_eur'] = df['avg_value_eur'].fillna(0)

    st.session_state.debug_log_processing = processing_log_details

    return df, log_area


def filter_and_rank_metrics(metrics_df: pd.DataFrame, min_avg_value: float,
                            min_proximity_to_high: float, log_area: List[str]
                            ) -> pd.DataFrame:
    """
    Fase 3 (Veloce): Filtra e ordina il DataFrame pre-calcolato in base ai controlli della UI.
    (Funzione invariata)
    """
    log_area.append(f"\n**4. Filtraggio e Ranking (Veloce)...**")

    if metrics_df.empty:
        log_area.append("- DataFrame metriche vuoto, nessun ranking possibile.")
        return pd.DataFrame()

    df = metrics_df.copy()

    df_pre_liq = len(df)
    df = df[df['avg_value_eur'] >= min_avg_value]
    log_area.append(
        f"- Filtro Liquidità: Rimossi {df_pre_liq - len(df)} ETF (sotto €{min_avg_value:,.0f}). Restanti: {len(df)}.")
    if df.empty: return pd.DataFrame()

    df_pre_trend = len(df)
    df = df[df['is_above_ma200'] == True]
    log_area.append(
        f"- Filtro Trend MA200: Rimossi {df_pre_trend - len(df)} ETF. Restanti: {len(df)}.")
    if df.empty: return pd.DataFrame()

    df_pre_mom = len(df)
    df = df[(df['roc_6m'] > 0) & (df['roc_12m'] > 0)]
    log_area.append(
        f"- Filtro Momentum Assoluto: Rimossi {df_pre_mom - len(df)} ETF. Restanti: {len(df)}.")
    if df.empty: return pd.DataFrame()

    df_pre_prox = len(df)
    df = df[df['proximity_to_high'] >= min_proximity_to_high]
    log_area.append(
        f"- Filtro Prossimità Massimi: Rimossi {df_pre_prox - len(df)} ETF (< {min_proximity_to_high:.0%}). **Restanti per il ranking finale: {len(df)}.**")
    if df.empty: return pd.DataFrame()

    df['quality_score'] = df['calmar_ratio'].rank(pct=True) * 100
    score_12m = df['roc_12m'].rank(pct=True) * 100
    score_6m = df['roc_6m'].rank(pct=True) * 100
    score_3m = df['roc_3m'].rank(pct=True) * 100
    score_prox = df['proximity_to_high'].rank(pct=True) * 100
    df['momentum_score'] = (score_12m * 0.4) + (score_6m * 0.3) + (score_3m * 0.2) + (score_prox * 0.1)
    df['breakout_score'] = df['volatility_compression'].rank(pct=True, ascending=True) * 100
    df['low_vol_score'] = df['volatility_6m'].rank(pct=True, ascending=True) * 100

    weights = {"momentum": 0.40, "quality": 0.05, "breakout": 0.50, "low_vol": 0.05}

    df['final_score'] = (
            df['momentum_score'] * weights['momentum'] +
            df['quality_score'] * weights['quality'] +
            df['breakout_score'] * weights['breakout'] +
            df['low_vol_score'] * weights['low_vol']
    )

    df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)

    view_cols = [
        "ticker", "isin", "name", "final_score",
        "momentum_score", "quality_score", "breakout_score", "low_vol_score",
        "roc_6m", "calmar_ratio", "volatility_6m", "proximity_to_high", "avg_value_eur"
    ]
    return df.reindex(columns=view_cols)