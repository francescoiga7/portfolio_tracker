# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional, Iterable
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
    try:
        vix_series = get_series("^VIX", period="5y")
        if vix_series is None or vix_series.empty:
            return {"vix": None, "regime": "Sconosciuto"}

        end_date = pd.to_datetime(as_of_date) if as_of_date else vix_series.index.max()
        series_as_of = vix_series[vix_series.index <= end_date]
        if series_as_of.empty:
            return {"vix": None, "regime": "Dati VIX non disponibili per la data"}

        current_vix = series_as_of.iloc[-1]
        regime = "Avverso al Rischio" if current_vix > 20 else "Favorevole al Rischio"
        return {"vix": current_vix, "regime": regime}
    except Exception:
        return {"vix": None, "regime": "Sconosciuto"}


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
    """
    unique_tickers = []
    if specific_isins:
        log_area.append(f"**1. Modalità Manuale: {len(specific_isins)} strumenti forniti.**")
        with st.spinner("Risoluzione ISIN/Ticker..."):
            resolved_tickers = []
            for item in specific_isins:
                if len(item) == 12:  # Assumiamo sia un ISIN
                    ticker = resolve_isin_one(item)
                    if ticker:
                        resolved_tickers.append(ticker)
                else:  # Assumiamo sia già un ticker
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
        # Per il trading tattico serve il DataFrame completo
        series_full = get_series(ticker, period="5y", as_dataframe=True)
        if series_full is not None and not series_full.empty:
            # Per mantenere la compatibilità, salviamo il df ma lo chiamiamo 'series'
            fetched_data.append({"ticker": ticker, "series": series_full})
        progress_bar.progress((i + 1) / len(unique_tickers), text=f"Download: {ticker}")
    progress_bar.empty()
    log_area.append(f"- **Download completato per {len(fetched_data)} strumenti con dati storici sufficienti.**")
    return fetched_data

import warnings


def _calculate_metrics_from_series(ticker: str, series_full: pd.DataFrame, as_of_date: date, log_entry: List[str]) -> \
        Optional[Dict]:  # <-- MODIFICA 1: Il tipo corretto è DataFrame
    """Calcola le metriche per un singolo ETF a partire dalla sua serie storica completa."""
    try:
        # --- Blocco per catturare i warnings ---
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            # <-- MODIFICA 2: Estrai solo la colonna 'Close' per i calcoli
            if 'Close' not in series_full.columns:
                log_entry.append("ERRORE: La colonna 'Close' non è presente nei dati.")
                return None

            close_prices_full = series_full['Close']

            # <-- MODIFICA 3: Usa la serie di prezzi di chiusura, non l'intero DataFrame
            end_date = pd.to_datetime(as_of_date)
            series = close_prices_full[close_prices_full.index <= end_date]

            if series.empty or len(series) < 252:
                log_entry.append("Dati storici insufficienti alla data selezionata.")
                return None

            info = get_info(ticker=ticker) or {}
            avg_volume_3m = info.get("averageDailyVolume3Month")
            # Ora series.iloc[-1] è un singolo numero (l'ultimo prezzo di chiusura)
            avg_value_eur = (avg_volume_3m * series.iloc[-1]) if avg_volume_3m is not None else 0

            series_12m = series[series.index >= (end_date - pd.DateOffset(months=12))]
            if len(series_12m) < 250:
                log_entry.append("Storico inferiore a 12 mesi.")
                return None

            series_6m = series_12m[series_12m.index >= (end_date - pd.DateOffset(months=6))]
            series_3m = series_6m[series_6m.index >= (end_date - pd.DateOffset(months=3))]

            # Ora tutti i calcoli vengono eseguiti sulla serie di prezzi, producendo singoli valori
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
                "avg_value_eur": avg_value_eur, "calmar_ratio": calmar_ratio, "roc_12m": roc_12m,
                "roc_6m": roc_6m, "roc_3m": roc_3m, "proximity_to_high": proximity_to_high,
                "volatility_compression": volatility_compression, "volatility_6m": volatility_6m,
            }

    except Exception as e:
        log_entry.append(f"ERRORE per il ticker **{ticker}**: {e}")
        print(f"DEBUG - Errore bloccante per il ticker {ticker}: {e}")
        return None


def process_screener_rankings(fetched_data: List[Dict], min_avg_value: float, as_of_date: date,
                              log_area: List[str]) -> pd.DataFrame:
    """
    Fase 2 (Veloce): Elabora i dati storici pre-caricati per calcolare la classifica
    alla data specificata.
    """
    log_area.append(f"\n**3. Analisi e Ranking alla data {as_of_date.strftime('%d/%m/%Y')}...**")

    all_metrics = []
    st.session_state.debug_log_processing = []  # Pulisce il log di elaborazione
    for data in fetched_data:
        log_entry = [data['ticker']]
        metrics = _calculate_metrics_from_series(data['ticker'], data['series'], as_of_date, log_entry)
        if metrics:
            all_metrics.append(metrics)
        st.session_state.debug_log_processing.append(log_entry)

    if not all_metrics: return pd.DataFrame()
    df = pd.DataFrame(all_metrics)
    log_area.append(f"- {len(df)} ETF con metriche calcolabili.")

    essential_cols = ['calmar_ratio', 'roc_12m', 'roc_6m', 'roc_3m', 'proximity_to_high', 'volatility_compression',
                      'volatility_6m']
    df = df.dropna(subset=essential_cols)
    log_area.append(f"- {len(df)} ETF dopo aver rimosso quelli con dati metrici incompleti.")

    df['avg_value_eur'] = df['avg_value_eur'].fillna(0)
    df_pre_liq = len(df)
    df = df[df['avg_value_eur'] >= min_avg_value]
    log_area.append(
        f"- Rimossi {df_pre_liq - len(df)} ETF per bassa liquidità (sotto €{min_avg_value:,.0f}). Restanti: {len(df)}.")

    df_pre_mom = len(df)
    df = df[(df['roc_6m'] > 0) & (df['roc_12m'] > 0)]
    log_area.append(
        f"- Rimossi {df_pre_mom - len(df)} ETF per momentum assoluto negativo. **Restanti per il ranking finale: {len(df)}.**")
    if df.empty: return pd.DataFrame()

    df['quality_score'] = df['calmar_ratio'].rank(pct=True) * 100
    score_12m = df['roc_12m'].rank(pct=True) * 100
    score_6m = df['roc_6m'].rank(pct=True) * 100
    score_3m = df['roc_3m'].rank(pct=True) * 100
    score_prox = df['proximity_to_high'].rank(pct=True) * 100
    df['momentum_score'] = (score_12m * 0.4) + (score_6m * 0.3) + (score_3m * 0.2) + (score_prox * 0.1)
    df['breakout_score'] = df['volatility_compression'].rank(pct=True, ascending=True) * 100
    df['low_vol_score'] = df['volatility_6m'].rank(pct=True, ascending=True) * 100

    weights = {"momentum": 0.50, "quality": 0.25, "breakout": 0.15, "low_vol": 0.10}
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
        "roc_6m", "calmar_ratio", "volatility_6m", "avg_value_eur"
    ]
    return df.reindex(columns=view_cols)