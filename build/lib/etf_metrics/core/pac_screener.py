# -*- coding: utf-8 -*-
"""Screener tattico PAC: regime di mercato, caricamento dati, metriche, ranking.

Nessuna dipendenza da Streamlit: la cache e i progress bar sono gestiti dalla UI.
Le funzioni di caricamento restituiscono i log come valore di ritorno (cacheable).
"""
import logging
from typing import Dict, List, Optional, Iterable, Tuple
import pandas as pd
from datetime import date
import numpy as np
import polars as pl

from etf_metrics.clients.yahoo_client import (
    resolve_isin_one,
    get_series,
    get_series_bulk,
    get_names_bulk,
)
from etf_metrics.core.etf_search_engine import discover_universe, get_unique_preferred_tickers
from etf_metrics.shared.config import DEFAULT_SEED_QUERIES
from etf_metrics.core.data_manager import MarketDataManager

logger = logging.getLogger(__name__)


def get_market_regime(as_of_date: Optional[date] = None) -> Dict:
    """
    Restituisce lo stato del mercato.
    Gestisce errori di connessione restituendo un regime neutro (fallback).
    """
    try:
        end_date = pd.to_datetime(as_of_date) if as_of_date else pd.to_datetime(date.today())

        # VIX Check
        vix_series = get_series("^VIX", period="1y")
        current_vix = None
        regime = "Favorevole al Rischio"  # Default ottimista in caso di no-data

        if vix_series is not None and not vix_series.empty:
            series_vix_as_of = vix_series[vix_series.index <= end_date]
            if not series_vix_as_of.empty:
                current_vix = series_vix_as_of.iloc[-1]
                regime = "Avverso al Rischio" if current_vix > 35 else "Favorevole al Rischio"

        # SP500 Trend Check
        gspc_series = get_series("^GSPC", period="2y")
        market_trend = "Rialzista"  # Default ottimista

        if gspc_series is not None and not gspc_series.empty:
            series_gspc_as_of = gspc_series[gspc_series.index <= end_date]
            if len(series_gspc_as_of) > 200:
                sma200_gspc = series_gspc_as_of.rolling(window=200).mean().iloc[-1]
                current_price_gspc = series_gspc_as_of.iloc[-1]
                if pd.notna(sma200_gspc) and pd.notna(current_price_gspc):
                    market_trend = "Rialzista" if current_price_gspc > (sma200_gspc * 0.95) else "Ribassista"

        return {"vix": current_vix, "regime": regime, "market_trend": market_trend}

    except Exception as e:
        logger.error(f"Errore in get_market_regime: {e}")
        return {"vix": 15.0, "regime": "Sconosciuto (Fallback)", "market_trend": "Laterale"}


def fetch_screener_data(
    specific_isins: Optional[List[str]] = None,
    instrument_types: Tuple[str, ...] = ("ETF", "ETP", "ETN"),
    queries: Iterable[str] = DEFAULT_SEED_QUERIES,
    quotes_per_query: int = 200,
) -> Tuple[List[Dict], List[str]]:
    """
    Carica l'universo di strumenti (manualmente o per discovery) e aggiorna il DB.
    Ritorna (fetched_data, logs) dove logs è la lista dei messaggi di avanzamento.
    """
    logs: List[str] = []
    unique_tickers = []

    if specific_isins:
        logs.append(f"**1. Modalità Manuale: {len(specific_isins)} strumenti forniti.**")
        resolved_tickers = []
        for item in specific_isins:
            if len(item) == 12:
                ticker = resolve_isin_one(item)
                if ticker:
                    resolved_tickers.append(ticker)
            else:
                resolved_tickers.append(item)
        unique_tickers = list(filter(None, resolved_tickers))
        logs.append(f"- Trovati {len(unique_tickers)} ticker validi.")
    else:
        logs.append(f"**1. Discovery Automatica...**")
        raw_tickers = discover_universe(queries, quotes_per_query=quotes_per_query, instrument_types=list(instrument_types))
        unique_tickers = get_unique_preferred_tickers(raw_tickers)

    if not unique_tickers:
        return [], logs

    db_manager = MarketDataManager()

    tickers_to_download = db_manager.get_tickers_needing_update(unique_tickers)
    already_updated = len(unique_tickers) - len(tickers_to_download)

    if already_updated > 0:
        logs.append(f"- Cache Locale: {already_updated} ticker già aggiornati oggi.")

    if tickers_to_download:
        logs.append(f"**2. Download Cloud per {len(tickers_to_download)} strumenti (batch paralleli)...**")
        try:
            # Download in batch: isolati, ripristinabili, con skip automatico dei
            # ticker in cache negativa e registrazione dei nuovi "no data found".
            downloaded_data = get_series_bulk(
                tickers_to_download,
                period="5y",
                progress_callback=lambda done, total, ok: logs.append(
                    f"- Batch {done}/{total} completato ({ok} strumenti scaricati).")
                if total <= 5 or done == total or done % max(1, total // 5) == 0 else None,
            )

            failed_count = len(tickers_to_download) - len(downloaded_data)
            if failed_count:
                logs.append(f"- ⏭️ {failed_count} ticker senza dati: aggiunti alla cache negativa "
                            f"(non verranno più scaricati).")

            if downloaded_data:
                # Nomi in parallelo (una sola volta: poi restano nel DB)
                new_names = get_names_bulk(list(downloaded_data.keys()))
                db_manager.save_bulk_data(downloaded_data, names_dict=new_names)
                logs.append(f"- Salvati {len(downloaded_data)} nuovi ticker nel Database.")
        except Exception as e:
            logs.append(f"⚠️ Errore download: {e}")

    logs.append(f"**3. Caricamento Dati Unificati...**")

    loaded_data_dict = db_manager.load_data(unique_tickers)
    ticker_names = db_manager.get_ticker_names(unique_tickers)

    fetched_data = []
    for ticker, df in loaded_data_dict.items():
        if df.empty or len(df) < 50:
            continue

        asset_name = ticker_names.get(ticker, ticker)

        fetched_data.append({
            "ticker": ticker,
            "series": df,
            "name": asset_name,
            "isin": db_manager.resolve_ticker_to_isin(ticker) if hasattr(db_manager, 'resolve_ticker_to_isin') else ""
        })

    logs.append(f"- **Pronti per analisi: {len(fetched_data)} strumenti.**")
    return fetched_data, logs


def calculate_all_metrics(fetched_data: List[Dict], as_of_date: date) -> Tuple[pd.DataFrame, List[str]]:
    """Calcola le metriche multi-fattore (engine Polars). Ritorna (df, logs)."""
    log_area: List[str] = []
    log_area.append(f"\n**4. Esecuzione Calcolo Metriche (Polars Accelerated)...**")

    if not fetched_data:
        return pd.DataFrame(), log_area

    try:
        dfs_to_concat = []
        target_date_ts = pd.to_datetime(as_of_date)

        for item in fetched_data:
            df = item['series'].copy()
            df = df[df.index <= target_date_ts]
            if df.empty or len(df) < 60:
                continue

            df = df.reset_index()
            if 'Date' not in df.columns:
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'Date'})
                elif 'index' in df.columns:
                    df = df.rename(columns={'index': 'Date'})
                else:
                    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

            df['ticker'] = str(item['ticker'])
            df['name'] = str(item.get('name', item['ticker']))
            df['isin'] = str(item.get('isin', ''))

            cols = ['Date', 'Close', 'Volume', 'ticker', 'name', 'isin']

            existing = [c for c in cols if c in df.columns]
            df_subset = df[existing].copy()

            if 'Close' in df_subset.columns:
                df_subset['Close'] = df_subset['Close'].astype(float)
            if 'Volume' in df_subset.columns:
                df_subset['Volume'] = df_subset['Volume'].fillna(0).astype(float)

            dfs_to_concat.append(df_subset)

        if not dfs_to_concat:
            log_area.append("⚠️ Nessun dato storico sufficiente trovato dopo il filtro data.")
            return pd.DataFrame(), log_area

        full_pdf = pd.concat(dfs_to_concat, ignore_index=True)

        if 'Volume' not in full_pdf.columns:
            full_pdf['Volume'] = 0.0
        if 'isin' not in full_pdf.columns:
            full_pdf['isin'] = ""
        if 'Date' not in full_pdf.columns:
            raise ValueError("Colonna 'Date' mancante dopo la concatenazione.")

        lf = pl.from_pandas(full_pdf).lazy()

        metrics_lf = (
            lf.sort("Date")
            .group_by(["ticker", "name", "isin"])
            .agg([
                pl.last("Close").alias("current_price"),
                pl.col("Close").rolling_mean(window_size=200).last().alias("sma200"),

                pl.col("Close").rolling_mean(window_size=20).alias("sma20_series"),
                pl.col("Close").rolling_std(window_size=20).alias("std20_series"),

                pl.col("Close").pct_change(n=252).last().alias("roc_12m"),
                pl.col("Close").pct_change(n=126).last().alias("roc_6m"),
                pl.col("Close").pct_change(n=63).last().alias("roc_3m"),
                pl.col("Close").tail(252).max().alias("high_52w"),

                (pl.col("Close").pct_change().tail(126).std() * np.sqrt(252)).alias("volatility_6m"),
                (pl.col("Volume").tail(63).mean() * pl.last("Close")).alias("avg_value_eur"),

                ((pl.col("Close").tail(252) / pl.col("Close").tail(252).cum_max() - 1).min()).alias("mdd_1y")
            ])
        )

        res_df = metrics_lf.collect().to_pandas()

        # Post-Processing
        res_df['is_above_ma200'] = res_df['current_price'] > res_df['sma200']

        res_df['proximity_to_high'] = res_df.apply(
            lambda x: x['current_price'] / x['high_52w'] if (pd.notna(x['high_52w']) and x['high_52w'] > 0) else None,
            axis=1
        )

        res_df['calmar_ratio'] = res_df.apply(
            lambda x: -x['roc_12m'] / x['mdd_1y'] if (pd.notna(x['mdd_1y']) and x['mdd_1y'] != 0) else 0, axis=1
        )

        def calc_vol_compression(row):
            try:
                if isinstance(row['sma20_series'], (list, np.ndarray)) and isinstance(row['std20_series'],
                                                                                      (list, np.ndarray)):
                    sma = np.array(row['sma20_series'], dtype=float)
                    std = np.array(row['std20_series'], dtype=float)
                else:
                    return None

                if len(sma) < 126:
                    return None

                with np.errstate(divide='ignore', invalid='ignore'):
                    bw = (4 * std) / sma

                recent_bw = bw[-126:]
                recent_bw = recent_bw[~np.isnan(recent_bw)]

                if len(recent_bw) == 0:
                    return None

                min_bw = np.min(recent_bw)
                current_bw = bw[-1] if not np.isnan(bw[-1]) else recent_bw[-1]

                if min_bw > 0:
                    return current_bw / min_bw
                return None
            except Exception:
                return None

        res_df['volatility_compression'] = res_df.apply(calc_vol_compression, axis=1)

        res_df = res_df.drop(columns=['sma20_series', 'std20_series', 'mdd_1y'], errors='ignore')

        essential = ['calmar_ratio', 'roc_12m', 'roc_6m', 'is_above_ma200']
        res_df = res_df.dropna(subset=essential)
        res_df['avg_value_eur'] = res_df['avg_value_eur'].fillna(0)

        log_area.append(f"- Calcolo completato per {len(res_df)} ETF.")
        return res_df, log_area

    except Exception as e:
        error_msg = f"⚠️ CRASH POLARS: {str(e)}"
        logger.error(error_msg)
        log_area.append(error_msg)
        return pd.DataFrame(), log_area


def filter_and_rank_metrics(metrics_df: pd.DataFrame, min_avg_value: float,
                            min_proximity_to_high: float, log_area: List[str]) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()

    df = df[df['avg_value_eur'] >= min_avg_value]
    df = df[df['is_above_ma200'] == True]  # noqa: E712
    df = df[(df['roc_6m'] > 0) & (df['roc_12m'] > 0)]
    df = df[df['proximity_to_high'] >= min_proximity_to_high]

    if df.empty:
        return pd.DataFrame()

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

    cols = [
        "ticker", "isin", "name", "final_score",
        "momentum_score", "quality_score", "breakout_score", "low_vol_score",
        "roc_6m", "calmar_ratio", "volatility_6m", "proximity_to_high", "avg_value_eur"
    ]

    return df.sort_values("final_score", ascending=False).reset_index(drop=True)[cols]
