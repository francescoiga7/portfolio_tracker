import logging
from typing import Dict, List, Optional, Iterable, Tuple
import pandas as pd
import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
import warnings
from etf_metrics.clients.yahoo_client import get_series, get_info, resolve_isin_one
from .etf_search_engine import discover_universe, get_unique_preferred_tickers
from etf_metrics.shared.config import DEFAULT_SEED_QUERIES
from .metrics import compute_metrics_from_series

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False, ttl=60 * 15)
def get_market_regime(as_of_date: Optional[date] = None) -> Dict:
    try:
        end_date = pd.to_datetime(as_of_date) if as_of_date else pd.to_datetime(date.today())
        vix_series = get_series("^VIX", period="1y")
        current_vix = None
        regime = "Sconosciuto"
        if vix_series is not None and not vix_series.empty:
            series_vix_as_of = vix_series[vix_series.index <= end_date]
            if not series_vix_as_of.empty:
                current_vix = series_vix_as_of.iloc[-1]
                regime = "Avverso al Rischio" if current_vix > 20 else "Favorevole al Rischio"
        gspc_series = get_series("^GSPC", period="2y")
        market_trend = "Sconosciuto"
        if gspc_series is not None and not gspc_series.empty:
            series_gspc_as_of = gspc_series[gspc_series.index <= end_date]
            if len(series_gspc_as_of) > 200:
                sma200_gspc = series_gspc_as_of.rolling(window=200).mean().iloc[-1]
                current_price_gspc = series_gspc_as_of.iloc[-1]
                if pd.notna(sma200_gspc) and pd.notna(current_price_gspc):
                    market_trend = "Rialzista" if current_price_gspc > sma200_gspc else "Ribassista"
        return {"vix": current_vix, "regime": regime, "market_trend": market_trend}
    except Exception as e:
        logger.error(f"Errore in get_market_regime: {e}")
        return {"vix": None, "regime": "Sconosciuto", "market_trend": "Sconosciuto"}

@st.cache_data(show_spinner="Caricamento e Download Massivo dati...", ttl=60 * 30)
def fetch_screener_data(
        log_area: List[str],
        specific_isins: Optional[List[str]] = None,
        instrument_types: List[str] = ["ETF", "ETP", "ETN"],
        queries: Iterable[str] = DEFAULT_SEED_QUERIES,
        quotes_per_query: int = 200
) -> List[Dict]:
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
        unique_tickers = get_unique_preferred_tickers(raw_tickers)
        log_area.append(f"- **Universo finale da analizzare: {len(unique_tickers)} ticker unici.**")
    if not unique_tickers:
        return []
    fetched_data = []
    log_area.append(f"**2. Avvio Download Massivo per {len(unique_tickers)} strumenti...**")
    try:
        bulk_df = yf.download(unique_tickers, period="5y", group_by='ticker', auto_adjust=False, threads=True)
        is_single = len(unique_tickers) == 1
        for ticker in unique_tickers:
            try:
                if is_single:
                    df = bulk_df.copy()
                else:
                    if ticker not in bulk_df.columns.levels[0]:
                        continue
                    df = bulk_df[ticker].copy()
                df = df.dropna(how='all')
                if 'Close' not in df.columns or df.empty or len(df) < 50:
                    continue
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                info = get_info(ticker=ticker)
                fetched_data.append({
                    "ticker": ticker,
                    "series": df,
                    "name": info.get("longName") or info.get("shortName") or ticker,
                    "isin": info.get("isin") or (ticker if len(ticker) == 12 else None)
                })
            except Exception:
                continue
    except Exception as e:
        log_area.append(f"⚠️ Errore critico nel bulk download: {e}")
        return []
    log_area.append(f"- **Download completato correttamente per {len(fetched_data)} strumenti.**")
    return fetched_data

def _calculate_metrics_from_series(ticker: str, series_full: pd.DataFrame, as_of_date: date, log_entry: List[str], name: str, isin: str) -> Optional[Dict]:
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            if 'Close' not in series_full.columns:
                return None
            close_prices_full = series_full['Close']
            end_date = pd.to_datetime(as_of_date)
            series = close_prices_full[close_prices_full.index <= end_date]
            volume_series = series_full['Volume'][series_full.index <= end_date] if 'Volume' in series_full.columns else None
            if series.empty or len(series) < 252:
                return None
            avg_value_eur = 0
            if volume_series is not None and not volume_series.empty:
                recent_vol = volume_series.iloc[-63:]
                avg_vol = recent_vol.mean()
                if pd.notna(avg_vol) and avg_vol > 0:
                    avg_value_eur = avg_vol * series.iloc[-1]
            sma200 = series.rolling(window=200).mean().iloc[-1]
            current_price = series.iloc[-1]
            is_above_ma200 = (current_price > sma200) if pd.notna(sma200) and pd.notna(current_price) else False
            series_12m = series[series.index >= (end_date - pd.DateOffset(months=12))]
            if len(series_12m) < 250:
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
            if pd.notna(sma20.iloc[-1]) and sma20.iloc[-1] != 0:
                bollinger_width_series = (4 * std20) / sma20
                bw_window = bollinger_width_series[bollinger_width_series.index >= (end_date - pd.DateOffset(months=6))]
                min_bw_6m = bw_window.min()
                current_bw = bollinger_width_series.iloc[-1]
                volatility_compression = current_bw / min_bw_6m if (min_bw_6m > 0 and pd.notna(min_bw_6m)) else None
            else:
                volatility_compression = None
            returns_6m = series_6m.pct_change().dropna()
            volatility_6m = returns_6m.std(ddof=1) * np.sqrt(252) if len(returns_6m) >= 2 else None
            log_entry.append("OK")
            return {
                "ticker": ticker,
                "isin": isin,
                "name": name,
                "avg_value_eur": avg_value_eur,
                "is_above_ma200": is_above_ma200,
                "proximity_to_high": proximity_to_high,
                "calmar_ratio": calmar_ratio, "roc_12m": roc_12m,
                "roc_6m": roc_6m, "roc_3m": roc_3m,
                "volatility_compression": volatility_compression, "volatility_6m": volatility_6m,
            }
    except Exception as e:
        log_entry.append(f"ERRORE: {e}")
        return None

@st.cache_data(show_spinner="Calcolo metriche per la data selezionata...", ttl=60 * 15)
def calculate_all_metrics(fetched_data: List[Dict], as_of_date: date) -> Tuple[pd.DataFrame, List[str]]:
    log_area = []
    log_area.append(f"\n**3. Esecuzione Calcolo Metriche (in cache) alla data {as_of_date.strftime('%d/%m/%Y')}...**")
    all_metrics = []
    processing_log_details = []
    for data in fetched_data:
        log_entry = [data['ticker']]
        metrics = _calculate_metrics_from_series(data['ticker'], data['series'], as_of_date, log_entry, data.get('name'), data.get('isin'))
        if metrics:
            all_metrics.append(metrics)
        processing_log_details.append(log_entry)
    if not all_metrics:
        log_area.append("- Nessuna metrica calcolabile trovata.")
        return pd.DataFrame(), log_area
    df = pd.DataFrame(all_metrics)
    log_area.append(f"- {len(df)} ETF con metriche calcolabili.")
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
    log_area.append(f"\n**4. Filtraggio e Ranking (Veloce)...**")
    if metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()
    df_pre_liq = len(df)
    df = df[df['avg_value_eur'] >= min_avg_value]
    log_area.append(f"- Filtro Liquidità: Rimossi {df_pre_liq - len(df)} ETF. Restanti: {len(df)}.")
    if df.empty: return pd.DataFrame()
    df_pre_trend = len(df)
    df = df[df['is_above_ma200'] == True]
    log_area.append(f"- Filtro Trend MA200: Rimossi {df_pre_trend - len(df)} ETF. Restanti: {len(df)}.")
    if df.empty: return pd.DataFrame()
    df_pre_mom = len(df)
    df = df[(df['roc_6m'] > 0) & (df['roc_12m'] > 0)]
    log_area.append(f"- Filtro Momentum Assoluto: Rimossi {df_pre_mom - len(df)} ETF. Restanti: {len(df)}.")
    if df.empty: return pd.DataFrame()
    df_pre_prox = len(df)
    df = df[df['proximity_to_high'] >= min_proximity_to_high]
    log_area.append(f"- Filtro Prossimità Massimi: Rimossi {df_pre_prox - len(df)} ETF. Restanti: {len(df)}.")
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