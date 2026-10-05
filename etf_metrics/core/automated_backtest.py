# -*- coding: utf-8 -*-
"""
Backtest algoritmico multi-strategia (Xetra / ETF / Azioni).

Ogni strategia implementa l'interfaccia `BaseStrategy`:
  - prepare(df, spy_df, cfg): aggiunge colonne / indicatori specifici
  - can_buy(ctx):            dice se il regime di mercato ammette acquisti
  - entry(row, ctx):         punteggia i candidati all'ingresso (score, reason)
  - exit_signal(pos, row):   regole di uscita specifiche -> (reason, portion, price)
  - update_position(pos):    manutenzione posizione (trailing stop, flag, ecc.)

Il motore (`run_market_aware_backtest`) gestisce portafoglio, commissioni,
tasse, limiti mensili di acquisto e filtro di correlazione in modo identico
per tutte le strategie: il confronto tra algoritmi e' quindi "fair".

Inoltre:
  - `generate_fake_market_data`: crea dati OHLCV sintetici (scenari bull /
    bear / crash / laterali) per simulare SENZA connessione a Yahoo Finance.
  - `run_strategy_comparison`: esegue piu' strategie sugli stessi dati.
  - `compute_buy_and_hold_curves`: benchmark buy & hold per battere S&P500.
"""
import pandas as pd
import numpy as np
import logging
import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from typing import Callable, Optional, Dict, List, Tuple

from etf_metrics.clients.yahoo_client import get_series_bulk
from etf_metrics.core.data_manager import MarketDataManager

logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE MOTORE (default, sovrascrivibile per strategia) ---
CONFIG = {
    'MAX_POSITIONS': 3,
    'REBALANCE_DAYS': 5,
    'ENABLE_MONTHLY_LIMIT': True,
    'MAX_BUYS_PER_MONTH': 4,
    'COMMISSION': 2.0,
    'SPY_TICKER': 'SPY',
    'MACRO_TREND_FILTER': True,
    'BEAR_VOLATILITY_THRESHOLD': 3.0,
    'MIN_RS_SCORE': 80,
    'MIN_ADX': 25,
    'MIN_PROX_HIGH': 0.85,
    'STOP_LOSS_ATR_MULT': 2.5,  # aumentare a 3 aumenta volatilità e rendimento
    'TIME_STOP_DAYS': 21,
    'TP1_PCT': 0.12,  # 0.08, migliore 0.12
    'MAX_CORRELATION': 0.65
}


# =====================================================================
# 1. GENERATORE DATI FAKE (nessuna connessione a Yahoo Finance richiesta)
# =====================================================================

FAKE_SCENARIOS: Dict[str, str] = {
    "mixed_regimes": "Mercato realistico: bull → correzione → bear → rimbalzo → nuovo bull (consigliato)",
    "bull_market": "Mercato rialzista prolungato con pullback moderati",
    "bear_market": "Mercato ribassista prolungato con rally ribassisti",
    "crash_recovery": "Bull lento → crollo rapido -35% → ripresa a V",
    "sideways_chop": "Mercato laterale e volatile (lo scenario più difficile)",
}

# Segmenti: (giorni indicativi, rendimento totale del segmento, volatilità annua)
_SCENARIO_SEGMENTS = {
    "mixed_regimes": [
        (320, 0.35, 0.15), (60, -0.12, 0.22), (170, -0.28, 0.30),
        (200, 0.32, 0.24), (330, 0.45, 0.14), (60, -0.10, 0.20), (240, 0.28, 0.15),
    ],
    "bull_market": [
        (500, 0.65, 0.13), (70, -0.12, 0.18), (400, 0.50, 0.12),
        (100, -0.08, 0.16), (300, 0.40, 0.13),
    ],
    "bear_market": [
        (120, 0.08, 0.14), (380, -0.45, 0.32), (150, 0.15, 0.22),
        (300, -0.35, 0.28), (150, 0.10, 0.18),
    ],
    "crash_recovery": [
        (400, 0.40, 0.12), (35, -0.35, 0.55), (180, 0.50, 0.38), (400, 0.45, 0.13),
    ],
    "sideways_chop": [
        (600, 0.02, 0.20), (100, -0.15, 0.26), (200, 0.12, 0.18),
        (100, -0.12, 0.24), (300, 0.05, 0.21),
    ],
}

# Roster fisso: ruoli "archetipo" per rendere interpretabili i risultati.
#   (ticker, descrizione, beta vs benchmark, vol idiosincratica annua, alpha annuo, prezzo iniziale, volume base)
_FAKE_ROSTER = [
    ("FTEC",  "Tech Growth",        1.70, 0.20,  0.055, 120.0, 800_000),
    ("FAI",   "AI & Semicond.",     1.90, 0.26,  0.070,  90.0, 900_000),
    ("FBIO",  "Biotech",            1.35, 0.32,  0.010,  60.0, 300_000),
    ("FFIN",  "Financials",         1.10, 0.13,  0.004,  80.0, 500_000),
    ("FENE",  "Energy",             0.95, 0.24, -0.015,  70.0, 400_000),
    ("FCONS", "Consumer Def.",      0.75, 0.10,  0.005, 100.0, 350_000),
    ("FGLD",  "Oro (difensivo)",   -0.15, 0.14,  0.018,  85.0, 250_000),
    ("FBND",  "Obbligazioni",        0.05, 0.05,  0.020,  95.0, 200_000),
    ("FINV",  "Inverse Market",    -1.00, 0.15, -0.025,  50.0, 600_000),
]


def _scale_segments(segs: List[Tuple[int, float, float]], total_days: int) -> List[Tuple[int, float, float]]:
    """Scala proporzionalmente i segmenti dello scenario per riempire esattamente total_days."""
    tot = sum(s[0] for s in segs) or 1
    scaled, acc = [], 0
    for i, (d, r, v) in enumerate(segs):
        nd = max(1, int(round(d * total_days / tot)))
        if i == len(segs) - 1:
            nd = max(1, total_days - acc)
        nd = min(nd, max(1, total_days - acc))
        scaled.append((nd, r, v))
        acc += nd
    if acc < total_days:
        d0, r0, v0 = scaled[-1]
        scaled[-1] = (d0 + (total_days - acc), r0, v0)
    return scaled


def _build_benchmark_returns(rng: np.random.Generator, segs, n_days: int) -> np.ndarray:
    """Costruisce i rendimenti daily del benchmark seguendo i segmenti di scenario."""
    rets = np.zeros(n_days)
    pos = 0
    for days, tot_ret, ann_vol in segs:
        d = min(days, n_days - pos)
        if d <= 0:
            break
        drift = tot_ret / d
        vol_d = ann_vol / np.sqrt(252.0)
        z = rng.standard_normal(d)
        # Code grasse nei regimi stressati (bear / crash): rari salti di 2-4 sigma
        if ann_vol >= 0.25:
            mask = rng.random(d) < 0.025
            signs = np.where(rng.random(d) < 0.75, -1.0, 1.0)
            z = z + mask * signs * rng.uniform(2.0, 4.0, size=d)
        rets[pos:pos + d] = drift + vol_d * z
        pos += d
    return rets


def _ohlc_from_path(idx, close: np.ndarray, rets: np.ndarray, rng: np.random.Generator,
                    base_volume: float) -> pd.DataFrame:
    """Costruisce OHLC + Volume realistici a partire dalla serie dei close."""
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.004, n - 1))

    spread = np.abs(rng.normal(0.004, 0.002, n)) + np.abs(rets) * 0.5 + 0.002
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1.0, n)) * spread)
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1.0, n)) * spread)

    # Volume: rumore lognormale + espansione sui giorni di grosso movimento (orderflow realistico)
    volume = base_volume * np.exp(rng.normal(0, 0.30, n)) * (1 + 8.0 * np.abs(rets))

    o = np.round(open_, 2); h = np.round(high, 2); l = np.round(low, 2); c = np.round(close, 2)
    h = np.maximum(np.maximum(h, o), c)
    l = np.minimum(np.minimum(l, o), c)

    return pd.DataFrame(
        {"Open": o, "High": h, "Low": l, "Close": c, "Volume": volume.astype(np.int64)},
        index=idx,
    )


def generate_fake_market_data(
    n_random_tickers: int = 4,
    years: float = 4.0,
    seed: int = 42,
    scenario: str = "mixed_regimes",
    benchmark_ticker: str = "SPY",
    end_date=None,
    warmup_days: int = 320,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Genera un universo di dati OHLCV sintetici per i backtest.

    Ritorna (data_dict, meta):
      data_dict: {ticker: DataFrame[Open, High, Low, Close, Volume]}, benchmark incluso
      meta:      {'sim_start': data di inizio simulazione (dopo il warmup), ...}

    Gli archetipi fissi (tech, biotech, oro, obbligazioni, inverse, ...)
    rendono interpretabile il comportamento delle strategie in ogni regime.
    """
    scenario = scenario if scenario in _SCENARIO_SEGMENTS else "mixed_regimes"
    rng = np.random.default_rng(seed)

    sim_days = max(int(years * 252), 126)
    total_days = warmup_days + sim_days
    end = pd.to_datetime(end_date) if end_date is not None else pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=end, periods=total_days)

    # --- Benchmark (proxy S&P500) ---
    segs = _scale_segments(_SCENARIO_SEGMENTS[scenario], total_days)
    bench_ret = _build_benchmark_returns(rng, segs, total_days)
    bench_close = 500.0 * np.cumprod(1.0 + bench_ret)

    data: Dict[str, pd.DataFrame] = {}
    data[benchmark_ticker] = _ohlc_from_path(idx, bench_close, bench_ret, rng, base_volume=50_000_000)

    # --- Universo titoli ---
    roster = list(_FAKE_ROSTER)
    for i in range(1, int(n_random_tickers) + 1):
        roster.append((
            f"FK{i:02d}", f"Random {i}",
            rng.uniform(0.6, 1.6), rng.uniform(0.08, 0.30), rng.uniform(-0.02, 0.05),
            rng.uniform(40.0, 150.0), rng.uniform(2e5, 8e5),
        ))

    # Stress days: nei giorni molto negativi del benchmark (flight to quality)
    stress = np.clip(-bench_ret / 0.02, 0.0, 2.0)

    for (tk, _desc, beta, idio, alpha, p0, base_vol) in roster:
        idio_d = idio / np.sqrt(252.0)
        alpha_d = alpha / 252.0
        ret = alpha_d + beta * bench_ret + idio_d * rng.standard_normal(total_days)

        if beta <= 0.10:      # difensivi (oro/bond) guadagnano nei giorni neri
            ret = ret + stress * 0.004
        elif beta >= 1.5:     # high beta soffre un filo di più
            ret = ret - stress * 0.002

        close = p0 * np.cumprod(1.0 + ret)
        data[tk] = _ohlc_from_path(idx, close, ret, rng, base_volume=float(base_vol))

    meta = {
        "scenario": scenario,
        "scenario_desc": FAKE_SCENARIOS[scenario],
        "seed": seed,
        "years": years,
        "warmup_days": warmup_days,
        "sim_start": idx[warmup_days],
        "end": idx[-1],
        "benchmark": benchmark_ticker,
        "tickers": [r[0] for r in roster],
        "descriptions": {r[0]: r[1] for r in roster},
    }
    return data, meta


# =====================================================================
# 2. INDICATORI BASE (invariati rispetto alla versione originale)
# =====================================================================

def calculate_advanced_metrics_vectorized(df, spy_df=None):
    if 'Close' not in df.columns:
        return df
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df['Vol_20'] = df['Close'].pct_change().rolling(20).std() * 100

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr14 = ranges.max(axis=1).rolling(14).sum()
    plus_di14 = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di14 = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    df['ADX'] = dx.rolling(14).mean()

    df['High_52w'] = df['Close'].rolling(252).max()
    df['Prox_High'] = df['Close'] / df['High_52w']

    df['RS_Score'] = 0.0
    if spy_df is not None:
        common = df.index.intersection(spy_df.index)
        if len(common) > 65:
            stock_ret = df['Close'].pct_change(63)
            spy_ret = spy_df['Close'].pct_change(63)
            rel_perf = (stock_ret.loc[common] - spy_ret.loc[common]) * 100
            df.loc[common, 'RS_Score'] = rel_perf

    return df


def _assess_market_regime(price_matrix, current_date, market_data, benchmark=None):
    """Regime di mercato nella data indicata (BULL / BEAR / VOLATILE / DANGER / NEUTRAL).

    - Con benchmark disponibile (es. SPY): Close vs SMA200, volatilità ATR, RSI.
      Se la data non è nell'indice del benchmark (calendari USA != Xetra, ad es.
      il 1° maggio o il 3 ottobre) viene usata l'ultima seduta disponibile
      precedente (lookup as-of) invece di degradare al proxy.
    - Senza benchmark: proxy equal-weight dell'universo sugli ultimi 200 giorni
      di TRADING. (FIX BUG STORICO: prima si usavano 200 giorni SOLARI ≈ 142
      sedute, quindi la soglia `len >= 200` non era mai raggiungibile e il
      regime restava per sempre NEUTRAL: nessuna strategia comprava mai e il
      backtest riportava tutte le metriche a 0 con i soli dati del DB locale.)
    """
    spy = benchmark or CONFIG['SPY_TICKER']
    spy_df = market_data.get(spy)
    row = None
    if spy_df is not None and not spy_df.empty:
        sub = spy_df.loc[:current_date]
        if len(sub):
            row = sub.iloc[-1]

    if row is None:
        hist = price_matrix.loc[:current_date]
        if len(hist) < 200:
            return "NEUTRAL"
        proxy = hist.tail(200).mean(axis=1)
        return "BULL" if proxy.iloc[-1] > proxy.mean() else "BEAR"

    if row['Close'] < row['SMA200']:
        return "BEAR"
    atr_pct = (row['ATR'] / row['Close']) * 100
    if atr_pct > CONFIG['BEAR_VOLATILITY_THRESHOLD']:
        return "VOLATILE"
    if row['RSI'] < 35:
        return "DANGER"
    return "BULL"


def _precompute_regimes(bench_df: pd.DataFrame,
                        sim_dates: pd.DatetimeIndex) -> Optional[pd.Series]:
    """Regime di mercato per ogni data di simulazione, in un'unica passata vettoriale.

    Replica esattamente `_assess_market_regime` (ramo benchmark) evitando il
    costo per-giorno: con universi grandi (es. tutto Xetra) la valutazione del
    regime giorno per giorno domina il tempo di simulazione.
    Ritorna una Series allineata a sim_dates, oppure None se il benchmark non
    ha gli indicatori necessari (il chiamante ricade nel calcolo per-giorno).
    """
    needed = {'Close', 'SMA200', 'ATR', 'RSI'}
    if bench_df is None or bench_df.empty or not needed.issubset(bench_df.columns):
        return None
    try:
        close, sma = bench_df['Close'], bench_df['SMA200']
        atr_pct = bench_df['ATR'] / close * 100.0
        rsi = bench_df['RSI']
        # Stessa precedenza dell'if-chain originale (il primo match vince):
        # BEAR > VOLATILE > DANGER > BULL. Con le maschere vettoriali si aplica
        # in ordine INVERSO: l'ultima scrittura ha la precedenza più alta.
        regime = pd.Series("BULL", index=bench_df.index, dtype=object)
        regime[rsi < 35] = "DANGER"
        regime[atr_pct > CONFIG['BEAR_VOLATILITY_THRESHOLD']] = "VOLATILE"
        regime[close < sma] = "BEAR"
        # Allineamento as-of alle date di simulazione (calendari diversi)
        return regime.reindex(sim_dates, method='ffill').fillna("NEUTRAL")
    except Exception as e:
        logger.debug(f"Precompute regimi non riuscito ({e}): fallback per-giorno")
        return None


# Chiave riservata (non può collidere con un ticker reale) del benchmark di
# riserva equal-weight dell'universo, usato quando il benchmark ufficiale
# (SPY) non è disponibile nel DB — es. universi Xetra puri.
BENCH_PROXY_KEY = '__UNIVERSE_BENCH__'


def build_benchmark_proxy(market_data: Dict[str, pd.DataFrame],
                          key: str = BENCH_PROXY_KEY) -> Optional[pd.DataFrame]:
    """Costruisce un benchmark equal-weight dell'universo (fallback senza SPY).

    Indice a ribilanciamento giornaliero: ogni giorno la variazione % di ogni
    titolo (calcolata sulla propria serie di sedute) viene mediata in parti
    uguali tra i titoli quotati quel giorno e composta in un livello. Così il
    peso di ogni titolo resta 1/N ogni giorno, indipendentemente da quando è
    stato quotato e da quanto è salito in passato.

    NOTA: non normalizzare i livelli e mediarli — un indice fatto così pesa
    ogni titolo in proporzione alla sua performance cumulata storica (paniere
    sovrappesato sui vincitori) e non è un benchmark equal-weight.

    OHLCV sintetici (per regime ATR/RSI e RS), utilizzabile come riferimento
    del benchmark quando nel DB non c'è SPY (es. universi Xetra puri).
    """
    frames = [df for t, df in market_data.items()
              if df is not None and not df.empty and t != key]
    if len(frames) < 2:
        return None

    idx = frames[0].index
    for df in frames[1:]:
        idx = idx.union(df.index)

    # Close dell'indice: media equal-weight dei ritorni giornalieri (ogni titolo
    # pesa 1/N ogni giorno, indipendentemente da performance cumulata passata)
    crets, orats, hrats, lrats = {}, {}, {}, {}
    for i, df in enumerate(frames):
        c = df['Close']
        r = c.pct_change()
        crets[i] = r.replace([np.inf, -np.inf], np.nan).reindex(idx)
        # scostamenti intraday rispetto al PROPRIO close: h >= max(o, 1),
        # l <= min(o, 1) per titolo -> le medie preservano l'ordine OHLC
        if {'Open', 'High', 'Low'} <= set(df.columns):
            cc = c.where(np.isfinite(c) & (c > 0))
            orats[i] = (df['Open'] / cc).reindex(idx)
            hrats[i] = (df['High'] / cc).reindex(idx)
            lrats[i] = (df['Low'] / cc).reindex(idx)

    mean_cr = pd.DataFrame(crets).mean(axis=1)  # solo i quotati quel giorno
    close = (1.0 + mean_cr.fillna(0.0)).cumprod()

    def _ratio_mean(d):
        if not d:
            return None
        return pd.DataFrame(d).mean(axis=1)

    o = _ratio_mean(orats)
    h = _ratio_mean(hrats)
    l = _ratio_mean(lrats)
    open_ = close * (o.fillna(1.0) if o is not None else 1.0)
    high = close * (h.fillna(1.0) if h is not None else 1.0)
    low = close * (l.fillna(1.0) if l is not None else 1.0)

    if 'Volume' in frames[0].columns:
        vols = pd.concat(
            [df['Volume'].astype(float).reindex(idx) for df in frames], axis=1)
        vol = vols.mean(axis=1)
    else:
        vol = pd.Series(1.0, index=idx)

    proxy = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low, 'Close': close,
        'Volume': vol,
    })
    # belt & braces: ordine OHLC garantito anche con set di titoli diversi
    # tra i campi (NaN indipendenti)
    proxy['High'] = proxy[['High', 'Open', 'Close']].max(axis=1)
    proxy['Low'] = proxy[['Low', 'Open', 'Close']].min(axis=1)
    proxy.index.name = 'date'
    return proxy


def check_correlation_strict(candidate, portfolio, price_matrix, current_date):
    if not portfolio:
        return True, None
    if candidate not in price_matrix.columns:
        return True, None
    start = current_date - timedelta(days=60)
    hist = price_matrix.loc[start:current_date]
    if len(hist) < 30:
        return True, None
    cand_series = hist[candidate]
    for p in portfolio:
        if p not in hist.columns:
            continue
        corr = cand_series.corr(hist[p])
        if corr > CONFIG['MAX_CORRELATION']:
            return False, p
    return True, None


def calculate_ai_smart_score(row):
    if row['Close'] < row['SMA50']:
        return 0, "Below SMA50"
    if row['Prox_High'] < CONFIG['MIN_PROX_HIGH']:
        return 0, "Too far from Highs"
    if row['ADX'] < CONFIG['MIN_ADX']:
        return 0, "Weak Trend"
    score = 50
    score += (row['RS_Score'] * 2)
    score += (row['ADX'] / 2)
    if row['Prox_High'] > 0.95:
        score += 10
    if row['Vol_20'] > 5.0:
        score -= 15
    if row['RSI'] > 85:
        score -= 25
    return max(0, score), f"AI Score: {score:.0f} | ProxHigh: {row['Prox_High']:.2f}"


def prepare_market_data(tickers, period="5y", allow_download=True, progress_callback=None):
    """Prepara i dati per il backtest.

    - Carica dal DB locale tutti i ticker richiesti (incluso il benchmark SPY)
    - Se allow_download=True, li aggiorna con logica DELTA (update_tickers):
      download completo SOLO per i ticker nuovi, aggiornamento incrementale per
      gli altri — la prima volta con DB vuoto il download avviene qui, nei giri
      successivi si scarica solo ciò che manca
    - progress_callback(done, total, label) opzionale per la UI
    """
    db_manager = MarketDataManager()
    all_tickers = list(set(tickers + [CONFIG['SPY_TICKER']]))
    if allow_download:
        try:
            from etf_metrics.core.data_updater import update_tickers
            update_tickers(all_tickers, period=period, skip_updated_today=True,
                           progress_callback=progress_callback)
        except Exception as e:
            logger.warning(f"Aggiornamento delta non riuscito ({e}): fallback bulk")
            # Fallback: comportamento storico (bulk completo per i non aggiornati oggi)
            missing = db_manager.get_tickers_needing_update(all_tickers)
            if missing:
                new_data = get_series_bulk(missing, period=period,
                                           progress_callback=progress_callback)
                if new_data:
                    db_manager.save_bulk_data(new_data)

    loaded = db_manager.load_data(all_tickers)
    # Indicatori base + benchmark di riserva: se SPY non è nel DB (es. universo
    # Xetra puro) viene costruita la proxy equal-weight dell'universo, così il
    # regime di mercato e l'RS restano significativi anche offline
    processed, _bench_key = _ensure_processed(loaded, CONFIG['SPY_TICKER'])
    return processed


def load_recent_market_data(tickers, period="2y", allow_download=True, min_rows=135):
    """Dati recenti per Scanner e Portafoglio Live, letti PRIMA dal DB locale.

    - Ticker già aggiornati oggi: letti dal DB, zero richieste di rete
    - Ticker mancanti o non aggiornati oggi: aggiornati via delta (update_tickers)
      e salvati nel DB, così i giri successivi sono completamente offline
    - Rate limit Yahoo (429) attivo: i download vengono sospesi e si usa comunque
      il DB locale per chi ha dati

    Ritorna {ticker: DataFrame OHLCV} SENZA indicatori (li calcola il chiamante).
    """
    dm = MarketDataManager()
    requested = [str(t).strip().upper() for t in (tickers or []) if t and str(t).strip()]
    if not requested:
        return {}
    all_tickers = list(dict.fromkeys(requested + [CONFIG['SPY_TICKER']]))

    if allow_download:
        try:
            from etf_metrics.core.data_updater import update_tickers
            update_tickers(all_tickers, period=period, skip_updated_today=True)
        except Exception as e:
            logger.warning(f"Aggiornamento dati non riuscito, uso il DB locale: {e}")

    loaded = dm.load_data(all_tickers)
    return {t: df for t, df in loaded.items()
            if df is not None and not df.empty and len(df) >= min_rows}


def _ensure_processed(market_data: Dict[str, pd.DataFrame], benchmark: str,
                      min_rows: int = 200) -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
    """Garantisce che i dati (es. fake) abbiano gli indicatori base calcolati.

    Ritorna (dati_processati, chiave_benchmark_effettiva). Se il benchmark
    richiesto (es. SPY) non è disponibile, viene costruita e aggiunta una proxy
    equal-weight dell'universo (BENCH_PROXY_KEY): regimi di mercato, RS e curve
    buy&hold restano utilizzabili anche con un DB Xetra senza titoli USA.
    """
    market_data = dict(market_data)  # non mutare l'input del chiamante

    bench_key: Optional[str] = None
    if benchmark in market_data and market_data[benchmark] is not None \
            and not market_data[benchmark].empty:
        bench_key = benchmark
    elif BENCH_PROXY_KEY in market_data and market_data[BENCH_PROXY_KEY] is not None \
            and not market_data[BENCH_PROXY_KEY].empty:
        bench_key = BENCH_PROXY_KEY

    if bench_key is None:
        proxy = build_benchmark_proxy(market_data)
        if proxy is not None:
            market_data[BENCH_PROXY_KEY] = proxy
            bench_key = BENCH_PROXY_KEY

    bench_df = market_data.get(bench_key) if bench_key else None
    if bench_df is not None and not bench_df.empty and 'SMA200' not in bench_df.columns:
        bench_df = calculate_advanced_metrics_vectorized(bench_df, None)
        market_data[bench_key] = bench_df

    out: Dict[str, pd.DataFrame] = {}
    for t, df in market_data.items():
        if df is None or df.empty:
            continue
        if 'SMA200' in df.columns:
            out[t] = df
        elif len(df) > min_rows:
            out[t] = calculate_advanced_metrics_vectorized(df, bench_df)
    return out, bench_key


# =====================================================================
# 3. FRAMEWORK STRATEGIE
# =====================================================================

class _MarketCtx:
    """Contesto di mercato passato alle strategie ad ogni giorno di simulazione."""
    __slots__ = ("regime", "is_crash", "current_date", "market_data", "price_matrix", "cfg", "benchmark")

    def __init__(self):
        self.regime = "NEUTRAL"
        self.is_crash = False
        self.current_date = None
        self.market_data = {}
        self.price_matrix = None
        self.cfg = dict(CONFIG)
        self.benchmark = "SPY"


class BaseStrategy:
    """Interfaccia comune di tutte le strategie di trading."""
    key = "base"
    name = "Base"
    description = ""
    min_entry_score = 60
    requires_bull = True      # acquisti ammessi solo in regime BULL
    use_stops = True          # il motore mantiene uno stop ATR

    defaults: Dict = {}       # override del CONFIG del motore

    def prepare(self, df: pd.DataFrame, spy_df=None, cfg=None) -> pd.DataFrame:
        return df

    def can_buy(self, ctx: _MarketCtx) -> bool:
        return (not self.requires_bull) or ctx.regime == "BULL"

    def entry(self, row, ctx: _MarketCtx):
        """Ritorna (score, reason). score < min_entry_score -> nessun acquisto."""
        return 0, ""

    def exit_signal(self, pos: Dict, row, ctx: _MarketCtx):
        """Ritorna None oppure (reason, portion, price). portion=1.0 vendita totale."""
        return None

    def update_position(self, pos: Dict, row, ctx: _MarketCtx):
        """Manutenzione giornaliera (trailing stop, ecc.). Chiamato solo se non si esce oggi."""
        pass

    def position_size_factor(self, row, ctx: _MarketCtx) -> float:
        return 1.0

    def initial_stop(self, row, ctx: _MarketCtx, cfg: Dict):
        if not self.use_stops:
            return None
        atr = row.get('ATR')
        if atr is None or pd.isna(atr) or atr <= 0:
            return None
        return row['Close'] - (atr * cfg['STOP_LOSS_ATR_MULT'])


# ---------------------------------------------------------------------
# 3a. AI Enhanced Momentum (strategia ORIGINALE, portata 1:1)
# ---------------------------------------------------------------------
class AIMomentumStrategy(BaseStrategy):
    key = "ai_momentum"
    name = "🧠 AI Enhanced Momentum (originale)"
    description = """
**Strategia originale dell'app** — trend following con filtro macro e ranking "AI Score".

- **Filtro macro:** acquisti solo in regime `BULL` (benchmark > SMA200, volatilità normale).
- **Ranking:** RS vs benchmark a 63gg, ADX > 25, prossimità ai massimi 52 settimane (> 85%).
- **Uscite:** stop ATR 2.5x, take profit parziale +12% ( vende 25% e porta lo stop a breakeven ),
  trailing stop, uscita macro in regime di crisi, time stop 21gg.
- **Frequenza:** ribilanciamento ogni 5 giorni, max 4 acquisti/mese.
"""
    defaults = {
        'MAX_POSITIONS': 3, 'REBALANCE_DAYS': 5, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 4, 'STOP_LOSS_ATR_MULT': 2.5, 'TP1_PCT': 0.12,
        'TIME_STOP_DAYS': 21,
    }

    def can_buy(self, ctx):
        return ctx.regime == "BULL"

    def entry(self, row, ctx):
        try:
            return calculate_ai_smart_score(row)
        except Exception:
            return 0, "Dati insufficienti"

    def exit_signal(self, pos, row, ctx):
        curr_price = row['Close']
        roi = (curr_price / pos['entry_price']) - 1

        if not pos.get('tp1_taken', False) and roi >= ctx.cfg['TP1_PCT']:
            pos['stop_loss'] = pos['entry_price'] * 1.01
            pos['tp1_taken'] = True
            return ("TP1 (Lock)", 0.25, curr_price)

        if ctx.is_crash and roi < 0.05:
            return ("MACRO RISK", 1.0, curr_price)

        days_held = (ctx.current_date - pos['entry_date']).days
        if days_held >= ctx.cfg['TIME_STOP_DAYS'] and roi < 0.01:
            return ("TIME STOP", 1.0, curr_price)
        return None

    def update_position(self, pos, row, ctx):
        curr_price = row['Close']
        roi = (curr_price / pos['entry_price']) - 1
        if pos.get('tp1_taken', False) or roi > 0.03:
            atr = row.get('ATR')
            if atr is not None and pd.notna(atr):
                new_stop = curr_price - (atr * ctx.cfg['STOP_LOSS_ATR_MULT'])
                if pos.get('stop_loss') is None or new_stop > pos['stop_loss']:
                    pos['stop_loss'] = new_stop


# ---------------------------------------------------------------------
# 3b. Dual Momentum (Antonacci) — la strategia "low commissioni"
# ---------------------------------------------------------------------
class DualMomentumStrategy(BaseStrategy):
    key = "dual_momentum"
    name = "🐢 Dual Momentum (rotazione mensile)"
    description = """
**Dual Momentum in stile Antonacci** — la strategia "low commissioni" per eccellenza.

- **Ogni 21 giorni** (rotazione mensile): classifica i ticker per **momentum a 12 mesi**.
- **Momentum assoluto:** acquista solo se il momentum del ticker è **positivo**
  (in bear market prolungato resta in cash: è il filtro anti-orma).
- **Uscita:** quando il momentum torna negativo → vendita e rotazione.
  Nessun stop loss giornaliero, solo un ampio stop di sicurezza 5x ATR.
- **Poche operazioni all'anno** (tipicamente 3-8): ideale se le commissioni ti mangiano i profitti.
"""
    requires_bull = False
    min_entry_score = 1
    defaults = {
        'MAX_POSITIONS': 2, 'REBALANCE_DAYS': 21, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 6, 'STOP_LOSS_ATR_MULT': 5.0, 'MOMENTUM_LOOKBACK': 252,
    }

    def prepare(self, df, spy_df=None, cfg=None):
        lb = (cfg or {}).get('MOMENTUM_LOOKBACK', 252)
        df = df.copy()
        df['MOM_LB'] = df['Close'].pct_change(lb)
        return df

    def entry(self, row, ctx):
        mom = row.get('MOM_LB')
        if mom is None or pd.isna(mom) or mom <= 0:
            return 0, "Momentum non positivo"
        lb = ctx.cfg.get('MOMENTUM_LOOKBACK', 252)
        return mom * 100.0, f"Dual Momentum {lb}gg: {mom:+.1%}"

    def exit_signal(self, pos, row, ctx):
        mom = row.get('MOM_LB')
        if mom is not None and pd.notna(mom) and mom < 0:
            return ("MOMENTUM FLIP", 1.0, row['Close'])
        return None


# ---------------------------------------------------------------------
# 3c. Bear Market / Regime Switcher
# ---------------------------------------------------------------------
class BearMarketStrategy(BaseStrategy):
    key = "bear_market"
    name = "🐻 Bear Market Regime Switcher"
    description = """
**Regime Switcher** — pensata per NON farsi massacrare (e persino guadagnare) nei bear market.

- **Regime `BULL`:** compra i titoli più forti (RS vs benchmark, ADX > 20, sopra SMA50).
- **Regime `BEAR`:** cerca i titoli che **salgono mentre il mercato scende**
  (ETF inversi, oro, obbligazioni — archetipi FINV / FGLD / FBND nei dati fake).
- **Regime `VOLATILE` / `DANGER`:** resta in cash (difesa attiva).
- **Uscite:** flip di regime con posizione in perdita, rottura SMA50, trailing stop 3x ATR,
  time stop 45gg. Max 2 posizioni, ribilanciamento ogni 10 giorni.
"""
    requires_bull = False
    min_entry_score = 60
    defaults = {
        'MAX_POSITIONS': 2, 'REBALANCE_DAYS': 10, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 4, 'STOP_LOSS_ATR_MULT': 3.0, 'TIME_STOP_DAYS': 45,
    }

    def prepare(self, df, spy_df=None, cfg=None):
        df = df.copy()
        df['MOM_63'] = df['Close'].pct_change(63)
        return df

    def can_buy(self, ctx):
        return ctx.regime in ("BULL", "BEAR")

    def entry(self, row, ctx):
        c = row['Close']
        sma50 = row.get('SMA50')
        rs = row.get('RS_Score', 0.0)
        adx = row.get('ADX')
        if sma50 is None or pd.isna(sma50):
            return 0, "Dati insufficienti"
        if adx is not None and pd.notna(adx) and adx < 20:
            return 0, "Trend debole (ADX)"

        if ctx.regime == "BULL":
            if c <= sma50:
                return 0, "Sotto SMA50"
            if pd.isna(rs) or rs <= 0:
                return 0, "RS non positivo"
            score = 50 + rs * 2 + ((adx if pd.notna(adx) else 20) / 2)
            return score, f"BULL: RS {rs:+.1f} ADX {adx:.0f}"

        if ctx.regime == "BEAR":
            mom = row.get('MOM_63')
            if c <= sma50:
                return 0, "Sotto SMA50 (bear)"
            if mom is None or pd.isna(mom) or mom <= 0:
                return 0, "Momentum 63gg non positivo"
            score = 55 + (rs if pd.notna(rs) else 0) * 2 + min(15.0, mom * 100.0)
            return score, f"BEAR SHIELD: Mom63 {mom:+.1%} RS {rs:+.1f}"

        return 0, "Regime non operativo"

    def exit_signal(self, pos, row, ctx):
        c = row['Close']
        roi = (c / pos['entry_price']) - 1
        sma50 = row.get('SMA50')

        # Flip di regime con posizione in perdita / poco profitto
        if ctx.regime != pos.get('regime_at_entry', ctx.regime) and roi < 0.05:
            return (f"REGIME FLIP ({ctx.regime})", 1.0, c)

        if sma50 is not None and pd.notna(sma50) and c < sma50:
            return ("SMA50 ROTTA", 1.0, c)

        days_held = (ctx.current_date - pos['entry_date']).days
        if days_held >= ctx.cfg['TIME_STOP_DAYS'] and roi < 0.01:
            return ("TIME STOP", 1.0, c)
        return None

    def update_position(self, pos, row, ctx):
        c = row['Close']
        roi = (c / pos['entry_price']) - 1
        if roi > 0.05:
            atr = row.get('ATR')
            if atr is not None and pd.notna(atr):
                new_stop = c - atr * ctx.cfg['STOP_LOSS_ATR_MULT']
                if pos.get('stop_loss') is None or new_stop > pos['stop_loss']:
                    pos['stop_loss'] = new_stop


# ---------------------------------------------------------------------
# 3d. Volume Profile
# ---------------------------------------------------------------------
def _rolling_volume_profile(df: pd.DataFrame, window: int = 60, bins: int = 24,
                            va_pct: float = 0.70):
    """Profilo del volume per prezzo rolling: ritorna (POC, VAH, VAL) come array.

    Versione vettorizzata (sliding_window_view + bincount con offset di riga):
    su un titolo da ~2500 sedute è ~10x più veloce del loop per-riga — su
    universi grandi (tutto Xetra) è la differenza tra minuti e secondi.
    Semantica identica alla versione originale a loop.
    """
    n = len(df)
    poc = np.full(n, np.nan); vah = np.full(n, np.nan); val = np.full(n, np.nan)
    if n <= window or window <= 0 or bins <= 0:
        return poc, vah, val

    close = df['Close'].to_numpy(float); high = df['High'].to_numpy(float)
    low = df['Low'].to_numpy(float); vol = np.nan_to_num(df['Volume'].to_numpy(float))
    tp = (close + high + low) / 3.0

    # Finestre [i-window, i) per i in [window, n): riga j <-> i = j + window.
    # [:-1] esclude l'ultima finestra, che conterrebbe il giorno corrente i
    # (la finestra del profilo è strettamente passata, come nell'originale).
    W_tp = np.lib.stride_tricks.sliding_window_view(tp, window)[:-1]
    W_vol = np.lib.stride_tricks.sliding_window_view(vol, window)[:-1]
    m = W_tp.shape[0]

    lo = np.nanmin(W_tp, axis=1)
    hi = np.nanmax(W_tp, axis=1)
    span = hi - lo
    tot = W_vol.sum(axis=1)
    valid = np.isfinite(lo) & np.isfinite(hi) & (span > 0) & (tot > 0)

    # Indice di bin per elemento: np.digitize(x, linspace(lo, hi, bins+1)) - 1
    # equivale a ceil((x-lo)/span*bins) - 1, clampato a [0, bins-1]
    safe_span = np.where(valid, span, 1.0)
    rel = np.where(valid[:, None], (W_tp - lo[:, None]) / safe_span[:, None], 0.0)
    b = np.clip(np.ceil(rel * bins).astype(np.int64) - 1, 0, bins - 1)

    # Istogramma pesato per riga in un solo bincount (riga*bin + colonna)
    rows = np.repeat(np.arange(m), window)
    hist = np.bincount(rows * bins + b.ravel(),
                       weights=np.where(valid[:, None], W_vol, 0.0).ravel(),
                       minlength=m * bins).reshape(m, bins)

    poc_i = hist.argmax(axis=1)
    lo_v = np.where(valid, lo, 0.0)

    # Espansione della value area attorno al POC (due punte, come l'originale)
    for j in np.nonzero(valid)[0]:
        h = hist[j]
        target = h.sum() * va_pct
        cum = h[poc_i[j]]; up = int(poc_i[j]) + 1; dn = int(poc_i[j]) - 1
        while cum < target and (up < bins or dn >= 0):
            up_val = h[up] if up < bins else -1.0
            dn_val = h[dn] if dn >= 0 else -1.0
            if up_val >= dn_val:
                cum += max(0.0, up_val); up += 1
            else:
                cum += max(0.0, dn_val); dn -= 1
        i = j + window
        poc[i] = lo_v[j] + (poc_i[j] + 0.5) * safe_span[j] / bins
        vah[i] = lo_v[j] + min(up, bins) * safe_span[j] / bins
        val[i] = lo_v[j] + max(dn + 1, 0) * safe_span[j] / bins
    return poc, vah, val


class VolumeProfileStrategy(BaseStrategy):
    key = "volume_profile"
    name = "📊 Volume Profile (Value Area Breakout)"
    description = """
**Volume Profile** — dove "vive" il volume istituzionale.

- Costruisce il **profilo del volume per prezzo** su 60 giorni:
  `POC` (Point of Control, il prezzo più scambiato) e **Value Area al 70%**.
- **Acquista** quando il close rompe **sopra il Value Area High** con volume > 1.5x la media
  e prezzo sopra SMA50 (breakout da area di valore = accumulo completato).
- **Esce** quando il prezzo rientra nel value area sotto il POC (fallimento del breakout)
  o su trailing stop 3x ATR.
- Ribilanciamento ogni 5 giorni, max 3 posizioni.
"""
    min_entry_score = 60
    defaults = {
        'MAX_POSITIONS': 3, 'REBALANCE_DAYS': 5, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 4, 'STOP_LOSS_ATR_MULT': 3.0, 'VP_WINDOW': 60,
    }

    def prepare(self, df, spy_df=None, cfg=None):
        df = df.copy()
        window = int((cfg or {}).get('VP_WINDOW', 60))
        poc, vah, val = _rolling_volume_profile(df, window=window)
        df['POC'] = poc; df['VAH'] = vah; df['VAL'] = val
        vol_sma = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / vol_sma.replace(0, np.nan)
        return df

    def entry(self, row, ctx):
        c = row['Close']
        vah, poc, sma50 = row.get('VAH'), row.get('POC'), row.get('SMA50')
        vr = row.get('Vol_Ratio')
        if any(v is None or pd.isna(v) for v in (vah, poc, sma50, vr)):
            return 0, "Dati insufficienti"
        if c <= sma50:
            return 0, "Sotto SMA50"
        if c <= vah:
            return 0, "Dentro value area"
        if vr < 1.5:
            return 0, f"Volume debole (x{vr:.1f})"

        rs = row.get('RS_Score', 0.0)
        score = 50 + min(25.0, (vr - 1.5) * 25.0) + float(np.clip(rs, -15, 25)) \
                + min(10.0, (c / vah - 1) * 200.0)
        return score, f"Breakout VAH {vah:.2f} | Vol x{vr:.1f} | POC {poc:.2f}"

    def exit_signal(self, pos, row, ctx):
        poc = row.get('POC')
        if poc is not None and pd.notna(poc) and row['Close'] < poc:
            return ("RIENTRO VALUE AREA", 1.0, row['Close'])
        return None

    def update_position(self, pos, row, ctx):
        c = row['Close']
        roi = (c / pos['entry_price']) - 1
        if roi > 0.05:
            atr = row.get('ATR')
            if atr is not None and pd.notna(atr):
                new_stop = c - atr * ctx.cfg['STOP_LOSS_ATR_MULT']
                if pos.get('stop_loss') is None or new_stop > pos['stop_loss']:
                    pos['stop_loss'] = new_stop


# ---------------------------------------------------------------------
# 3e. Order Flow (approssimato da candele daily)
# ---------------------------------------------------------------------
class OrderFlowStrategy(BaseStrategy):
    key = "orderflow"
    name = "🌊 Order Flow (CVD & Accumulo)"
    description = """
**Order Flow approssimato da candele daily** — stima la pressione compratrice reale.

- **Delta proxy:** `Volume × posizione del close nel range H-L` (CLV in [-1, +1]):
  close vicino all'alto con volume alto = netto acquista.
- **CVD** (Cumulative Volume Delta): somma cumulativa del delta, con la sua media 20gg.
- **Giorni di accumulo:** close nel 40% superiore del range con volume sopra la media.
- **Acquista** quando: sopra SMA50, CVD > media CVD (accumulo in corso) e
  **≥ 3 giorni di accumulo** negli ultimi 10, con RS positivo.
- **Esce** in distribuzione (CVD sotto la media per 3 giorni), a rottura SMA50 o trailing 3x ATR.
"""
    min_entry_score = 60
    defaults = {
        'MAX_POSITIONS': 3, 'REBALANCE_DAYS': 5, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 4, 'STOP_LOSS_ATR_MULT': 3.0,
    }

    def prepare(self, df, spy_df=None, cfg=None):
        df = df.copy()
        rng_ = (df['High'] - df['Low'])
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / rng_.mask(rng_ == 0)
        df['CLV'] = clv.fillna(0.0)

        df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
        df['DELTA'] = df['Volume'] * df['CLV']
        df['CVD'] = df['DELTA'].cumsum()
        df['CVD_SMA'] = df['CVD'].rolling(20).mean()

        acc = ((df['CLV'] > 0.6) & (df['Volume'] > df['Vol_SMA20'])).astype(int)
        df['ACC_DAYS'] = acc.rolling(10).sum()
        cvd_weak = (df['CVD'] < df['CVD_SMA']).astype(int)
        df['CVD_WEAK_DAYS'] = cvd_weak.rolling(3).sum()
        return df

    def entry(self, row, ctx):
        c = row['Close']
        sma50, cvd, cvd_sma = row.get('SMA50'), row.get('CVD'), row.get('CVD_SMA')
        acc = row.get('ACC_DAYS')
        if any(v is None or pd.isna(v) for v in (sma50, cvd, cvd_sma, acc)):
            return 0, "Dati insufficienti"
        if c <= sma50:
            return 0, "Sotto SMA50"
        if cvd <= cvd_sma:
            return 0, "CVD sotto la media"
        if acc < 3:
            return 0, f"Accumulo insufficiente ({acc:.0f}/10gg)"

        rs = row.get('RS_Score', 0.0)
        score = 45 + acc * 5.0 + float(np.clip(rs, -10, 15))
        return score, f"Accumulo {acc:.0f}gg | CVD>SMA | RS {rs:+.1f}"

    def exit_signal(self, pos, row, ctx):
        weak = row.get('CVD_WEAK_DAYS')
        sma50 = row.get('SMA50')
        if weak is not None and pd.notna(weak) and weak >= 3:
            return ("DISTRIBUZIONE (CVD)", 1.0, row['Close'])
        if sma50 is not None and pd.notna(sma50) and row['Close'] < sma50:
            return ("SMA50 ROTTA", 1.0, row['Close'])
        return None

    def update_position(self, pos, row, ctx):
        c = row['Close']
        roi = (c / pos['entry_price']) - 1
        if roi > 0.05:
            atr = row.get('ATR')
            if atr is not None and pd.notna(atr):
                new_stop = c - atr * ctx.cfg['STOP_LOSS_ATR_MULT']
                if pos.get('stop_loss') is None or new_stop > pos['stop_loss']:
                    pos['stop_loss'] = new_stop


# ---------------------------------------------------------------------
# 3f. Mean Reversion (pullback profondo)
# ---------------------------------------------------------------------
class MeanReversionStrategy(BaseStrategy):
    key = "mean_reversion"
    name = "🎯 Mean Reversion (RSI-2 Pullback)"
    description = """
**Mean Reversion** — compra i panic dip dei titoli in trend lungo.

- **Acquista** quando `RSI a 2 giorni < 10` (caduta brusca) MA il titolo resta in
  trend lungo (`Close > SMA200`): stai comprando il panico, non il declino.
- **Esce** al ripristino della media: `RSI-2 > 80` o close sopra SMA20.
- **Difese:** stop ATR 2.5x, time stop 10 giorni (se non recupera, si esce).
- Buon complemento al trend following: guadagna dove le strategie di trend stanno flat.
"""
    min_entry_score = 60
    defaults = {
        'MAX_POSITIONS': 3, 'REBALANCE_DAYS': 5, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 6, 'STOP_LOSS_ATR_MULT': 2.5, 'TIME_STOP_DAYS': 10,
    }

    def prepare(self, df, spy_df=None, cfg=None):
        df = df.copy()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(2).mean()
        loss = (-delta.clip(upper=0)).rolling(2).mean()
        rs = gain / loss.mask(loss == 0)
        df['RSI_2'] = (100 - 100 / (1 + rs)).fillna(50)
        df['SMA_20'] = df['Close'].rolling(20).mean()
        return df

    def entry(self, row, ctx):
        rsi2 = row.get('RSI_2'); sma200 = row.get('SMA200')
        if rsi2 is None or pd.isna(rsi2) or sma200 is None or pd.isna(sma200):
            return 0, "Dati insufficienti"
        if row['Close'] <= sma200:
            return 0, "Sotto SMA200 (no trend lungo)"
        if rsi2 >= 10:
            return 0, f"RSI2 {rsi2:.0f} non abbastanza basso"

        rs = row.get('RS_Score', 0.0)
        score = 60 + (10 - rsi2) * 1.5 + (10 if pd.notna(rs) and rs > 0 else 0)
        return score, f"Pullback RSI2 {rsi2:.0f} su trend SMA200"

    def exit_signal(self, pos, row, ctx):
        rsi2 = row.get('RSI_2'); sma20 = row.get('SMA_20')
        if rsi2 is not None and pd.notna(rsi2) and rsi2 > 80:
            return ("RIPRISTINO RSI2", 1.0, row['Close'])
        if sma20 is not None and pd.notna(sma20) and row['Close'] > sma20:
            return ("MEDIA RAGGIUNTA (SMA20)", 1.0, row['Close'])
        days_held = (ctx.current_date - pos['entry_date']).days
        if days_held >= ctx.cfg['TIME_STOP_DAYS']:
            return ("TIME STOP", 1.0, row['Close'])
        return None


# ---------------------------------------------------------------------
# 3g. Turtle / Donchian Breakout
# ---------------------------------------------------------------------
class TurtleBreakoutStrategy(BaseStrategy):
    key = "turtle_breakout"
    name = "🐢 Turtle Breakout (Donchian 55/20)"
    description = """
**Turtle Trading moderno** — il breakout che lascia correre i trend.

- **Acquista** quando il close rompe il **massimo degli ultimi 55 giorni** con volume
  in espansione (> 1.2x media) e prezzo sopra SMA50.
- **Esce** sotto il **minimo degli ultimi 20 giorni** (uscita Donchian classica)
  o su trailing stop 3x ATR dopo il +5%.
- **Pochissimi trade**: entra solo sui breakout più forti e resta in posizione per mesi.
  Storica dei Turtle: cattura 2-3 grandi trend all'anno e li cavalca fino in fondo.
"""
    min_entry_score = 60
    defaults = {
        'MAX_POSITIONS': 3, 'REBALANCE_DAYS': 5, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 4, 'STOP_LOSS_ATR_MULT': 3.0, 'TIME_STOP_DAYS': 90,
    }

    def prepare(self, df, spy_df=None, cfg=None):
        df = df.copy()
        df['HH_55'] = df['High'].rolling(55).max().shift(1)
        df['LL_20'] = df['Low'].rolling(20).min().shift(1)
        vol_sma = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / vol_sma.replace(0, np.nan)
        return df

    def entry(self, row, ctx):
        c = row['Close']
        hh, sma50, vr = row.get('HH_55'), row.get('SMA50'), row.get('Vol_Ratio')
        if any(v is None or pd.isna(v) for v in (hh, sma50, vr)):
            return 0, "Dati insufficienti"
        if c <= sma50:
            return 0, "Sotto SMA50"
        if c <= hh:
            return 0, "Nessun breakout 55gg"
        if vr < 1.2:
            return 0, f"Volume debole (x{vr:.1f})"

        rs = row.get('RS_Score', 0.0)
        score = 55 + min(35.0, (c / hh - 1) * 300.0) + float(np.clip(rs, -15, 20)) \
                + min(10.0, (vr - 1.2) * 20.0)
        return score, f"Breakout 55gg {hh:.2f} | Vol x{vr:.1f}"

    def exit_signal(self, pos, row, ctx):
        ll = row.get('LL_20')
        if ll is not None and pd.notna(ll) and row['Close'] < ll:
            return ("DONCHIAN EXIT (LL20)", 1.0, row['Close'])
        days_held = (ctx.current_date - pos['entry_date']).days
        roi = (row['Close'] / pos['entry_price']) - 1
        if days_held >= ctx.cfg['TIME_STOP_DAYS'] and roi < 0.02:
            return ("TIME STOP", 1.0, row['Close'])
        return None

    def update_position(self, pos, row, ctx):
        c = row['Close']
        roi = (c / pos['entry_price']) - 1
        if roi > 0.05:
            atr = row.get('ATR')
            if atr is not None and pd.notna(atr):
                new_stop = c - atr * ctx.cfg['STOP_LOSS_ATR_MULT']
                if pos.get('stop_loss') is None or new_stop > pos['stop_loss']:
                    pos['stop_loss'] = new_stop


# ---------------------------------------------------------------------
# REGISTRO STRATEGIE
# ---------------------------------------------------------------------
STRATEGIES: Dict[str, BaseStrategy] = {
    s.key: s for s in [
        AIMomentumStrategy(),
        DualMomentumStrategy(),
        BearMarketStrategy(),
        VolumeProfileStrategy(),
        OrderFlowStrategy(),
        MeanReversionStrategy(),
        TurtleBreakoutStrategy(),
    ]
}

STRATEGY_ORDER: List[str] = [
    "ai_momentum", "dual_momentum", "bear_market",
    "volume_profile", "orderflow", "mean_reversion", "turtle_breakout",
]


def get_strategy(key: str) -> BaseStrategy:
    strat = STRATEGIES.get(key)
    if strat is None:
        raise ValueError(f"Strategia sconosciuta: '{key}'. Disponibili: {list(STRATEGIES)}")
    return strat


# =====================================================================
# 4. MOTORE DI BACKTEST (gestisce portafoglio/commissioni/tasse per tutte)
# =====================================================================

class _FastRowPanel:
    """Accesso rapido alle righe per data di simulazione.

    Nel loop giorno-per-giorno, `df.loc[data]` costa ~60-100µs per chiamata:
    con universi grandi (tutto Xetra) e ribilanciamenti ogni 5 giorni sono
    centinaia di migliaia di accessi che dominano il tempo di simulazione.
    Questo pannello preindicizza le date una sola volta (get_indexer) e
    costruisce le righe da array numpy (~5x più veloce), mantenendo la stessa
    API delle strategie (pd.Series indicizzata per nome colonna).
    """
    __slots__ = ('_entries',)

    def __init__(self, market_data: Dict[str, pd.DataFrame], sim_dates: pd.DatetimeIndex):
        self._entries = {}
        for t, df in market_data.items():
            if df is None or df.empty:
                continue
            pos = df.index.get_indexer(sim_dates)
            arr = df.to_numpy()
            close_col = df.columns.get_loc('Close') if 'Close' in df.columns else None
            self._entries[t] = (df.columns, arr, pos, close_col)

    def row(self, ticker: str, i: int):
        """La riga del ticker alla i-esima data di simulazione, None se assente."""
        e = self._entries.get(ticker)
        if e is None:
            return None
        cols, arr, pos, _ = e
        p = pos[i]
        if p < 0:
            return None
        return pd.Series(arr[p], index=cols)

    def close(self, ticker: str, i: int):
        """Il prezzo Close del ticker alla i-esima data (None se non disponibile)."""
        e = self._entries.get(ticker)
        if e is None:
            return None
        _, arr, pos, cc = e
        p = pos[i]
        if p < 0 or cc is None:
            return None
        v = arr[p, cc]
        return float(v) if np.isfinite(v) else None


# Cache in-processo dei frame preparati dalle strategie: i re-run (es. cambio
# commissioni in UI) non ricalcolano gli indicatori. Budget in righe per
# limitare la memoria (configurabile via env).
_PREPARED_CACHE: Dict[Tuple[str, str], Tuple[str, pd.DataFrame]] = {}
_PREPARED_CACHE_ROWS = 0
_PREPARED_CACHE_MAX_ROWS = int(os.environ.get('ETF_METRICS_PREPARED_CACHE_ROWS', 6_000_000))


def _frame_fingerprint(df: pd.DataFrame) -> str:
    """Identificativo economico del contenuto di un frame (per invalidare la cache)."""
    try:
        return (f"{len(df)}|{df.index[0]}|{df.index[-1]}|"
                f"{df['Close'].iloc[0]:.6f}|{df['Close'].iloc[-1]:.6f}")
    except Exception:
        return f"{id(df)}|{len(df)}"


def _prepare_strategy_frames(strat, market_data: Dict[str, pd.DataFrame],
                             spy_df, cfg: Dict) -> Dict[str, pd.DataFrame]:
    """Applica `strat.prepare` a ogni frame con cache in-processo.

    Le strategie aggiungono colonne a copie dei frame base (mai mutate): la
    cache restituisce lo stesso oggetto preparato finché dati e parametri non
    cambiano (fingerprint su lunghezza, estremi dell'indice e prezzi).
    """
    global _PREPARED_CACHE_ROWS
    try:
        params_key = json.dumps(cfg, default=str, sort_keys=True)
    except Exception:
        params_key = str(sorted((str(k), str(v)) for k, v in cfg.items()))
    spy_fp = _frame_fingerprint(spy_df) if spy_df is not None else '-'
    prefix = f"{strat.key}|{params_key}|{spy_fp}"

    out: Dict[str, pd.DataFrame] = {}
    for t, df in market_data.items():
        if df is None or df.empty:
            continue
        fp = _frame_fingerprint(df)
        want = f"{prefix}|{fp}"
        entry = _PREPARED_CACHE.get((strat.key, t))
        if entry is not None and entry[0] == want:
            out[t] = entry[1]
            continue
        prepared = strat.prepare(df, spy_df, cfg)
        _PREPARED_CACHE[(strat.key, t)] = (want, prepared)
        _PREPARED_CACHE_ROWS += len(prepared)
        out[t] = prepared

    if _PREPARED_CACHE_ROWS > _PREPARED_CACHE_MAX_ROWS:
        _PREPARED_CACHE.clear()
        _PREPARED_CACHE_ROWS = 0
    return out


def run_market_aware_backtest(
    tickers: list,
    start_date="2015-01-01",
    initial_capital=10000,
    preloaded_data=None,
    tax_rate=26.0,
    allow_fractional=True,
    progress_callback: Optional[Callable[[float], None]] = None,
    strategy: str = "ai_momentum",
    strategy_params: Optional[Dict] = None,
):
    """
    Esegue il backtest della strategia selezionata.
    `progress_callback` (opzionale) riceve la frazione di avanzamento [0,1];
    è onere del chiamante (UI) gestire il ciclo di vita della barra di progresso.
    """
    strat = get_strategy(strategy)
    cfg = dict(CONFIG)
    cfg.update(strat.defaults)
    if strategy_params:
        cfg.update(strategy_params)
    benchmark = cfg['SPY_TICKER']

    if preloaded_data:
        market_data, _bench_key = _ensure_processed(
            {t: df for t, df in preloaded_data.items() if df is not None and not df.empty},
            benchmark,
        )
    else:
        market_data = prepare_market_data(tickers)
        if market_data:
            market_data, _bench_key = _ensure_processed(market_data, benchmark)  # no-op se già pronti

    if not market_data:
        return pd.DataFrame(), initial_capital

    # Benchmark effettivo: SPY se disponibile, altrimenti la proxy equal-weight
    # dell'universo aggiunta da _ensure_processed (universi Xetra senza USA)
    if benchmark not in market_data and BENCH_PROXY_KEY in market_data:
        benchmark = BENCH_PROXY_KEY

    # Indicatori specifici della strategia (con cache: i re-run non ricalcolano)
    spy_df = market_data.get(benchmark)
    market_data = _prepare_strategy_frames(strat, market_data, spy_df, cfg)

    ref = benchmark if benchmark in market_data else list(market_data.keys())[0]
    sim_dates = market_data[ref].index[market_data[ref].index >= pd.to_datetime(start_date)]
    price_matrix = pd.DataFrame({t: d['Close'] for t, d in market_data.items()}).ffill()

    # Regime di mercato precalcolato in una passata vettoriale (con universi
    # grandi il calcolo per-giorno dominava il tempo di simulazione)
    regime_series = _precompute_regimes(market_data.get(benchmark), sim_dates) \
        if benchmark in market_data else None

    # Accesso rapido alle righe per data (evita df.loc[data] nel loop)
    panel = _FastRowPanel(market_data, sim_dates)

    cash = float(initial_capital)
    positions = {}
    trade_log = []
    tax_credit = 0.0
    current_sim_month = -1
    buys_this_month = 0
    commission = float(cfg['COMMISSION'])

    ctx = _MarketCtx()
    ctx.cfg = cfg
    ctx.benchmark = benchmark
    ctx.market_data = market_data
    ctx.price_matrix = price_matrix

    # --- HELPER PER CALCOLO NAV (FIX DRAWDOWN) ---
    def _get_current_nav(curr_cash, curr_positions, i):
        equity = curr_cash
        for p_ticker, p_data in curr_positions.items():
            curr_p = panel.close(p_ticker, i)
            if curr_p is None:
                curr_p = p_data['entry_price']
            equity += p_data['qty'] * curr_p
        return equity

    for i, current_date in enumerate(sim_dates):
        if progress_callback and i % 50 == 0:
            progress_callback((i + 1) / len(sim_dates))

        if current_date.month != current_sim_month:
            current_sim_month = current_date.month
            buys_this_month = 0

        regime = regime_series.iloc[i] if regime_series is not None \
            else _assess_market_regime(price_matrix, current_date, market_data, benchmark)
        is_bull = (regime == "BULL")
        is_crash = (regime == "DANGER" or regime == "VOLATILE")

        ctx.regime = regime
        ctx.is_crash = is_crash
        ctx.current_date = current_date

        tokens_to_sell = []

        # 1. GESTIONE USCITE
        for t, pos in positions.items():
            row = panel.row(t, i)
            if row is None:
                continue

            action = None
            # 1a. Stop loss "fisico" (gestito dal motore se la strategia usa gli stop)
            if strat.use_stops and pos.get('stop_loss') is not None \
                    and pd.notna(pos['stop_loss']) and row['Low'] < pos['stop_loss']:
                exit_price = max(row['Open'], pos['stop_loss'])
                action = ("STOP LOSS", 1.0, exit_price)
            else:
                # 1b. Regole di uscita specifiche della strategia
                try:
                    sig = strat.exit_signal(pos, row, ctx)
                except Exception:
                    sig = None
                if sig is not None:
                    reason, portion = sig[0], sig[1]
                    price = sig[2] if len(sig) > 2 and sig[2] is not None else row['Close']
                    action = (reason, portion, price)
                else:
                    # 1c. Manutenzione posizione (trailing stop, flag TP, ecc.)
                    try:
                        strat.update_position(pos, row, ctx)
                    except Exception:
                        pass

            if action is not None:
                tokens_to_sell.append((t, action[2], action[0], action[1]))

        # ESECUZIONE VENDITE
        for t, price, reason, portion in tokens_to_sell:
            pos = positions[t]
            qty_sell = pos['qty'] * portion
            if not allow_fractional:
                qty_sell = int(qty_sell)
                if qty_sell == 0 and portion > 0.9:
                    qty_sell = pos['qty']

            if qty_sell <= 0:
                continue

            net = (qty_sell * price) - commission
            cost_portion = pos['cost_basis'] * (qty_sell / pos['qty'])
            gain = net - cost_portion

            tax = 0.0
            if gain > 0:
                taxable = max(0, gain - tax_credit)
                tax = taxable * (tax_rate / 100.0)
                tax_credit = max(0, tax_credit - gain)
            else:
                tax_credit += abs(gain)

            cash += (net - tax)

            # Aggiorna posizione
            if portion >= 0.99 or (pos['qty'] - qty_sell) < (0.001 if allow_fractional else 1):
                del positions[t]
            else:
                positions[t]['qty'] -= qty_sell
                positions[t]['cost_basis'] -= cost_portion

            # LOGGING: Total_Equity invece di Cash
            current_nav = _get_current_nav(cash, positions, i)
            trade_log.append({
                "Date": current_date.date(), "Ticker": t, "Action": "SELL",
                "Price": price, "Reason": reason,
                "PnL_Net": gain - tax,
                "Capital": cash,
                "Total_Equity": current_nav,
                "Qty": qty_sell,
                "Commission": commission,
                "Strategy": strat.key,
            })

        # 2. GESTIONE INGRESSI
        buy_allowed = strat.can_buy(ctx) and \
            (not cfg['ENABLE_MONTHLY_LIMIT'] or buys_this_month < cfg['MAX_BUYS_PER_MONTH'])

        if buy_allowed and i % cfg['REBALANCE_DAYS'] == 0:
            free_slots = cfg['MAX_POSITIONS'] - len(positions)

            if free_slots > 0 and cash > 50:
                candidates = []
                for t in tickers:
                    if t == benchmark or t in positions:
                        continue
                    row = panel.row(t, i)
                    if row is None:
                        continue
                    try:
                        score, reason = strat.entry(row, ctx)
                    except Exception:
                        continue
                    if score is not None and pd.notna(score) and score >= strat.min_entry_score:
                        candidates.append({'t': t, 'score': float(score), 'row': row, 'reason': reason})

                candidates.sort(key=lambda x: x['score'], reverse=True)

                for cand in candidates:
                    if free_slots <= 0 or cash < 50:
                        break
                    if cfg['ENABLE_MONTHLY_LIMIT'] and buys_this_month >= cfg['MAX_BUYS_PER_MONTH']:
                        break

                    t = cand['t']
                    row = cand['row']

                    is_safe_corr, conflict = check_correlation_strict(t, list(positions.keys()), price_matrix,
                                                                      current_date)
                    if not is_safe_corr:
                        continue

                    try:
                        risk_factor = strat.position_size_factor(row, ctx)
                    except Exception:
                        risk_factor = 1.0
                    alloc_per_slot = (cash / free_slots) * risk_factor * 0.98

                    if allow_fractional:
                        qty = alloc_per_slot / row['Close']
                    else:
                        qty = int(alloc_per_slot / row['Close'])

                    cost = (qty * row['Close']) + commission

                    if qty > 0 and cost <= cash:
                        cash -= cost
                        initial_stop = strat.initial_stop(row, ctx, cfg)

                        positions[t] = {
                            'qty': qty, 'cost_basis': cost, 'entry_price': row['Close'],
                            'entry_date': current_date, 'stop_loss': initial_stop,
                            'tp1_taken': False, 'regime_at_entry': regime,
                        }

                        current_nav = _get_current_nav(cash, positions, i)
                        trade_log.append({
                            "Date": current_date.date(), "Ticker": t, "Action": "BUY",
                            "Price": row['Close'], "Reason": cand['reason'],
                            "PnL_Net": 0.0,
                            "Capital": cash,
                            "Total_Equity": current_nav,
                            "Qty": qty,
                            "Commission": commission,
                            "Strategy": strat.key,
                        })
                        free_slots -= 1
                        buys_this_month += 1

    # Chiusura Finale (Mark to Market)
    final_nav = cash
    for t, pos in positions.items():
        try:
            p = market_data[t].iloc[-1]['Close']
        except Exception:
            p = pos['entry_price']
        val = pos['qty'] * p
        final_nav += val

        trade_log.append({
            "Date": sim_dates[-1].date(), "Ticker": t, "Action": "HOLD (End)",
            "Price": p, "Reason": "Portfolio Value",
            "PnL_Net": val - pos['cost_basis'],
            "Capital": cash,
            "Total_Equity": final_nav,
            "Qty": pos['qty'],
            "Commission": 0.0,
            "Strategy": strat.key,
        })

    return pd.DataFrame(trade_log), final_nav


# =====================================================================
# 5. CONFRONTO STRATEGIE + BENCHMARK BUY & HOLD
# =====================================================================

def _default_comparison_workers() -> int:
    """Worker massimi per il confronto strategie in parallelo.

    Default: min(strategie, cpu, 4) su Linux/macOS (fork); 1 altrove.
    Configurabile con la variabile d'ambiente ETF_METRICS_BACKTEST_WORKERS.
    Il tetto di default limita anche la memoria (ogni worker prepara in locale
    le proprie copie degli indicatori della strategia).
    """
    raw = (os.environ.get('ETF_METRICS_BACKTEST_WORKERS') or '').strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    if not hasattr(os, 'fork'):
        return 1
    return max(1, min(len(STRATEGIES), os.cpu_count() or 1, 4))


# Dati condivisi con i worker del confronto parallelo: con il contesto "fork"
# i processi figli ereditano la memoria del genitore (copy-on-write), senza
# pickle di gigabyte di DataFrame.
_FORK_COMPARISON_DATA = None


def _comparison_worker(strategy_key, tickers, start_date, initial_capital,
                       tax_rate, allow_fractional, strategy_params):
    """Esegue una strategia nel processo worker (usa i dati ereditati via fork)."""
    df_trades, final = run_market_aware_backtest(
        tickers,
        start_date=start_date,
        initial_capital=initial_capital,
        preloaded_data=_FORK_COMPARISON_DATA,
        tax_rate=tax_rate,
        allow_fractional=allow_fractional,
        strategy=strategy_key,
        strategy_params=strategy_params,
    )
    return strategy_key, df_trades, final


def run_strategy_comparison(
    tickers: list,
    strategies: Optional[List[str]] = None,
    start_date="2015-01-01",
    initial_capital=10000,
    preloaded_data=None,
    tax_rate=26.0,
    allow_fractional=True,
    progress_callback: Optional[Callable[[float], None]] = None,
    strategy_params: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Esegue più strategie sugli stessi dati (confronto fair).
    Ritorna {strategy_key: {'trades': DataFrame, 'final': float, 'name': str}}.

    Su Linux/macOS le strategie girano in processi separati (fork): su una CPU
    moderna con 8+ core il confronto completo è ~3-4x più veloce. Ogni worker
    fallito viene automaticamente rieseguito in seriale nel processo padre.
    """
    keys = [k for k in (strategies or STRATEGY_ORDER) if k in STRATEGIES]
    results: Dict[str, Dict] = {}
    n = len(keys)

    def _store(key, df_trades, final):
        results[key] = {
            "trades": df_trades,
            "final": final,
            "name": STRATEGIES[key].name,
        }

    def _run_serial(key):
        df_trades, final = run_market_aware_backtest(
            tickers,
            start_date=start_date,
            initial_capital=initial_capital,
            preloaded_data=shared,
            tax_rate=tax_rate,
            allow_fractional=allow_fractional,
            strategy=key,
            strategy_params=strategy_params,
        )
        _store(key, df_trades, final)

    # I dati vengono caricati/preparati UNA sola volta per tutte le strategie
    # (fair comparison + un solo accesso al DB invece di uno per strategia)
    shared = preloaded_data
    if shared is None and keys:
        loaded = prepare_market_data(tickers)
        if loaded:
            shared = loaded

    workers = min(_default_comparison_workers(), n)
    use_parallel = (workers > 1 and shared is not None and hasattr(os, 'fork'))

    if use_parallel:
        global _FORK_COMPARISON_DATA
        _FORK_COMPARISON_DATA = shared
        try:
            ctx = multiprocessing.get_context('fork')
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futures = {ex.submit(
                    _comparison_worker, key, tickers, start_date, initial_capital,
                    tax_rate, allow_fractional, strategy_params): key for key in keys}
                done = 0
                for fut in as_completed(futures):
                    key = futures[fut]
                    try:
                        _, df_trades, final = fut.result()
                        _store(key, df_trades, final)
                    except Exception as e:
                        # Worker morto (es. OOM): la strategia viene rieseguita qui
                        logger.warning(f"Worker della strategia {key} fallito ({e}): "
                                       "esecuzione seriale di fallback")
                        _run_serial(key)
                    done += 1
                    if progress_callback:
                        try:
                            progress_callback(done / n)
                        except Exception:
                            pass
        finally:
            _FORK_COMPARISON_DATA = None
    else:
        for j, key in enumerate(keys):
            if progress_callback:
                progress_callback(j / max(1, n))
            _run_serial(key)

    # preserva l'ordine richiesto (in parallelo le strategie completano in
    # ordine sparso)
    results = {k: results[k] for k in keys if k in results}
    if progress_callback:
        progress_callback(1.0)
    return results


def compute_buy_and_hold_curves(
    market_data: Dict[str, pd.DataFrame],
    tickers: List[str],
    start_date,
    initial_capital: float,
    benchmark_ticker: str = "SPY",
) -> Dict[str, pd.Series]:
    """
    Curve equity buy & hold di riferimento:
      - 'benchmark': tutto il capitale sul benchmark (S&P500 proxy)
      - 'equal_weight': capitale suddiviso in parti uguali sull'universo
    """
    start = pd.to_datetime(start_date)
    out: Dict[str, pd.Series] = {}

    if benchmark_ticker in market_data:
        s = market_data[benchmark_ticker]['Close']
        s = s[s.index >= start].dropna()
        if len(s) > 1:
            out['benchmark'] = initial_capital * (s / s.iloc[0])
    elif BENCH_PROXY_KEY in market_data:
        # senza SPY il benchmark è la proxy equal-weight dell'universo: chiave
        # separata, così la UI può etichettarla correttamente
        s = market_data[BENCH_PROXY_KEY]['Close']
        s = s[s.index >= start].dropna()
        if len(s) > 1:
            out['benchmark_proxy'] = initial_capital * (s / s.iloc[0])

    cols = [t for t in tickers if t in market_data and t != benchmark_ticker]
    if len(cols) >= 2:
        pm = pd.DataFrame({t: market_data[t]['Close'] for t in cols}).ffill()
        pm = pm[pm.index >= start].dropna()
        if len(pm) > 1:
            rets = pm.pct_change().fillna(0).mean(axis=1)
            out['equal_weight'] = initial_capital * (1 + rets).cumprod()

    return out


def align_equity_curves(curves: Dict[str, pd.Series]) -> pd.DataFrame:
    """Allinea curve equity con indici diversi in un unico DataFrame (ffill + bfill)."""
    df = pd.DataFrame(curves).sort_index()
    df = df.ffill().bfill()
    return df
