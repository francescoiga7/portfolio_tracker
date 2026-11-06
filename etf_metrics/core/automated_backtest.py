# -*- coding: utf-8 -*-
"""Backtest algoritmico multi-strategia (Xetra / ETF / azioni).

Mappa del modulo:
  1. CONFIG: default del motore, sovrascrivibili per strategia (attributo
     `defaults`) e dai parametri della run.
  2. Igiene dei dati e indicatori: `_sanitize_ohlcv`,
     `calculate_advanced_metrics_vectorized`, `_assess_market_regime`,
     `_precompute_regimes`, `build_benchmark_proxy` (proxy equal-weight
     dell'universo, chiave riservata `BENCH_PROXY_KEY` che non collide con
     ticker reali).
  3. Caricamento: `prepare_market_data`, `load_recent_market_data`,
     `_ensure_processed`.
  4. Framework strategie: `BaseStrategy` e le strategie attive nel registro
     `STRATEGIES`; `STRATEGY_ORDER` e' l'ordine di presentazione in UI. Le
     strategie ritirate vivono in `retired_strategies.py` (riattivabili con
     `activate()`); cronaca completa in `docs/STRATEGY_LEDGER.md`.
  5. Motore: `run_market_aware_backtest` (portafoglio/commissioni/tasse per
     tutte le strategie) con `_FastRowPanel` e la cache dei frame preparati.
  6. Confronto: `run_strategy_comparison` (piu' strategie sugli stessi dati,
     parallelo via fork con dati condivisi in `_FORK_COMPARISON_DATA`) e
     `compute_buy_and_hold_curves` (benchmark buy & hold).

Interfaccia strategia (`BaseStrategy`):
  - prepare(df, spy_df, cfg): aggiunge colonne/indicatori specifici
  - can_buy(ctx):   il regime di mercato ammette acquisti?
  - entry(row, ctx): punteggia i candidati -> (score, reason)
  - exit_signal(pos, row, ctx): regole di uscita -> None | (reason, portion, price)
  - update_position(pos, row, ctx): manutenzione (trailing stop, flag)

Il motore `run_market_aware_backtest` gestisce portafoglio, commissioni,
tasse, limiti mensili e filtro correlazione in modo identico per tutte le
strategie (confronto fair). Ogni ingresso passa un gate di liquidita'
(prezzo/volume/ADV) e il NAV e' mark-to-market a ogni seduta: drawdown e
Sharpe restano onesti anche nei giorni senza trade.

Chiavi principali del CONFIG:
  - posizioni/acquisti: MAX_POSITIONS, REBALANCE_DAYS, ENABLE_MONTHLY_LIMIT,
    MAX_BUYS_PER_MONTH, COMMISSION;
  - regime macro: SPY_TICKER (benchmark), MACRO_TREND_FILTER,
    BEAR_VOLATILITY_THRESHOLD (ATR% oltre cui il regime e' VOLATILE);
  - qualita' degli ingressi: MIN_RS_SCORE, MIN_ADX, MIN_PROX_HIGH,
    STOP_LOSS_ATR_MULT, TIME_STOP_DAYS, TP1_PCT, MAX_CORRELATION;
  - gate di liquidita' degli ingressi: un ordine su un titolo che nessuno
    scambia non e' eseguibile e gonfia i risultati in modo irripetibile.
    ENABLE_LIQUIDITY_GATE attiva i vincoli MIN_ENTRY_PRICE (EUR minimi di
    prezzo per l'ingresso), MIN_ENTRY_VOLUME (pezzi scambiati il giorno
    dell'ordine) e MAX_POSITION_VS_ADV (posizione <= 20% del turnover medio
    20gg in EUR).
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
from etf_metrics.core.tax_engine import ItalianTaxLedger
from etf_metrics.core.trading import atr_multiplier_from_vol

logger = logging.getLogger(__name__)

CONFIG = {
    'MAX_POSITIONS': 3,
    'REBALANCE_DAYS': 5,
    'ENABLE_MONTHLY_LIMIT': True,
    'MAX_BUYS_PER_MONTH': 4,
    'COMMISSION': 1.0,
    'SPY_TICKER': 'SPY',
    'MACRO_TREND_FILTER': True,
    'BEAR_VOLATILITY_THRESHOLD': 3.0,
    'MIN_RS_SCORE': 80,
    'MIN_ADX': 25,
    'MIN_PROX_HIGH': 0.85,
    'STOP_LOSS_ATR_MULT': 2.5,
    'TIME_STOP_DAYS': 21,
    'TP1_PCT': 0.12,
    'MAX_CORRELATION': 0.65,
    'ENABLE_LIQUIDITY_GATE': True,
    'MIN_ENTRY_PRICE': 1.0,
    'MIN_ENTRY_VOLUME': 500,
    'MAX_POSITION_VS_ADV': 0.20,
}


_PRICE_BREAK_LOG = float(np.log(3.0))


def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ripulisce una serie OHLCV da errori di livello prezzo.

    Le serie con stub a prezzo segnaposto, tick corrotti o reverse split non
    aggiustati generano rendimenti falsi di migliaia di punti. Regole (su
    principi di mercato, NON sulle performance):
      1) righe con Close non finito o <= 0 -> eliminate;
      2) |log-rend| tra due righe > log(3) (prezzo che triplica/crolla a un
         terzo in un passaggio): errore di livello -> la serie viene troncata
         all'ULTIMA rottura (vale il blocco piu' recente). Vale per QUALUNQUE
         gap di calendario: i salti attraverso sospensioni lunghe sono
         reverse split non aggiustati, non riaperture genuine
         (3TSE.DE +3.605%/493gg).
    Dettagli e motivazioni: docs/STRATEGY_LEDGER.md.
    Il DB NON viene modificato (la pulizia avviene al caricamento: backtest,
    scanner e live vedono gli stessi dati). Idempotente.
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return df
    out = df
    close = pd.to_numeric(out['Close'], errors='coerce')
    ok = np.isfinite(close.to_numpy(dtype=float)) & (close.to_numpy(dtype=float) > 0)
    if not ok.all():
        out = out[ok]
    if len(out) < 2:
        return out
    c = out['Close'].to_numpy(dtype=float)
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        return out
    lr = np.log(c[1:] / c[:-1])
    breaks = np.isfinite(lr) & (np.abs(lr) > _PRICE_BREAK_LOG)
    if breaks.any():
        out = out.iloc[int(np.nonzero(breaks)[0][-1]) + 1:]
    return out


def calculate_advanced_metrics_vectorized(df, spy_df=None):
    """Aggiunge al frame gli indicatori base usati da regime e strategie:
    SMA50/SMA200, ATR14, Vol_20 (volatilita' 20gg in %), ADV_EUR_20, RSI14,
    ADX14, High_52w/Prox_High e RS_Score (rendimento relativo 63gg vs
    benchmark, solo con spy_df e almeno 65 sedute comuni).

    ADV_EUR_20 e' il turnover medio in EUR (mediana 20gg, CAUSALE: shift(1) —
    a fine giornata si conosce la storia fino a ieri) ed alimenta il gate di
    liquidita': posizione <= MAX_POSITION_VS_ADV * ADV.
    """
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
    if 'Volume' in df.columns:
        _dv = df['Close'] * pd.to_numeric(df['Volume'], errors='coerce')
        df['ADV_EUR_20'] = _dv.rolling(20).median().shift(1)
    else:
        df['ADV_EUR_20'] = np.nan

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
    """Regime di mercato alla data (BULL / BEAR / VOLATILE / DANGER / NEUTRAL).

    Con benchmark: Close vs SMA200, ATR%, RSI; lookup as-of sull'ultima
    seduta disponibile (calendari USA != Xetra). Senza benchmark: proxy
    equal-weight dell'universo sugli ultimi 200 giorni di TRADING (non
    solari: 200 solari ≈ 142 sedute, la soglia len >= 200 non sarebbe mai
    raggiungibile e il regime resterebbe per sempre NEUTRAL).
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
    """Regime per ogni data di simulazione in un'unica passata vettoriale.

    Replica il ramo benchmark di `_assess_market_regime` (il calcolo
    per-giorno domina il tempo con universi grandi). La precedenza e' la
    stessa dell'if-chain originale (BEAR > VOLATILE > DANGER > BULL): con le
    maschere vettoriali le categorie vengono applicate in ordine INVERSO,
    perche' l'ultima scrittura vince. L'allineamento alle date di simulazione
    e' as-of (reindex + ffill: calendari di mercato diversi). Ritorna una
    Series allineata a sim_dates, oppure None se il benchmark non ha gli
    indicatori (il chiamante ricade nel calcolo per-giorno).
    """
    needed = {'Close', 'SMA200', 'ATR', 'RSI'}
    if bench_df is None or bench_df.empty or not needed.issubset(bench_df.columns):
        return None
    try:
        close, sma = bench_df['Close'], bench_df['SMA200']
        atr_pct = bench_df['ATR'] / close * 100.0
        rsi = bench_df['RSI']
        regime = pd.Series("BULL", index=bench_df.index, dtype=object)
        regime[rsi < 35] = "DANGER"
        regime[atr_pct > CONFIG['BEAR_VOLATILITY_THRESHOLD']] = "VOLATILE"
        regime[close < sma] = "BEAR"
        return regime.reindex(sim_dates, method='ffill').fillna("NEUTRAL")
    except Exception as e:
        logger.debug(f"Precompute regimi non riuscito ({e}): fallback per-giorno")
        return None


BENCH_PROXY_KEY = '__UNIVERSE_BENCH__'


def build_benchmark_proxy(market_data: Dict[str, pd.DataFrame],
                          key: str = BENCH_PROXY_KEY) -> Optional[pd.DataFrame]:
    """Benchmark equal-weight dell'universo (fallback senza SPY).

    Indice a ribilanciamento giornaliero: media dei rendimenti % giornalieri
    dei titoli quotati quel giorno (skipna: chi non e' quotato non pesa),
    composta in un livello. NON normalizzare i livelli e mediarli: peserebbe
    ogni titolo per la sua performance cumulata (paniere sovrappesato sui
    vincitori).

    I prezzi OHLC sintetici sono costruiti dai rapporti Open/High/Low
    rispetto al PROPRIO close di ciascun titolo (h >= max(o, 1),
    l <= min(o, 1)): le medie dei rapporti preservano l'ordine OHLC; un
    passaggio finale belt & braces (High = max(High, Open, Close),
    Low = min(Low, Open, Close)) lo garantisce anche con NaN indipendenti.
    Ritorna OHLCV validi per indicatori di regime e RS; None con meno di
    2 titoli.
    """
    frames = [df for t, df in market_data.items()
              if df is not None and not df.empty and t != key]
    if len(frames) < 2:
        return None

    idx = frames[0].index
    for df in frames[1:]:
        idx = idx.union(df.index)

    crets, orats, hrats, lrats = {}, {}, {}, {}
    for i, df in enumerate(frames):
        c = df['Close']
        r = c.pct_change()
        crets[i] = r.replace([np.inf, -np.inf], np.nan).reindex(idx)
        if {'Open', 'High', 'Low'} <= set(df.columns):
            cc = c.where(np.isfinite(c) & (c > 0))
            orats[i] = (df['Open'] / cc).reindex(idx)
            hrats[i] = (df['High'] / cc).reindex(idx)
            lrats[i] = (df['Low'] / cc).reindex(idx)

    mean_cr = pd.DataFrame(crets).mean(axis=1)
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
    proxy['High'] = proxy[['High', 'Open', 'Close']].max(axis=1)
    proxy['Low'] = proxy[['Low', 'Open', 'Close']].min(axis=1)
    proxy.index.name = 'date'
    return proxy


def check_correlation_strict(candidate, portfolio, price_matrix, current_date):
    """Veto di correlazione: il candidato non e' acquistabile se la sua
    correlazione (finestra 60gg, minimo 30 sedute) con una posizione gia' in
    portafoglio supera MAX_CORRELATION. Ritorna (False, ticker_in_conflitto)
    o (True, None); senza dati sufficienti il veto non scatta.
    """
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
    """Punteggio 'AI Score' degli ingressi momentum: base 50 + RS vs
    benchmark (x2) + ADX/2, bonus prossimita' ai massimi (> 0.95), penalita'
    volatilita' (Vol_20 > 5%) e RSI ipercomprato (> 85). Esclude subito chi
    e' sotto SMA50, lontano dai massimi o in trend debole. Ritorna
    (score, reason).
    """
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
    """Carica i dati per il backtest dal DB locale (+ benchmark SPY).

    Con allow_download=True aggiorna con logica DELTA (download completo
    solo per i ticker nuovi, incrementale per gli altri; la prima volta con
    DB vuoto il download avviene qui); se l'aggiornamento delta fallisce
    (es. rate limit o updater non disponibile) cade su un bulk completo per
    i soli ticker non aggiornati oggi. progress_callback(done, total, label)
    opzionale per la UI. Ritorna {ticker: DataFrame} con indicatori base e
    benchmark di riserva gia' pronti (vedi _ensure_processed).
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
            missing = db_manager.get_tickers_needing_update(all_tickers)
            if missing:
                new_data = get_series_bulk(missing, period=period,
                                           progress_callback=progress_callback)
                if new_data:
                    db_manager.save_bulk_data(new_data)

    loaded = db_manager.load_data(all_tickers)
    processed, _bench_key = _ensure_processed(loaded, CONFIG['SPY_TICKER'])
    return processed


def load_recent_market_data(tickers, period="2y", allow_download=True, min_rows=135):
    """Dati recenti per Scanner e Portafoglio Live, letti PRIMA dal DB locale.

    Aggiorna via delta solo i ticker mancanti o non aggiornati oggi; con
    rate limit Yahoo (429) attivo usa comunque il DB locale. Ritorna
    {ticker: DataFrame OHLCV} SENZA indicatori (li calcola il chiamante);
    min_rows e' il minimo di righe per accettare un ticker (default 135:
    ~6 mesi di sedute).
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
    return {t: _sanitize_ohlcv(df) for t, df in loaded.items()
            if df is not None and not df.empty and len(df) >= min_rows}


def _ensure_processed(market_data: Dict[str, pd.DataFrame], benchmark: str,
                      min_rows: int = 200) -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
    """Sanifica le serie e garantisce gli indicatori base (SMA200 ecc.).

    Non muta l'input del chiamante (lavora su una copia del dict).
    L'igiene e' applicata solo ai frame SENZA indicatori: i gia' processati
    lo sono, e la sanificazione e' comunque idempotente. Ritorna
    (dati_processati, chiave_benchmark_effettiva): se il benchmark richiesto
    non e' disponibile viene costruita e aggiunta la proxy equal-weight
    (BENCH_PROXY_KEY), cosi' regimi, RS e curve buy&hold restano utilizzabili
    anche con un DB senza il benchmark ufficiale.
    """
    market_data = dict(market_data)

    cleaned: Dict[str, pd.DataFrame] = {}
    for t, df in market_data.items():
        if df is None or df.empty:
            continue
        if 'SMA200' in df.columns:
            cleaned[t] = df
        else:
            cdf = _sanitize_ohlcv(df)
            if cdf is not None and not cdf.empty:
                cleaned[t] = cdf
    market_data = cleaned

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
    """Interfaccia comune di tutte le strategie di trading.

    Attributi di classe:
      - min_entry_score: soglia del punteggio di ingresso;
      - requires_bull: se True (default) gli acquisti sono ammessi solo in
        regime BULL; le strategie che comprano anche in BEAR lo sovrascrivono
        con un can_buy dedicato;
      - use_stops: se True il motore mantiene lo stop ATR "fisico"
        (STOP_LOSS_ATR_MULT x ATR sotto il close di ingresso) oltre agli
        stop gestiti dalla strategia in update_position;
      - defaults: override selettivo del CONFIG del motore.
    """
    key = "base"
    name = "Base"
    description = ""
    min_entry_score = 60
    requires_bull = True
    use_stops = True

    defaults: Dict = {}

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

    def exit_outlook(self, pos: Dict, row, ctx: _MarketCtx) -> str:
        """Riepilogo leggibile delle condizioni di uscita ATTUALI di una
        posizione. Informativo e non decisionale: serve solo al Portafoglio
        Live, mai al motore. Righe '•' separate da newline; '' se niente da dire.
        """
        stop = pos.get('stop_loss')
        c = row['Close']
        if stop is not None and pd.notna(stop):
            return (f"• Stop: €{stop:.2f} ({stop / c - 1:+.1%} dal prezzo attuale) "
                    "— vende se il prezzo scende sotto")
        return ""


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

    def exit_outlook(self, pos, row, ctx):
        c = row['Close']
        lines = []
        stop = pos.get('stop_loss')
        if stop is not None and pd.notna(stop):
            tag = "breakeven (post-TP1)" if pos.get('tp1_taken') else "trailing 2.5×ATR"
            lines.append(f"• Stop {tag}: €{stop:.2f} ({stop / c - 1:+.1%}) — vende se il prezzo scende sotto")
        if not pos.get('tp1_taken', False):
            tp1 = pos['entry_price'] * (1 + ctx.cfg['TP1_PCT'])
            lines.append(f"• Take profit parziale: a €{tp1:.2f} (+{ctx.cfg['TP1_PCT']:.0%}) vende il 25% e porta lo stop a breakeven")
        lines.append("• Uscita macro: se il regime entra in crisi con P/L < +5% → vende tutto")
        left = ctx.cfg['TIME_STOP_DAYS'] - (ctx.current_date - pos['entry_date']).days
        quando = f"tra {left} gg " if left > 0 else ""
        lines.append(f"• Time stop ({ctx.cfg['TIME_STOP_DAYS']} gg dall'ingresso): {quando}vende tutto se il P/L resta < +1%")
        return "\n".join(lines)


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

    def exit_outlook(self, pos, row, ctx):
        c = row['Close']
        roi = (c / pos['entry_price']) - 1
        lines = []
        ll = row.get('LL_20')
        if ll is not None and pd.notna(ll):
            lines.append(f"• Uscita Donchian: vende se il Close scende sotto il minimo 20gg (€{ll:.2f}, {ll / c - 1:+.1%})")
        stop = pos.get('stop_loss')
        if stop is not None and pd.notna(stop):
            tag = "trailing 3×ATR" if roi > 0.05 else "iniziale"
            lines.append(f"• Stop {tag}: €{stop:.2f} ({stop / c - 1:+.1%}) — vende se il prezzo scende sotto")
        left = ctx.cfg['TIME_STOP_DAYS'] - (ctx.current_date - pos['entry_date']).days
        quando = f"tra {left} gg " if left > 0 else ""
        lines.append(f"• Time stop ({ctx.cfg['TIME_STOP_DAYS']} gg dall'ingresso): {quando}vende tutto se il P/L resta < +2%")
        return "\n".join(lines)


class TrendFusionStrategy(BaseStrategy):
    key = "trend_fusion"
    name = "🌊 Trend Fusion (Turtle + Bear Switch)"
    description = """
**Trend Fusion** — Turtle Trading in regime normale, gestione difensiva in crisi.

- **Regime `BULL` (tartaruga 🐢):** compra quando il close rompe il **massimo
  degli ultimi 55 giorni** con volume > 1.2x la media e prezzo sopra SMA50
  (entry Turtle pura). Le posizioni vengono gestite con l'uscita **Donchian
  LL20**, che lascia correre i trend, e trailing stop 3x ATR dopo il +5%.
- **Regime `BEAR` (scudo 🐻):** cerca i titoli che **salgono mentre il mercato
  scende** (momentum 63gg positivo, sopra SMA50, ADX solido): ETF inversi,
  oro, obbligazioni, titoli difensivi.
- **Regime `VOLATILE` / `DANGER` / `NEUTRAL`:** nessun nuovo acquisto (cash).
- **Uscite in crisi (BEAR/VOLATILE/DANGER):** le posizioni in perdita o con
  poco profitto (< 5%) vengono chiuse subito (flip di regime), i vincitori
  restano con trailing stop; rottura SMA50 = uscita.
- **Frequenza:** pochissimi trade di qualità, come lo storico dei Turtle.
"""
    min_entry_score = 60
    requires_bull = False
    defaults = {
        'MAX_POSITIONS': 3, 'REBALANCE_DAYS': 5, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 4, 'STOP_LOSS_ATR_MULT': 3.0,
        'TIME_STOP_DAYS': 90,
    }

    def can_buy(self, ctx):
        return ctx.regime in ("BULL", "BEAR")
    def prepare(self, df, spy_df=None, cfg=None):
        df = df.copy()
        df['HH_55'] = df['High'].rolling(55).max().shift(1)
        df['LL_20'] = df['Low'].rolling(20).min().shift(1)
        vol_sma = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / vol_sma.replace(0, np.nan)
        df['MOM_63'] = df['Close'].pct_change(63)
        return df

    def entry(self, row, ctx):
        if ctx.regime == "BULL":
            return self._entry_turtle(row)
        if ctx.regime == "BEAR":
            return self._entry_bear_shield(row)
        return 0, f"Cash (regime {ctx.regime})"

    def _entry_turtle(self, row):
        """Entry Turtle pura: breakout Donchian 55 con conferma di volume."""
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
        score = 55 + min(35.0, (c / hh - 1) * 300.0) \
                + float(np.clip(rs, -15, 20)) + min(10.0, (vr - 1.2) * 20.0)
        return max(0, score), f"TURTLE: breakout 55gg {hh:.2f} | Vol x{vr:.1f} | RS {rs:+.0f}"

    def _entry_bear_shield(self, row):
        """Entry Bear Shield: chi sale mentre il mercato scende."""
        c = row['Close']
        sma50, rs, adx = row.get('SMA50'), row.get('RS_Score', 0.0), row.get('ADX')
        mom = row.get('MOM_63')
        if sma50 is None or pd.isna(sma50):
            return 0, "Dati insufficienti"
        if adx is not None and pd.notna(adx) and adx < 20:
            return 0, "Trend debole (ADX)"
        if c <= sma50:
            return 0, "Sotto SMA50 (bear)"
        if mom is None or pd.isna(mom) or mom <= 0:
            return 0, "Momentum 63gg non positivo"
        score = 55 + (rs if pd.notna(rs) else 0) * 2 + min(15.0, mom * 100.0)
        return max(0, score), f"BEAR SHIELD: Mom63 {mom:+.1%} | RS {rs:+.1f}"

    def exit_signal(self, pos, row, ctx):
        """Uscite a doppio regime.

        In crisi (BEAR/VOLATILE/DANGER) esce subito cio' che NON e'
        difensivo (aperto in BULL e mai partito: P/L < 5%): le posizioni
        scudo aperte in BEAR restano in posizione anche se il regime
        sfarfalla — e' il loro terreno; la rottura della SMA50 chiude anche
        gli scudi. In regime normale la gestione e' Turtle (l'uscita Donchian
        LL20 lascia correre il trend); una posizione scudo che non ha reso
        nel bear esce (FINE BEAR) per liberare capitale per i nuovi breakout.
        """
        c = row['Close']
        roi = (c / pos['entry_price']) - 1
        born_in_bear = pos.get('regime_at_entry') == "BEAR"

        if ctx.regime in ("BEAR", "VOLATILE", "DANGER"):
            if not born_in_bear and roi < 0.05:
                return (f"REGIME FLIP ({ctx.regime})", 1.0, c)
            sma50 = row.get('SMA50')
            if sma50 is not None and pd.notna(sma50) and c < sma50:
                return ("SMA50 ROTTA", 1.0, c)
            return None

        if born_in_bear and roi < 0.05:
            return ("FINE BEAR (flip)", 1.0, c)
        ll = row.get('LL_20')
        if ll is not None and pd.notna(ll) and c < ll:
            return ("DONCHIAN EXIT (LL20)", 1.0, c)
        days_held = (ctx.current_date - pos['entry_date']).days
        if days_held >= ctx.cfg['TIME_STOP_DAYS'] and roi < 0.02:
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

    def exit_outlook(self, pos, row, ctx):
        c = row['Close']
        roi = (c / pos['entry_price']) - 1
        lines = []
        if pos.get('regime_at_entry') == "BEAR":
            lines.append("• Posizione scudo (aperta in BEAR): esce quando il regime torna normale se il P/L resta < +5%")
        else:
            lines.append("• Uscita regime: se passa a BEAR/VOLATILE/DANGER con P/L < +5% → vende tutto")
        ll = row.get('LL_20')
        if ll is not None and pd.notna(ll):
            lines.append(f"• Uscita Donchian: vende se il Close scende sotto il minimo 20gg (€{ll:.2f}, {ll / c - 1:+.1%})")
        stop = pos.get('stop_loss')
        if stop is not None and pd.notna(stop):
            lines.append(f"• Trailing stop 3×ATR: €{stop:.2f} ({stop / c - 1:+.1%}) — vende se il prezzo scende sotto")
        else:
            lines.append(f"• Trailing stop 3×ATR: si attiva al +5% di P/L (ora {roi:+.1%})")
        left = ctx.cfg['TIME_STOP_DAYS'] - (ctx.current_date - pos['entry_date']).days
        quando = f"tra {left} gg " if left > 0 else ""
        lines.append(f"• Time stop ({ctx.cfg['TIME_STOP_DAYS']} gg dall'ingresso): {quando}vende tutto se il P/L resta < +2%")
        return "\n".join(lines)


class GoldenChandelierStrategy(BaseStrategy):
    key = "golden_chandelier"
    name = "🛡️ Golden Chandelier (logica live satellite)"
    description = """
**Replica backtest della logica live del Portafoglio Tracker** (`core/trading.py`,
segnale "V28 Golden Mean" + trailing stop Chandelier sulle posizioni satellite):
prima d'ora questa logica NON era testabile in backtest — ora il confronto è possibile.

- **Ingresso (Golden Mean):** prezzo sopra **SMA 130** e **momentum 6 mesi positivo**
  (nessun filtro di regime, nessun ADX, nessun filtro volume: come la logica live).
- **Uscita trend:** prezzo sotto **SMA 130** = "VENDI (Trend Break)".
- **Uscita stop:** **Chandelier Exit** sul massimo dei Close dall'acquisto meno
  ATR × moltiplicatore **adattivo alla volatilità** — la STESSA scala della logica
  live (`atr_multiplier_from_vol`): vol annua del benchmark < 15% → 3.0x,
  < 25% → 3.5x, < 35% → 4.5x, panico → 6.0x. Nel backtest il VIX è sostituito
  dalla volatilità annualizzata del benchmark calcolata sui dati del DB
  (offline, nessuna chiamata a Yahoo).
- **Niente** take profit parziali, time stop o filtri macro: fedele alla logica live.
- **Attenzione:** senza filtro macro compra anche in bear market — è il costo
  della fedeltà al live; il backtest ora lo misura.
"""
    requires_bull = False
    use_stops = True
    min_entry_score = 55
    defaults = {
        'MAX_POSITIONS': 3, 'REBALANCE_DAYS': 5, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 4, 'TIME_STOP_DAYS': 3650,
    }

    def prepare(self, df, spy_df=None, cfg=None):
        """Colonne della logica live: SMA_130, MOM_6M e BENCH_ANNVOL (volatilita'
        annualizzata del benchmark allineata as-of: proxy del VIX, offline)."""
        df = df.copy()
        df['SMA_130'] = df['Close'].rolling(130).mean()
        df['MOM_6M'] = df['Close'].pct_change(126)
        if spy_df is not None and 'Vol_20' in spy_df.columns:
            bench_vol = (spy_df['Vol_20'] * np.sqrt(252.0)) \
                .reindex(df.index, method='ffill')
            df['BENCH_ANNVOL'] = bench_vol
        else:
            df['BENCH_ANNVOL'] = np.nan
        return df

    def can_buy(self, ctx):
        """Nessun filtro di regime: la logica live non lo prevede."""
        return True

    def initial_stop(self, row, ctx, cfg):
        """Chandelier dal primo giorno: entry - ATR x mult(vol benchmark).
        NON lo stop ATR generico del motore: la logica live non lo prevede."""
        atr = row.get('ATR')
        if atr is None or pd.isna(atr) or atr <= 0:
            return None
        mult = atr_multiplier_from_vol(row.get('BENCH_ANNVOL'))
        return row['Close'] - (atr * mult)

    def entry(self, row, ctx):
        """Fedele alla logica live: score = 55 + min(mom*100, 45), ingresso per
        qualsiasi momentum 6M positivo sopra la SMA130."""
        c, sma, mom = row['Close'], row.get('SMA_130'), row.get('MOM_6M')
        if sma is None or pd.isna(sma) or mom is None or pd.isna(mom):
            return 0, "Dati insufficienti"
        if c <= sma:
            return 0, "Sotto SMA130"
        if mom <= 0:
            return 0, "Momentum 6M negativo"
        score = 55 + min(45.0, mom * 100.0)
        return score, f"GOLDEN MEAN: Mom6M {mom:+.1%} | SMA130 {sma:.2f}"

    def exit_signal(self, pos, row, ctx):
        """'VENDI (Trend Break)': prezzo sotto la SMA130."""
        c, sma = row['Close'], row.get('SMA_130')
        if sma is not None and pd.notna(sma) and c < sma:
            return ("TREND BREAK (SMA130)", 1.0, c)
        return None

    def update_position(self, pos, row, ctx):
        """Chandelier Exit: massimo Close dall'ingresso - ATR x mult(vol
        benchmark). Come nella logica live NON c'e' ratchet: lo stop segue il
        massimo e la volatilita' correnti (puo' allargarsi se l'ATR cresce).
        Il motore lo esegue perche' use_stops e' True."""
        atr = row.get('ATR')
        if atr is None or pd.isna(atr) or atr <= 0:
            return
        pos['max_close'] = max(pos.get('max_close', pos['entry_price']), row['Close'])
        mult = atr_multiplier_from_vol(row.get('BENCH_ANNVOL'))
        pos['stop_loss'] = pos['max_close'] - (atr * mult)

    def exit_outlook(self, pos, row, ctx):
        c = row['Close']
        lines = []
        stop = pos.get('stop_loss')
        if stop is not None and pd.notna(stop):
            atr = row.get('ATR')
            detail = ""
            if atr is not None and pd.notna(atr) and atr > 0:
                mult = atr_multiplier_from_vol(row.get('BENCH_ANNVOL'))
                detail = f" (max Close €{pos.get('max_close', pos['entry_price']):.2f} − {mult:.1f}×ATR)"
            lines.append(f"• Chandelier Exit{detail}: €{stop:.2f} ({stop / c - 1:+.1%}) — vende se il prezzo scende sotto")
            lines.append("  Lo stop segue il massimo Close raggiunto: sale quando il prezzo sale")
        sma = row.get('SMA_130')
        if sma is not None and pd.notna(sma):
            lines.append(f"• Trend break: vende se il Close scende sotto la SMA130 (€{sma:.2f}, {sma / c - 1:+.1%})")
        return "\n".join(lines)


class GoldenChandelierDefensiveStrategy(GoldenChandelierStrategy):
    key = "golden_chandelier_def"
    name = "🚦 Golden Chandelier Difensivo (filtro macro)"
    description = """
**ESPERIMENTO a variabile singola** — la logica live del Portafoglio Tracker
(Golden Chandelier) con in più il **filtro macro** di AI Momentum:

- **Ingresso:** come golden_chandelier (SMA130 + momentum 6M) MA **solo in regime BULL**.
- **Uscita d'emergenza:** regime di crisi con posizione in perdita → "MACRO CRASH EXIT".
- **Uscite identiche** a golden_chandelier: Chandelier Exit adattivo + trend break SMA130.

**A cosa serve:** misurare quanto del drawdown della logica live viene dal comprare
in bear market. Se il MaxDD cala in modo significativo a parità di rendimento,
la raccomandazione è aggiungere il filtro macro alla logica live di `trading.py`.
"""
    requires_bull = False

    def can_buy(self, ctx):
        """Acquisti solo in BULL: stesso pacchetto macro di AI Momentum
        (niente acquisti in bear/volatile), gestito esplicitamente qui."""
        return ctx.regime == "BULL"

    def exit_signal(self, pos, row, ctx):
        if ctx.is_crash:
            roi = (row['Close'] / pos['entry_price']) - 1
            if roi < 0.05:
                return ("MACRO CRASH EXIT", 1.0, row['Close'])
        return super().exit_signal(pos, row, ctx)

    def exit_outlook(self, pos, row, ctx):
        base = super().exit_outlook(pos, row, ctx)
        macro = "• Macro crash: se il regime entra in crisi con P/L < +5% → vende tutto"
        return "\n".join(x for x in (macro, base) if x)


class LowVolTrendStrategy(GoldenChandelierStrategy):
    key = "low_vol_trend"
    name = "🧘 Steady Winners (trend a bassa volatilità)"
    description = """
**ESPERIMENTO a variabile singola** — test dell'**anomalia low-volatility** su Xetra:

- **Ingresso:** come golden_chandelier (SMA130 + momentum 6M) MA solo se la
  **volatilità annualizzata del ticker ≤ 1.25× quella del benchmark**:
  si comprano i vincitori *calmi* (bond, oro, dividendi, large-cap stabili),
  non i pump ad alta volatilità.
- **Ranking:** momentum + bonus di calma (più il titolo è tranquillo rispetto
  al mercato, più sale in classifica).
- **Uscite identiche** a golden_chandelier: Chandelier Exit adattivo + trend break SMA130.

**A cosa serve:** capire se su questo universo conviene il profil difensivo
(meno rendimento, molto meno drawdown) invece di inseguire i vincitori caldi.
"""
    requires_bull = False

    def prepare(self, df, spy_df=None, cfg=None):
        """Colonne di golden_chandelier + ANN_VOL: volatilita' annualizzata del
        ticker in % (Vol_20 x sqrt(252), o std 20gg se Vol_20 manca)."""
        df = super().prepare(df, spy_df, cfg)
        if 'Vol_20' in df.columns:
            df['ANN_VOL'] = df['Vol_20'] * np.sqrt(252.0)
        else:
            df['ANN_VOL'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252.0)
        return df

    def entry(self, row, ctx):
        """Filtro low-vol: solo titoli calmi (vol annua <= 1.25x quella del
        benchmark), con ranking che premia la calma. Senza benchmark degrada
        all'entry golden semplice."""
        c, sma, mom = row['Close'], row.get('SMA_130'), row.get('MOM_6M')
        ann_vol, bench_vol = row.get('ANN_VOL'), row.get('BENCH_ANNVOL')
        if any(v is None or pd.isna(v) for v in (sma, mom, ann_vol)):
            return 0, "Dati insufficienti"
        if c <= sma:
            return 0, "Sotto SMA130"
        if mom <= 0:
            return 0, "Momentum 6M negativo"
        if bench_vol is not None and pd.notna(bench_vol) and bench_vol > 0:
            vol_ratio = float(ann_vol) / float(bench_vol)
            if vol_ratio > 1.25:
                return 0, f"Troppo volatile ({vol_ratio:.1f}x benchmark)"
            score = 55 + min(35.0, mom * 100.0) + (1.25 - vol_ratio) * 12.0
            return score, f"STEADY: Mom6M {mom:+.1%} | Vol {vol_ratio:.2f}x benchmark"
        return 55 + min(45.0, mom * 100.0), f"STEADY (no bench): Mom6M {mom:+.1%}"


STRATEGIES: Dict[str, BaseStrategy] = {
    s.key: s for s in [
        AIMomentumStrategy(),
        TrendFusionStrategy(),
        TurtleBreakoutStrategy(),
        GoldenChandelierStrategy(),
        GoldenChandelierDefensiveStrategy(),
        LowVolTrendStrategy(),
    ]
}

STRATEGY_ORDER: List[str] = [
    "ai_momentum", "trend_fusion", "turtle_breakout", "golden_chandelier",
    "golden_chandelier_def", "low_vol_trend",
]


def get_strategy(key: str) -> BaseStrategy:
    """Istanza della strategia dal registro (ValueError se la chiave non e'
    tra quelle attive)."""
    strat = STRATEGIES.get(key)
    if strat is None:
        raise ValueError(f"Strategia sconosciuta: '{key}'. Disponibili: {list(STRATEGIES)}")
    return strat


class _FastRowPanel:
    """Accesso rapido alle righe per data di simulazione.

    `df.loc[data]` costa ~60-100µs per chiamata: con universi grandi e
    ribilanciamenti frequenti e' il collo di bottiglia del loop. Questo
    pannello preindicizza le date una sola volta e costruisce le righe da
    array numpy (~5x piu' veloce), con la stessa API delle strategie
    (pd.Series indicizzata per nome colonna).
    """
    __slots__ = ('_entries', '_closes_ff')

    def __init__(self, market_data: Dict[str, pd.DataFrame], sim_dates: pd.DatetimeIndex):
        self._entries = {}
        self._closes_ff = {}
        n_sim = len(sim_dates)
        for t, df in market_data.items():
            if df is None or df.empty:
                continue
            pos = df.index.get_indexer(sim_dates)
            arr = df.to_numpy()
            close_col = df.columns.get_loc('Close') if 'Close' in df.columns else None
            self._entries[t] = (df.columns, arr, pos, close_col)
            if close_col is not None and n_sim:
                cf = np.full(n_sim, np.nan)
                valid = pos >= 0
                if valid.any():
                    vals = pd.to_numeric(pd.Series(arr[pos[valid], close_col]),
                                         errors='coerce').to_numpy(dtype=float)
                    cf[valid] = vals
                self._closes_ff[t] = pd.Series(cf).ffill().to_numpy()

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

    def close_ff(self, ticker: str, i: int):
        """Ultimo Close noto alla i-esima data di simulazione (ffill).

        A differenza di `close`, non restituisce None se il ticker non ha
        seduta quel giorno (None solo se non ha mai quotato). Usato dal NAV
        giornaliero mark-to-market e dalla chiusura finale."""
        arr = self._closes_ff.get(ticker)
        if arr is None or i < 0 or i >= len(arr):
            return None
        v = arr[i]
        return float(v) if np.isfinite(v) else None


_PREPARED_CACHE: Dict[Tuple[str, str], Tuple[str, pd.DataFrame]] = {}
_PREPARED_CACHE_ROWS = 0
_PREPARED_CACHE_MAX_ROWS = int(os.environ.get('ETF_METRICS_PREPARED_CACHE_ROWS', 6_000_000))


def _frame_fingerprint(df: pd.DataFrame) -> str:
    """Identificativo economico del contenuto di un frame (invalida la cache)."""
    try:
        return (f"{len(df)}|{df.index[0]}|{df.index[-1]}|"
                f"{df['Close'].iloc[0]:.6f}|{df['Close'].iloc[-1]:.6f}")
    except Exception:
        return f"{id(df)}|{len(df)}"


def _prepare_strategy_frames(strat, market_data: Dict[str, pd.DataFrame],
                             spy_df, cfg: Dict) -> Dict[str, pd.DataFrame]:
    """Applica `strat.prepare` a ogni frame con cache in-processo.

    La cache restituisce lo stesso oggetto finche' dati e parametri non
    cambiano (fingerprint su lunghezza, estremi dell'indice e prezzi): i
    re-run (es. cambio commissioni in UI) non ricalcolano gli indicatori.
    Ha un budget in righe (env ETF_METRICS_PREPARED_CACHE_ROWS, default 6M):
    superato il budget, viene svuotata per limitare la memoria.
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


def _liquidity_gate(row, alloc_value: float, cfg: Dict) -> Tuple[bool, Optional[float]]:
    """Gate di commerciabilita' dell'ingresso.

    Un ordine di `alloc_value` euro passa solo se: prezzo >= MIN_ENTRY_PRICE,
    Volume del giorno >= MIN_ENTRY_VOLUME, allocazione <= MAX_POSITION_VS_ADV
    del turnover medio (ADV_EUR_20: mediana 20gg, causale con shift(1)).
    Ritorna (esito, quota ADV usata o None). Dati senza Volume/ADV =
    ingresso rifiutato (comportamento conservativo).
    """
    try:
        price = float(row['Close'])
    except (TypeError, ValueError):
        return False, None
    if not np.isfinite(price) or price < cfg.get('MIN_ENTRY_PRICE', 1.0):
        return False, None
    vol = row.get('Volume')
    try:
        vol = float(vol) if (vol is not None and not pd.isna(vol)) else None
    except (TypeError, ValueError):
        vol = None
    if vol is None or vol < cfg.get('MIN_ENTRY_VOLUME', 500):
        return False, None
    adv = row.get('ADV_EUR_20')
    try:
        adv = float(adv) if (adv is not None and not pd.isna(adv)) else None
    except (TypeError, ValueError):
        adv = None
    if adv is None or adv <= 0:
        return False, None
    frac = float(alloc_value) / adv
    if frac > cfg.get('MAX_POSITION_VS_ADV', 0.20):
        return False, None
    return True, frac


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
    loss_carryforward_years: Optional[int] = 4,
):
    """
    Esegue il backtest della strategia selezionata.

    Preparazione: dati da preloaded_data (gia' pronti: _ensure_processed e'
    no-op) o dal DB; benchmark effettivo = SPY se disponibile, altrimenti la
    proxy equal-weight aggiunta da _ensure_processed; frame della strategia
    tramite cache (i re-run non ricalcolano gli indicatori); regimi
    precalcolati in una passata vettoriale; accesso alle righe via
    _FastRowPanel.

    Ciclo su ogni seduta di simulazione:
      1. USCITE in tre livelli: (1a) stop loss "fisico" del motore (se la
         strategia usa stop: Low < stop -> vende a max(Open, stop));
         (1b) regole di uscita della strategia (exit_signal); (1c) se non si
         esce, manutenzione della posizione (update_position: trailing stop,
         flag TP). Ogni vendita paga la fiscalita': l'imposta (default 26%)
         scatta sulla plusvalenza DOPO il recupero delle minusvalenze
         realizzate precedenti (art. 68 c.9 TUIR, finestra
         loss_carryforward_years: None = illimitato, 0 = solo annuale).
      2. INGRESSI: solo se il regime lo ammette (can_buy), entro il limite
         mensile e solo ogni REBALANCE_DAYS; candidati ordinati per punteggio,
         veto di correlazione, sizing per slot e gate di liquidita'
         (prezzo/volume/ADV: gli ingressi scartati finiscono in
         liquidity_blocked).
      3. NAV mark-to-market a OGNI seduta (close ffilled: se il ticker non ha
         seduta quel giorno vale l'ultimo prezzo noto): DD e volatilita'
         onesti anche nei giorni senza trade; traccia anche il peso massimo
         mai raggiunto da una posizione sul NAV.

    Chiusura finale: le posizioni residue sono valutate mark-to-market
    all'ultima seduta (righe "HOLD (End)"), allo stesso prezzo del NAV
    giornaliero (finale e curva equity coincidono). Nel log delle vendite si
    registra Total_Equity (NAV) invece di Cash.

    Ritorna (df_trades, finale, diag):
      - df_trades: log BUY/SELL/HOLD; sulle righe di acquisto anche ADV_EUR
        (turnover medio 20gg) e ADV_Pct (quota usata: <= MAX_POSITION_VS_ADV)
      - finale:    NAV mark-to-market all'ultima seduta di simulazione
      - diag:      {'nav_daily': NAV di OGNI seduta (DD/Sharpe onesti),
                    'max_position_weight': peso max di una posizione sul NAV,
                    'liquidity_blocked': ingressi scartati dal gate,
                    'taxes_paid': imposte pagate (nette del recupero),
                    'tax_credit_residual': minusvalenze ancora recuperabili,
                    'losses_expired': minusvalenze decadute}

    progress_callback (opzionale): frazione di avanzamento [0,1]; il ciclo di
    vita della barra di progresso e' a carico del chiamante (UI).
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
            market_data, _bench_key = _ensure_processed(market_data, benchmark)

    if not market_data:
        return pd.DataFrame(), initial_capital

    if benchmark not in market_data and BENCH_PROXY_KEY in market_data:
        benchmark = BENCH_PROXY_KEY

    spy_df = market_data.get(benchmark)
    market_data = _prepare_strategy_frames(strat, market_data, spy_df, cfg)

    ref = benchmark if benchmark in market_data else list(market_data.keys())[0]
    sim_dates = market_data[ref].index[market_data[ref].index >= pd.to_datetime(start_date)]
    price_matrix = pd.DataFrame({t: d['Close'] for t, d in market_data.items()}).ffill()

    regime_series = _precompute_regimes(market_data.get(benchmark), sim_dates) \
        if benchmark in market_data else None

    panel = _FastRowPanel(market_data, sim_dates)

    n_sim = len(sim_dates)
    if n_sim == 0:
        return pd.DataFrame(), float(initial_capital), {
            'nav_daily': pd.Series(dtype=float),
            'max_position_weight': 0.0,
            'liquidity_blocked': 0,
            'taxes_paid': 0.0,
            'tax_credit_residual': 0.0,
            'losses_expired': 0.0,
        }

    cash = float(initial_capital)
    positions = {}
    trade_log = []
    tax_ledger = ItalianTaxLedger(tax_rate=tax_rate,
                                  carryforward_years=loss_carryforward_years)
    current_sim_month = -1
    buys_this_month = 0
    commission = float(cfg['COMMISSION'])

    nav_values = np.full(n_sim, np.nan)
    max_position_weight = 0.0
    liquidity_blocked = 0

    ctx = _MarketCtx()
    ctx.cfg = cfg
    ctx.benchmark = benchmark
    ctx.market_data = market_data
    ctx.price_matrix = price_matrix

    def _get_current_nav(curr_cash, curr_positions, i):
        """NAV mark-to-market: cassa + posizioni al close ffilled (se il
        ticker non ha seduta quel giorno — calendari diversi, sospensioni —
        vale l'ultimo prezzo noto; il prezzo di ingresso solo se mai quotato)."""
        equity = curr_cash
        for p_ticker, p_data in curr_positions.items():
            curr_p = panel.close_ff(p_ticker, i)
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
        is_crash = (regime == "DANGER" or regime == "VOLATILE")

        ctx.regime = regime
        ctx.is_crash = is_crash
        ctx.current_date = current_date

        tokens_to_sell = []

        for t, pos in positions.items():
            row = panel.row(t, i)
            if row is None:
                continue

            action = None
            if strat.use_stops and pos.get('stop_loss') is not None \
                    and pd.notna(pos['stop_loss']) and row['Low'] < pos['stop_loss']:
                exit_price = max(row['Open'], pos['stop_loss'])
                action = ("STOP LOSS", 1.0, exit_price)
            else:
                try:
                    sig = strat.exit_signal(pos, row, ctx)
                except Exception:
                    sig = None
                if sig is not None:
                    reason, portion = sig[0], sig[1]
                    price = sig[2] if len(sig) > 2 and sig[2] is not None else row['Close']
                    action = (reason, portion, price)
                else:
                    try:
                        strat.update_position(pos, row, ctx)
                    except Exception:
                        pass

            if action is not None:
                tokens_to_sell.append((t, action[2], action[0], action[1]))

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

            tax, _credit_used = tax_ledger.register_sale(gain, current_date)

            cash += (net - tax)

            if portion >= 0.99 or (pos['qty'] - qty_sell) < (0.001 if allow_fractional else 1):
                del positions[t]
            else:
                positions[t]['qty'] -= qty_sell
                positions[t]['cost_basis'] -= cost_portion

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

                    adv_val = row.get('ADV_EUR_20')
                    adv_val = float(adv_val) \
                        if (adv_val is not None and not pd.isna(adv_val)) else np.nan
                    adv_frac = np.nan
                    if cfg.get('ENABLE_LIQUIDITY_GATE', True):
                        ok_liq, adv_frac = _liquidity_gate(row, alloc_per_slot, cfg)
                        if not ok_liq:
                            liquidity_blocked += 1
                            continue

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
                            "ADV_EUR": adv_val,
                            "ADV_Pct": (adv_frac * 100.0) if np.isfinite(adv_frac) else np.nan,
                        })
                        free_slots -= 1
                        buys_this_month += 1

        nav_i = _get_current_nav(cash, positions, i)
        nav_values[i] = nav_i
        if positions and nav_i > 0:
            for pt, pdata in positions.items():
                pc = panel.close_ff(pt, i)
                if pc is not None:
                    w = (pdata['qty'] * pc) / nav_i
                    if w > max_position_weight:
                        max_position_weight = float(w)

    final_nav = float(nav_values[-1]) if np.isfinite(nav_values[-1]) else cash
    for t, pos in positions.items():
        p = panel.close_ff(t, n_sim - 1)
        if p is None:
            p = pos['entry_price']
        val = pos['qty'] * p

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

    nav_series = pd.Series(nav_values, index=sim_dates, name='NAV')
    diag = {
        'nav_daily': nav_series,
        'max_position_weight': float(max_position_weight),
        'liquidity_blocked': int(liquidity_blocked),
        'taxes_paid': float(tax_ledger.total_tax_paid),
        'tax_credit_residual': float(tax_ledger.available_credit(sim_dates[-1])),
        'losses_expired': float(tax_ledger.total_losses_expired),
    }
    return pd.DataFrame(trade_log), final_nav, diag


def _default_comparison_workers() -> int:
    """Worker massimi per il confronto parallelo.

    Default: min(strategie, cpu, 4) su Linux/macOS (fork), 1 altrove.
    Configurabile via env ETF_METRICS_BACKTEST_WORKERS. Il tetto limita
    anche la memoria (ogni worker prepara le proprie copie degli indicatori).
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


_FORK_COMPARISON_DATA = None


def _comparison_worker(strategy_key, tickers, start_date, initial_capital,
                       tax_rate, allow_fractional, strategy_params,
                       loss_carryforward_years):
    """Esegue una strategia nel processo worker.

    I dati arrivano da _FORK_COMPARISON_DATA: con il contesto fork i figli
    ereditano la memoria del genitore (copy-on-write), senza pickle di
    gigabyte di DataFrame.
    """
    df_trades, final, diag = run_market_aware_backtest(
        tickers,
        start_date=start_date,
        initial_capital=initial_capital,
        preloaded_data=_FORK_COMPARISON_DATA,
        tax_rate=tax_rate,
        allow_fractional=allow_fractional,
        strategy=strategy_key,
        strategy_params=strategy_params,
        loss_carryforward_years=loss_carryforward_years,
    )
    return strategy_key, df_trades, final, diag


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
    loss_carryforward_years: Optional[int] = 4,
) -> Dict[str, Dict]:
    """
    Esegue piu' strategie sugli stessi dati (confronto fair).

    I dati sono caricati/preparati UNA sola volta per tutte le strategie
    (fair comparison + un solo accesso al DB). Su Linux/macOS le strategie
    girano in processi separati (fork, ~3-4x piu' veloce); un worker morto
    (es. OOM) viene rieseguito in seriale nel processo padre. Il dizionario
    ritornato preserva l'ordine richiesto (in parallelo le strategie
    completano in ordine sparso).

    Ritorna {strategy_key: {'trades', 'final', 'name', 'nav' (Serie NAV
    giornaliera), 'diag'}}.
    """
    keys = [k for k in (strategies or STRATEGY_ORDER) if k in STRATEGIES]
    results: Dict[str, Dict] = {}
    n = len(keys)

    def _store(key, df_trades, final, diag=None):
        results[key] = {
            "trades": df_trades,
            "final": final,
            "name": STRATEGIES[key].name,
            "nav": (diag or {}).get('nav_daily'),
            "diag": diag or {},
        }

    def _run_serial(key):
        df_trades, final, diag = run_market_aware_backtest(
            tickers,
            start_date=start_date,
            initial_capital=initial_capital,
            preloaded_data=shared,
            tax_rate=tax_rate,
            allow_fractional=allow_fractional,
            strategy=key,
            strategy_params=strategy_params,
            loss_carryforward_years=loss_carryforward_years,
        )
        _store(key, df_trades, final, diag)

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
                    tax_rate, allow_fractional, strategy_params,
                    loss_carryforward_years): key for key in keys}
                done = 0
                for fut in as_completed(futures):
                    key = futures[fut]
                    try:
                        _, df_trades, final, diag = fut.result()
                        _store(key, df_trades, final, diag)
                    except Exception as e:
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
      - 'benchmark': tutto il capitale sul benchmark (o 'benchmark_proxy':
        proxy equal-weight dell'universo se SPY non c'e', chiave separata per
        l'etichetta corretta in UI)
      - 'equal_weight': capitale suddiviso in parti uguali sull'universo.
        La curva usa la media dei rendimenti dei soli ticker quotati quel
        giorno (skipna): con dropna() su TUTTE le colonne partirebbe solo
        quando l'ultimo ticker dell'universo e' quotato (anni di dati
        ignorati).
    """
    start = pd.to_datetime(start_date)
    out: Dict[str, pd.Series] = {}

    if benchmark_ticker in market_data:
        s = market_data[benchmark_ticker]['Close']
        s = s[s.index >= start].dropna()
        if len(s) > 1:
            out['benchmark'] = initial_capital * (s / s.iloc[0])
    elif BENCH_PROXY_KEY in market_data:
        s = market_data[BENCH_PROXY_KEY]['Close']
        s = s[s.index >= start].dropna()
        if len(s) > 1:
            out['benchmark_proxy'] = initial_capital * (s / s.iloc[0])

    cols = [t for t in tickers if t in market_data and t != benchmark_ticker]
    if len(cols) >= 2:
        pm = pd.DataFrame({t: market_data[t]['Close'] for t in cols}).ffill()
        pm = pm[pm.index >= start]
        if len(pm) > 1:
            rets = pm.pct_change().mean(axis=1)
            out['equal_weight'] = initial_capital * (1 + rets.fillna(0)).cumprod()

    return out
