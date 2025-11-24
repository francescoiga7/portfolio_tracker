# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging
from huggingface_hub import InferenceClient

from etf_metrics.clients.yahoo_client import get_series, get_info, resolve_isin_one
from etf_metrics.core.trading import analyze_ticker
# IMPORTIAMO LE NUOVE METRICHE
from etf_metrics.core.metrics import (
    compute_metrics_from_series,
    get_trend_signal,
    compute_sortino_ratio,
    compute_var
)
# IMPORTIAMO IL METEO DI MERCATO E INFO ETF
from etf_metrics.core.pac_screener import get_market_regime
from etf_metrics.core.etf_info import get_etf_extended_info, fetch_ter_justetf, fallback_ter_from_yahoo_info

logger = logging.getLogger(__name__)


def get_technical_context(ticker: str, isin: str = None) -> Optional[Dict]:
    """Recupera una biopsia completa a 360 gradi (Trading + Macro + Fondamentali)."""
    try:
        # 1. Dati Storici
        df = get_series(ticker, period="2y", as_dataframe=True)
        if df is None or df.empty: return None

        # 2. Trading Signal (Il motore "Alpha Predator")
        trading_signal = analyze_ticker_alpha(ticker, df)
        if not trading_signal: return None

        # 3. Metriche Classiche + AVANZATE
        close_series = df['Close']
        metrics = compute_metrics_from_series(close_series) or {}

        # -- NUOVO: Metriche Rischio Avanzate --
        metrics['sortino'] = compute_sortino_ratio(close_series)
        metrics['var_95'] = compute_var(close_series)  # Value at Risk

        # -- NUOVO: Proximity to Highs (Forza del Trend) --
        high_52w = close_series[-252:].max()
        prox_high = (close_series.iloc[-1] / high_52w) if high_52w > 0 else 0
        metrics['prox_high'] = prox_high

        # 4. Info Fondamentali (ETF)
        info_yf = get_info(ticker=ticker)
        # Se abbiamo l'ISIN, proviamo a prendere dati più precisi (TER, Dimensione)
        etf_extra = {}
        if isin:
            etf_extra = get_etf_extended_info(isin)
            # Fallback intelligente per il TER
            if not etf_extra.get('ter_pct'):
                ter = fetch_ter_justetf(isin) or fallback_ter_from_yahoo_info(info_yf)
                etf_extra['ter_pct'] = f"{ter}%" if ter else "N/D"

        # Uniamo le info
        name = etf_extra.get('longName') or info_yf.get('longName', ticker)

        # 5. Contesto Macro (Il "Meteo")
        market_ctx = get_market_regime()  # Scarica VIX e Trend SP500

        return {
            "ticker": ticker,
            "isin": isin,
            "name": name,
            "price": close_series.iloc[-1],
            "trading": trading_signal,
            "metrics": metrics,
            "etf_info": etf_extra,  # TER, Distribuzione, ecc.
            "market": market_ctx,  # Risk-On/Off, VIX
            "trend": get_trend_signal(close_series)
        }
    except Exception as e:
        logger.error(f"Error fetching context: {e}")
        return None


def generate_advisory_response(user_input: str, hf_token: Optional[str] = None,
                               model_id: str = "google/gemma-2-9b-it") -> str:
    clean_input = user_input.strip().upper()
    ticker = resolve_isin_one(clean_input)
    isin_found = clean_input if (len(clean_input) == 12 and clean_input[0:2].isalpha()) else None

    # Se l'input era già un ticker, non abbiamo l'ISIN, ma va bene lo stesso
    if not ticker and len(clean_input.split()) == 1 and len(clean_input) < 10:
        ticker = clean_input

    if not ticker:
        return "Non trovo l'asset. Inserisci un ISIN o Ticker valido."

    # Passiamo anche l'ISIN se l'abbiamo trovato, per i dati fondamentali
    ctx = get_technical_context(ticker, isin=isin_found)
    if not ctx:
        return f"Errore dati per **{clean_input}**."

    # --- Preparazione Dati per il Prompt ---
    t = ctx['trading']
    m = ctx['metrics']
    e = ctx['etf_info']
    macro = ctx['market']

    # Calcoli rapidi per il prompt
    cagr = m.get('cagr', 0)
    mdd = m.get('mdd', 0)
    sortino = m.get('sortino', 0)
    var = m.get('var_95', 0)
    prox = m.get('prox_high', 0) * 100  # Es. 95%

    # Logica PAC
    is_pac_candidate = (cagr > 3) and (mdd > -35)

    # Stringhe descrittive
    ter_str = e.get('ter_pct', 'N/D')
    policy = e.get('distribution', 'N/D')
    fund_size = e.get('fund_size', 'N/D')
    macro_sentiment = macro.get('regime', 'Neutrale')  # Risk-On / Risk-Off
    vix = macro.get('vix', 0)

    # --- AI GENERATION ---
    if hf_token:
        try:
            client = InferenceClient(model=model_id, token=hf_token)

            system_prompt = f"""
            Sei un analista finanziario esperto e diretto. Analisi su: {ctx['name']} ({ctx['ticker']}).

            1. CONTESTO DI MERCATO (Macro):
            - Regime: {macro_sentiment} (VIX: {vix:.2f}). 
            - Nota: Se Risk-Off (VIX>20), suggerisci prudenza o size ridotte.

            2. ANALISI TECNICA (Trading):
            - Prezzo: {ctx['price']:.2f} | Trend: {ctx['trend']}
            - Forza Relativa: Al {prox:.0f}% dei massimi a 52 settimane.
            - Segnale Algo: {t['signal']} (RSI: {t['indicators'].get('RSI', 50):.1f})
            - Livelli: Supporto {t['stop_loss']:.2f} | Resistenza {t['take_profit']:.2f}

            3. PROFILO RISCHIO (Quantitativo):
            - Volatilità "Cattiva" (Sortino): {sortino:.2f} (Sopra 1.5 è ottimo).
            - Rischio 1 Giorno (VaR 95%): {var:.2f}% (Perdita attesa in un giorno "no").
            - Max Drawdown: {mdd:.2f}% | CAGR: {cagr:.2f}%

            COMPITO:
            Dai un consiglio operativo discorsivo (COMPRA, VENDI, ACCUMULA o ATTENDI). Sii specifico e spiegami quali dati
            ti fanno trarre determinate conclusioni. Dammi una stima entro quanto tempo raggiunge il prezzo atteso
            Usa i dati "Macro" per contestualizzare (es. "Nonostante il segnale buy, il mercato è nervoso...").
            Usa il "VaR" per spiegare il rischio a breve.
            """

            response = client.chat_completion(
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=800,
                temperature=0.6
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"⚠️ **Errore AI:** {str(e)}\n\nAnalisi Standard:\n" + _fallback_template_response(ctx,
                                                                                                      is_pac_candidate)

    return _fallback_template_response(ctx, is_pac_candidate)


def _fallback_template_response(ctx, is_pac_candidate):
    """Fallback schematico arricchito."""
    t = ctx['trading']
    macro = ctx['market']
    m = ctx['metrics']

    msg = f"### 🤖 Analisi Quantitativa: **{ctx['name']}**\n\n"

    # Sezione Macro
    msg += f"🌍 **Contesto Macro:** {macro.get('regime', 'N/D')} (VIX {macro.get('vix', 0):.2f})\n"
    if macro.get('vix', 0) > 20:
        msg += "⚠️ *Attenzione: Volatilità di mercato elevata. Ridurre le size.*\n\n"
    else:
        msg += "✅ *Semaforo verde dal mercato generale.*\n\n"

    # Sezione Trading
    if "LONG" in t['signal']:
        msg += f"🚀 **VERDETTO: COMPRARE**\n"
        msg += f"Segnale **{t['signal']}**. Forza relativa al {m.get('prox_high', 0) * 100:.0f}% dei massimi.\n"
        msg += f"👉 **Target:** {t['take_profit']:.2f} | 🛡️ **Stop:** {t['stop_loss']:.2f}\n"
    elif "SHORT" in t['signal']:
        msg += f"⚠️ **VERDETTO: VENDERE**\nTrend negativo. Supporti rotti.\n"
    else:
        if is_pac_candidate:
            msg += f"💰 **VERDETTO: ACCUMULO (PAC)**\n"
            msg += f"Ottimo asset (Sortino {m.get('sortino', 0):.2f}) ma fase laterale.\n"
            msg += f"📉 Compra in area **{t['stop_loss']:.2f}**.\n"
        else:
            msg += f"✋ **VERDETTO: ATTENDI**\n"
            msg += f"Canale laterale {t['stop_loss']:.2f} - {t['take_profit']:.2f}. VaR giornaliero: {m.get('var_95', 0):.2f}%.\n"

    return msg