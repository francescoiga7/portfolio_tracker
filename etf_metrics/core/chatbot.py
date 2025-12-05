# -*- coding: utf-8 -*-
from typing import Optional, Dict
import logging
from huggingface_hub import InferenceClient

from etf_metrics.clients.yahoo_client import get_series, get_info, resolve_isin_one
from etf_metrics.core.trading import analyze_ticker
from etf_metrics.core.metrics import (
    compute_metrics_from_series,
    compute_sortino_ratio,
    compute_var
)
from etf_metrics.core.pac_screener import get_market_regime
from etf_metrics.core.etf_info import get_etf_extended_info, fetch_ter_justetf, fallback_ter_from_yahoo_info

logger = logging.getLogger(__name__)


def get_technical_context(ticker: str, isin: str = None) -> Optional[Dict]:
    """Recupera una biopsia completa a 360 gradi (Trading + Macro + Fondamentali)."""
    try:
        df = get_series(ticker, period="5y", as_dataframe=True)
        if df is None or df.empty: return None

        trading_context = analyze_ticker(ticker, df)
        if not trading_context: return None

        close_series = df['Close']
        metrics = compute_metrics_from_series(close_series) or {}

        metrics['sortino'] = compute_sortino_ratio(close_series)
        metrics['var_95'] = compute_var(close_series)

        high_52w = close_series[-252:].max()
        prox_high = (close_series.iloc[-1] / high_52w) if high_52w > 0 else 0
        metrics['prox_high'] = prox_high

        info_yf = get_info(ticker=ticker)
        etf_extra = {}
        if isin:
            etf_extra = get_etf_extended_info(isin)
            if not etf_extra.get('ter_pct'):
                ter = fetch_ter_justetf(isin) or fallback_ter_from_yahoo_info(info_yf)
                etf_extra['ter_pct'] = f"{ter}%" if ter else "N/D"

        name = etf_extra.get('longName') or info_yf.get('longName', ticker)

        market_ctx = get_market_regime()

        trend_str = trading_context.get("raw_signal", "NEUTRAL")

        return {
            "ticker": ticker,
            "isin": isin,
            "name": name,
            "price": close_series.iloc[-1],
            "trading": trading_context,
            "metrics": metrics,
            "etf_info": etf_extra,
            "market": market_ctx,
            "trend": trend_str
        }
    except Exception as e:
        logger.error(f"Error fetching context: {e}")
        return None


def generate_advisory_response(user_input: str, hf_token: Optional[str] = None,
                               model_id: str = "google/gemma-2-9b-it") -> str:
    clean_input = user_input.strip().upper()
    ticker = resolve_isin_one(clean_input)
    isin_found = clean_input if (len(clean_input) == 12 and clean_input[0:2].isalpha()) else None

    if not ticker and len(clean_input.split()) == 1 and len(clean_input) < 10:
        ticker = clean_input

    if not ticker:
        return "Non trovo l'asset. Inserisci un ISIN o Ticker valido."

    ctx = get_technical_context(ticker, isin=isin_found)
    if not ctx:
        return f"Errore dati per **{clean_input}**."

    t = ctx['trading']
    m = ctx['metrics']
    e = ctx['etf_info']
    macro = ctx['market']

    cagr = m.get('cagr', 0)
    mdd = m.get('mdd', 0)
    sortino = m.get('sortino', 0)
    var = m.get('var_95', 0)
    prox = m.get('prox_high', 0) * 100

    is_pac_candidate = (cagr > 3) and (mdd > -35)

    inds = t.get('indicators', {})
    rsi_val = inds.get('RSI', 50)

    if hf_token:
        try:
            client = InferenceClient(model=model_id, token=hf_token)

            system_prompt = f"""
            Sei un analista finanziario esperto. Analisi su: {ctx['name']} ({ctx['ticker']}).

            1. CONTESTO MACRO:
            - Regime: {macro.get('regime', 'N/D')} (VIX: {macro.get('vix', 0):.2f}).

            2. DATI TECNICI:
            - Prezzo: {ctx['price']:.2f} | Trend: {ctx['trend']}
            - Segnale Algoritmo: {t['signal']} (Confidenza {t.get('confidence', 0)}%)
            - RSI: {rsi_val:.1f} | Prossimità Massimi: {prox:.0f}%
            - Stop Loss Suggerito: {t.get('stop_loss', 0):.2f}

            3. RISCHIO:
            - Sortino Ratio: {sortino:.2f}
            - VaR 95% (1gg): {var:.2f}%
            - Max Drawdown: {mdd:.2f}%

            COMPITO:
            Consiglia (COMPRA, VENDI, ACCUMULA o ATTENDI) spiegando il perché basandoti sui dati sopra. Sii sintetico e professionale.
            """

            response = client.chat_completion(
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=800,
                temperature=0.6
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"⚠️ **Errore AI:** {str(e)}\n\n" + _fallback_template_response(ctx, is_pac_candidate)

    return _fallback_template_response(ctx, is_pac_candidate)


def _fallback_template_response(ctx, is_pac_candidate):
    """Fallback schematico."""
    t = ctx['trading']
    macro = ctx['market']
    m = ctx['metrics']

    msg = f"### 🤖 Analisi Quantitativa: **{ctx['name']}**\n\n"
    msg += f"🌍 **Macro:** {macro.get('regime', 'N/D')} (VIX {macro.get('vix', 0):.2f})\n"

    if "COMPRA" in t['signal']:
        msg += f"🚀 **VERDETTO: COMPRARE**\n"
        msg += f"Segnale **{t['signal']}** ({t.get('reason', '')}).\n"
        msg += f"🛡️ **Stop:** {t.get('stop_loss', 0):.2f}\n"
    elif "VENDI" in t['signal']:
        msg += f"⚠️ **VERDETTO: VENDERE**\nTrend negativo.\n"
    else:
        msg += f"✋ **VERDETTO: ATTENDI / MANTIENI**\n"
        msg += f"Fase laterale o nessun setup chiaro.\n"

    return msg