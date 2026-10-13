# -*- coding: utf-8 -*-
"""Museo delle strategie RITIRATE dal motore di backtest.

Le strategie qui dentro NON compaiono più nel "Confronto Strategie" né nel
Portafoglio Live: sono state pensionate perché meno efficienti (rendimento
risk-adjusted, coerenza tra regimi o qualità del segnale). Il codice resta
importabile per due motivi:

1. **Rispetto del criterio "tienine traccia"**: la cronaca completa (numeri,
   motivo del ritiro, eredità) vive in `docs/STRATEGY_LEDGER.md`.
2. **Ricerca futura**: quando si sperimenta un nuovo algoritmo conviene
   confrontarlo anche con le idee scartate — magari con i dati giusti
   resuscitano (es. ricostruire l'universo con i ticker delistati).

Come riattivare una strategia ritirata (per un confronto sperimentale):

    from etf_metrics.core.retired_strategies import activate
    activate("bear_market")   # da qui in poi è di nuovo in STRATEGIES

ATTENZIONE: il codice è conservato com'era al momento del ritiro. Non va
"migliorato" in silenzio: ogni variante è una nuova strategia, con una nuova
voce nel registro.
"""

from typing import Dict

import numpy as np
import pandas as pd

from etf_metrics.core.automated_backtest import BaseStrategy


# ---------------------------------------------------------------------
# Helper del Volume Profile (usato solo dalla strategia ritirata omonima)
# ---------------------------------------------------------------------
def _rolling_volume_profile(df: pd.DataFrame, window: int = 60, bins: int = 24,
                            va_pct: float = 0.70):
    """Profilo del volume per prezzo rolling: ritorna (POC, VAH, VAL) come array.

    Versione vettorizzata (sliding_window_view + bincount con offset di riga).
    Semantica identica alla versione originale a loop.
    """
    n = len(df)
    poc = np.full(n, np.nan); vah = np.full(n, np.nan); val = np.full(n, np.nan)
    if n <= window or window <= 0 or bins <= 0:
        return poc, vah, val

    close = df['Close'].to_numpy(float); high = df['High'].to_numpy(float)
    low = df['Low'].to_numpy(float); vol = np.nan_to_num(df['Volume'].to_numpy(float))
    tp = (close + high + low) / 3.0

    W_tp = np.lib.stride_tricks.sliding_window_view(tp, window)[:-1]
    W_vol = np.lib.stride_tricks.sliding_window_view(vol, window)[:-1]
    m = W_tp.shape[0]

    lo = np.nanmin(W_tp, axis=1)
    hi = np.nanmax(W_tp, axis=1)
    span = hi - lo
    tot = W_vol.sum(axis=1)
    valid = np.isfinite(lo) & np.isfinite(hi) & (span > 0) & (tot > 0)

    safe_span = np.where(valid, span, 1.0)
    rel = np.where(valid[:, None], (W_tp - lo[:, None]) / safe_span[:, None], 0.0)
    b = np.clip(np.ceil(rel * bins).astype(np.int64) - 1, 0, bins - 1)

    rows = np.repeat(np.arange(m), window)
    hist = np.bincount(rows * bins + b.ravel(),
                       weights=np.where(valid[:, None], W_vol, 0.0).ravel(),
                       minlength=m * bins).reshape(m, bins)

    poc_i = hist.argmax(axis=1)
    lo_v = np.where(valid, lo, 0.0)

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


# ---------------------------------------------------------------------
# 🐻 Bear Market Regime Switcher — RITIRATA 2026-10-12
# ---------------------------------------------------------------------
class BearMarketStrategy(BaseStrategy):
    key = "bear_market"
    name = "🐻 Bear Market Regime Switcher (ritirata)"
    description = """
**Regime Switcher** — pensata per NON farsi massacrare (e persino guadagnare) nei bear market.

- **Regime `BULL`:** compra i titoli più forti (RS vs benchmark, ADX > 20, sopra SMA50).
- **Regime `BEAR`:** cerca i titoli che **salgono mentre il mercato scende**
  (ETF inversi, oro, obbligazioni).
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
# 📊 Volume Profile — RITIRATA 2026-10-12
# ---------------------------------------------------------------------
class VolumeProfileStrategy(BaseStrategy):
    key = "volume_profile"
    name = "📊 Volume Profile (Value Area Breakout, ritirata)"
    description = """
**Volume Profile** — dove "vive" il volume istituzionale.

- Costruisce il **profilo del volume per prezzo** su 60 giorni:
  `POC` (Point of Control) e **Value Area al 70%**.
- **Acquista** quando il close rompe **sopra il Value Area High** con volume > 1.5x la media
  e prezzo sopra SMA50.
- **Esce** quando il prezzo rientra nel value area sotto il POC (fallimento del breakout)
  o su trailing stop 3x ATR.
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
            return 0, "Inside value area"
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
# 🔄 Dual Momentum 12-1 — RITIRATA 2026-10-12 (v2; la v1 fu ritirata 2026-10-05)
# ---------------------------------------------------------------------
class DualMomentumStrategy(BaseStrategy):
    key = "dual_momentum"
    name = "🔄 Dual Momentum 12-1 (rotazione mensile, ritirata)"
    description = """
**Dual Momentum in stile Antonacci/Jegadeesh-Titman** — progettata per essere
ROBUSTA, non ottimizzata: parametri standard della letteratura (momentum 12 mesi
saltando l'ultimo, ribilanciamento mensile), nessun numero tarato sui dati.

- **Momentum 12-1:** rendimento degli ultimi 252 giorni di borsa **escluso** l'ultimo
  mese (21 gg) — il salto evita l'inversione a breve termine.
- **Filtro assoluto (crash protection):** si compra solo se il momentum 12-1 è
  **positivo** e il prezzo è sopra la **SMA 200**; l'uscita scatta quando l'uno o
  l'altro si guasta.
- **Filtro relativo:** tra i candidati che superano i filtri vince il punteggio
  (momentum + RS vs benchmark): rotazione verso i vincitori.
- **Stop disastro:** ATR 4x solo come protezione estrema (il dual momentum
  classico non usa stop: le uscite sono di segnale).
- **POCO overfitting by design:** 3 sole regole, niente time stop, niente TP.
"""
    min_entry_score = 60
    defaults = {
        'MAX_POSITIONS': 3, 'REBALANCE_DAYS': 21, 'ENABLE_MONTHLY_LIMIT': True,
        'MAX_BUYS_PER_MONTH': 4, 'STOP_LOSS_ATR_MULT': 4.0, 'TIME_STOP_DAYS': 3650,
    }

    def prepare(self, df, spy_df=None, cfg=None):
        df = df.copy()
        df['MOM_12_1'] = df['Close'].pct_change(252).shift(21)
        return df

    def entry(self, row, ctx):
        c, mom, sma200 = row['Close'], row.get('MOM_12_1'), row.get('SMA200')
        if any(v is None or pd.isna(v) for v in (mom, sma200)):
            return 0, "Dati insufficienti"
        if c <= sma200:
            return 0, "Sotto SMA200"
        if mom <= 0:
            return 0, "Momentum 12-1 non positivo"
        rs = row.get('RS_Score', 0.0)
        rs_v = float(rs) if pd.notna(rs) else 0.0
        score = 55 + min(25.0, mom * 40.0) + np.clip(rs_v, -10, 20)
        return score, f"DUAL MOM: 12-1 {mom:+.1%} | RS {rs_v:+.0f}"

    def exit_signal(self, pos, row, ctx):
        c, mom, sma200 = row['Close'], row.get('MOM_12_1'), row.get('SMA200')
        if mom is not None and pd.notna(mom) and mom <= 0:
            return ("MOMENTUM ESAURITO", 1.0, c)
        if sma200 is not None and pd.notna(sma200) and c < sma200:
            return ("SMA200 ROTTA", 1.0, c)
        return None

    def update_position(self, pos, row, ctx):
        pass  # nessun trailing: le uscite sono di segnale (dual momentum classico)


# ---------------------------------------------------------------------
# REGISTRO DELLE RITIRATE
# ---------------------------------------------------------------------
# retired_on  = data del pensionamento
# reason      = motivo (numeri e dettagli in docs/STRATEGY_LEDGER.md)
# lineage     = dove vive oggi l'idea, se da qualche parte
RETIRED: Dict[str, Dict] = {
    s.key: {"class": s, "retired_on": None, "reason": "", "lineage": ""}
    for s in [BearMarketStrategy(), VolumeProfileStrategy(), DualMomentumStrategy()]
}
RETIRED["bear_market"].update({
    "retired_on": "2026-10-12",
    "reason": "Peggior profilo di rischio della suite: MaxDD -69.2% su 2021-2026 "
              "(Sharpe 0.3). L'idea di comprare ETF inversi/oro in BEAR funziona "
              "nel 2022 (+11.2%) ma il regime whipsaw del 2025-26 la fa sanguinare "
              "(-27.7% nel 2025-26) inseguendo prodotti inversi che decadono.",
    "lineage": "Il concetto di switch difensivo vive in trend_fusion "
               "(che però usa congiunzioni più prudenti) e nell'esperimento "
               "golden_chandelier_def (filtro macro sulla logica live).",
})
RETIRED["volume_profile"].update({
    "retired_on": "2026-10-12",
    "reason": "La meno coerente: unica sotto pari nello split 2021-23 (-2.8% vs "
              "S&P500 +28.9%) e con anni negativi in tre regimi diversi "
              "(2022, 2023, 2025). Il segnale VAH+volume da solo non basta: senza "
              "conferma di trend (ADX/RS) compra falsi breakout.",
    "lineage": "Il breakout dal Value Area con conferma volume è uno dei tre "
               "ingredienti di congiunzione di trend_fusion, che lo usa in modo "
               "più disciplinato.",
})
RETIRED["dual_momentum"].update({
    "retired_on": "2026-10-12",
    "reason": "Non supera il bar di mantenimento (docs/STRATEGY_LEDGER.md): "
              "nello split 2021-23 fa +4.5% contro +16.3% dell'equal-weight; "
              "Sharpe 0.3 (il peggiore insieme a bear_market); solo 78 trades "
              "(sotto la soglia di significatività, 100); il rendimento totale "
              "+392% poggia per due terzi su un solo anno (+267.5% nel 2026: "
              "concentrazione su pochi sopravvissuti del bias, non robustezza). "
              "Parametri corretti (Antonacci 12-1), risultati fragili su questo "
              "universo: l'idea è sana ma qui non paga.",
    "lineage": "v1 (momentum 252, 2 posizioni) ritirata il 2026-10-05 (commit "
               "191baea); v2 (12-1 + SMA200, parametri di letteratura) creata e "
               "ritirata il 2026-10-12. Da rivalutare su finestre più lunghe, "
               "asset class globali o universi meno distorti dal survivorship bias.",
})


def activate(key: str) -> bool:
    """Reinserisce una strategia ritirata nel registro attivo del backtest.

    Dopo `activate('bear_market')` la strategia ricompare in Confronto Strategie,
    backtest singolo e Portafoglio Live. È pensata per confronti sperimentali:
    se una ritirata battesse le attive con dati migliori, va promossa di nuovo
    (e documentata nel registro).

    Ritorna True se la strategia è ora attiva.
    """
    from etf_metrics.core.automated_backtest import STRATEGIES, STRATEGY_ORDER
    info = RETIRED.get(key)
    if info is None:
        raise ValueError(f"Strategia ritirata sconosciuta: '{key}'. "
                         f"Disponibili: {list(RETIRED)}")
    if key not in STRATEGIES:
        STRATEGIES[key] = info["class"]
        STRATEGY_ORDER.append(key)
    return True
