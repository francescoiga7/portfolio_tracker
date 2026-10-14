# -*- coding: utf-8 -*-
"""Backtest mensile dei Satelliti PAC: replay storico del consiglio mensile.

DUE famiglie di strategie, due motori:

1. **PAC-native** (PAC_STRATEGIES): progettate per un piano di accumulo.
   Decidono SOLO al ribilanciamento mensile (mai durante il mese), non usano
   stop giornalieri e reinvestono SUBITO i proventi delle vendite nel core
   (mai cassa ferma). Motore: `_run_pac_backtest_monthly`.

2. **Trading giornaliero** (STRATEGIES di automated_backtest): replay delle
   strategie del motore daily. La scansione dei candidati è GIORNALIERA
   finché il versamento del mese non viene investito (max 1 satellite al
   mese); vince il top-1 per punteggio (soglia min_entry_score) tra i
   candidati non correlati e ammessi dal filtro di regime (can_buy). Le
   posizioni sono gestite GIORNO PER GIORNO con stop ATR, TP1 parziale e
   uscite momentum/regime, come in `analyze_pac_portfolio`.

I mesi senza segnale (o senza candidato) destinano l'importo al CORE buy &
hold (core_fallback=True, come il consiglio live) oppure lo lasciano in
cassa; le strategie PAC-native con carry_over possono invece accumularlo.

Benchmark di riferimento: PAC puro sullo stesso core (tutti i mesi al core,
nessuna vendita), calcolato nello stesso loop.

Nessun lookahead: gli indicatori sono causali (rolling/shift) e ogni
decisione usa la riga del ticker alla data corrente — ciò che l'algoritmo
live avrebbe visto quel giorno.
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from etf_metrics.core.automated_backtest import (
    CONFIG, STRATEGIES, STRATEGY_ORDER, get_strategy,
    _ensure_processed, prepare_market_data, _prepare_strategy_frames,
    _precompute_regimes, _assess_market_regime, check_correlation_strict,
    _FastRowPanel, _MarketCtx, BENCH_PROXY_KEY, build_benchmark_proxy)


def etf_universe_in_db() -> List[str]:
    """Universo ETF (discovery) ∩ DB, SENZA fallback: solo ETF/ETC/ETP.

    Diversamente da `pac_universe_tickers` (pagina Satelliti PAC) non ricade
    mai su tutti i ticker del DB: il backtest PAC lavora solo sugli ETF.
    """
    from etf_metrics.core.etf_search_engine import load_etf_universe
    from etf_metrics.core.data_manager import MarketDataManager
    try:
        uni = set(load_etf_universe())
    except Exception:
        uni = set()
    if not uni:
        return []
    try:
        db = set(MarketDataManager().get_all_tickers())
    except Exception:
        db = set()
    return sorted(uni & db)


def _yearly_returns(ret: pd.Series) -> pd.Series:
    """Rendimento per anno civile, composto dai rendimenti giornalieri
    corretti per i versamenti (time-weighted: immune ai flussi di cassa)."""
    out = {}
    for year, grp in ret.groupby(ret.index.year):
        out[year] = float((1.0 + grp).prod() - 1.0)
    return pd.Series(out)


# 2. STRATEGIE PAC-NATIVE (mensili) ---
# =====================================================================
# Le strategie di trading giornaliero sono nate per gestire un portafoglio
# lump-sum ribilanciato ogni pochi giorni: stop ATR giornalieri, filtri di
# regime, max posizioni. Riutilizzate nel PAC producono sistematicamente
# multiple ~1.0x perché:
#   1. i proventi delle vendite restano in cassa (mai reinvestiti);
#   2. gli stop giornalieri vendono al primo ribasso ciò che è stato
#      comprato una volta al mese al top del ranking;
#   3. i filtri di regime bloccano l'acquisto proprio nei mesi in cui un
#      PAC comprerebbe a prezzi bassi.
# Le strategie PAC-native risolvono alla radice: decisione SOLO mensile,
# niente stop intramonth, proventi delle vendite subito nel core.


class _PACMonthCtx:
    """Snapshot passato alle strategie PAC-native al ribilanciamento mensile."""
    __slots__ = ('mi', 'date', 'month_key', 'ret', 'dd', 'above', 'rank',
                 'bench_ret', 'bench_dd', 'positions', 'budget', 'nav', 'target')


class _PACStrategy:
    """Interfaccia delle strategie PAC-native.

    `decide_month(ctx)` ritorna:
      ('buy', ticker, score, motivo)  → compra il satellite col budget del mese
      ('core', motivo)                → budget al core (o in cassa se core_fallback=False)
      ('skip', motivo)                → sospende il versamento (budget accumulato)
    `exit_position(pos, ctx)` ritorna il motivo della vendita al ribilanciamento
    (i proventi finiscono comunque nel core), oppure None per tenere.
    """
    key = "pac_base"
    name = "Base PAC"
    description = ""
    max_satellites = 8        # tetto posizioni simultanee
    max_carry_months = 0      # mesi di budget accumulabile (strategie skip)
    growth = None             # crescita mensile del percorso obiettivo (VA)

    def decide_month(self, ctx: _PACMonthCtx) -> tuple:
        raise NotImplementedError

    def exit_position(self, pos: Dict, ctx: _PACMonthCtx) -> Optional[str]:
        raise NotImplementedError


class PacMomentumStrategy(_PACStrategy):
    """Rotazione mensile in stile dual momentum (Antonacci)."""
    key = "pac_momentum"
    name = "🧭 Momentum Mensile (Dual Momentum)"
    description = """Rotazione mensile in stile dual momentum (Antonacci), adattata al PAC.

- **Acquisto** (al ribilanciamento mensile): classifica l'universo per rendimento
  a 12 mesi e compra il migliore non ancora detenuto, SOLO se il suo rendimento
  12m è positivo (momentum assoluto). Se la top-5 è già tutta detenuta o nessun
  ETF è positivo, il versamento va al core.
- **Uscita** (solo mensile, mai durante il mese): si vende quando l'ETF esce
  dalla top-5 dell'universo oppure perde il momentum assoluto (12m ≤ 0).
- **Proventi delle vendite**: reinvestiti subito nel core, mai in cassa.
- **Turnover atteso**: basso — poche vendite l'anno, zero whipsaw giornaliero."""
    exit_top_n = 5

    def decide_month(self, ctx):
        if len(ctx.positions) >= self.max_satellites:
            return ('core', f'max {self.max_satellites} satelliti detenuti → core')
        ret = ctx.ret
        n_valid = int(ret.notna().sum())
        valid = ret[ret.notna() & (ret > 0)].sort_values(ascending=False)
        for t, v in valid.items():
            if t in ctx.positions:
                continue
            rk = ctx.rank.get(t)
            if pd.notna(rk) and int(rk) > self.exit_top_n:
                return ('core', f'top-{self.exit_top_n} già detenuta → core')
            return ('buy', t, float(v),
                    f'momentum 12m {v:+.1%}, rank {int(rk)} su {n_valid} ETF')
        return ('core', 'nessun ETF con momentum 12m positivo → core')

    def exit_position(self, pos, ctx):
        t = pos['ticker']
        r = ctx.ret.get(t)
        if r is None or pd.isna(r) or r <= 0:
            return 'momentum 12m ≤ 0 o dato mancante'
        rk = ctx.rank.get(t)
        if rk is not None and pd.notna(rk) and int(rk) > self.exit_top_n:
            return f'uscito dalla top-{self.exit_top_n} (rank {int(rk)})'
        return None


class PacDipStrategy(_PACStrategy):
    """Compra il ribasso invece di bloccare l'acquisto nei mesi brutti."""
    key = "pac_dip"
    name = "📉 Buy the Dip (compra il ribasso)"
    description = """Il contrario del filtro macro di regime: compra quando il mercato cade.

- **Base**: nei mesi normali il versamento va al core.
- **Dip**: quando il benchmark è in drawdown > 15% dal massimo 52 settimane, il
  versamento compra l'ETF "buono" più battuto: drawdown ≥ 20% dal proprio massimo
  52s E rendimento 12m ≥ benchmark (forza relativa non persa).
- **Uscita**: al ribilanciamento mensile, dopo almeno 3 mesi, quando il prezzo
  torna al 95% del massimo 52s → si raccoglie il rimbalzo nel core.
- **Logica**: un PAC compra a prezzi bassi per costruzione; questa strategia
  concentra il "basso" dove c'è più valore relativo."""
    market_dd_trigger = -0.15   # drawdown del benchmark che attiva il dip
    cand_dd_min = -0.20         # drawdown minimo del candidato
    recover_frac = 0.95         # vendita quando torna al 95% del max 52s
    min_hold_months = 3

    def decide_month(self, ctx):
        bdd = ctx.bench_dd
        if bdd is None or bdd > self.market_dd_trigger:
            return ('core', 'mercato non in drawdown significativo → PAC al core')
        if len(ctx.positions) >= self.max_satellites:
            return ('core', f'max {self.max_satellites} satelliti detenuti → core')
        ret, dd = ctx.ret, ctx.dd
        br = ctx.bench_ret if ctx.bench_ret is not None else 0.0
        cands = [(t, dd[t]) for t in dd.index
                 if pd.notna(dd[t]) and dd[t] <= self.cand_dd_min
                 and pd.notna(ret.get(t)) and ret[t] >= br
                 and t not in ctx.positions]
        if not cands:
            return ('core', f'benchmark a {bdd:.0%} ma nessun candidato "buono" '
                            f'(drawdown ≥ {abs(self.cand_dd_min):.0%} e RS 12m ≥ benchmark)')
        cands.sort(key=lambda x: x[1])           # il più battuto
        t, d = cands[0]
        return ('buy', t, float(-d),
                f'il più battuto: {d:.0%} dal max 52s, RS 12m {ret[t]:+.1%} '
                f'≥ benchmark ({br:+.1%}); benchmark a {bdd:.0%}')

    def exit_position(self, pos, ctx):
        if ctx.mi - pos['entry_mi'] < self.min_hold_months:
            return None
        d = ctx.dd.get(pos['ticker'])
        if d is not None and pd.notna(d) and d >= self.recover_frac - 1.0:
            return (f'rimbalzo raccolto: prezzo ≥ {self.recover_frac:.0%} '
                    'del massimo 52s → core')
        return None


class PacValueAveragingStrategy(_PACStrategy):
    """Timing del versamento sul core (Edleson, versione a budget fisso)."""
    key = "pac_va"
    name = "⚖️ Value Averaging (percorso obiettivo)"
    description = """Timing del versamento sul core (value averaging di Edleson, versione a
budget fisso). Non detiene satelliti.

- **Percorso obiettivo**: dopo k mesi il portafoglio "dovrebbe" valere
  versamento × Σ(1+0,5%)^j (≈ +6%/anno).
- **NAV entro o sotto il percorso**: il versamento (più eventuali sospensioni
  accumulate) va subito nel core.
- **NAV sopra il percorso del 10%**: sospensione — il budget si accumula per i
  mesi successivi (max 2, poi va al core comunque).
- **Onestà**: in un mercato sempre rialzista perde contro il PAC core puro
  (compra tardi); il vantaggio arriva nei mercati laterali e volatili."""
    max_carry_months = 2
    growth = 0.005    # crescita mensile del percorso (+0,5%/mese ≈ +6%/anno)
    band = 0.10       # tolleranza sopra il percorso prima di sospendere

    def decide_month(self, ctx):
        nav, target = ctx.nav, ctx.target
        if nav is not None and target and nav > target * (1.0 + self.band):
            return ('skip', f'NAV €{nav:,.0f} sopra il percorso €{target:,.0f} '
                            f'del {self.band:.0%} → sospendo il versamento')
        return ('core', 'NAV entro o sotto il percorso obiettivo → PAC al core')

    def exit_position(self, pos, ctx):
        return None   # la VA non detiene satelliti


class PacFaberStrategy(_PACStrategy):
    """Trend following mensile in stile Faber (SMA 10 mesi)."""
    key = "pac_faber"
    name = "🐢 Faber 10 Mesi (trend lento)"
    description = """Trend following mensile in stile Faber (SMA 10 mesi), la versione più lenta.

- **Acquisto**: il migliore per rendimento 12m tra gli ETF con prezzo sopra la
  SMA a 10 mesi (210 sedute). Altrimenti il versamento va al core.
- **Uscita**: solo al ribilanciamento mensile, se la chiusura scende sotto la
  SMA a 10 mesi. Niente stop giornalieri.
- **Turnover atteso**: minimo (1-2 vendite l'anno)."""

    def decide_month(self, ctx):
        if len(ctx.positions) >= self.max_satellites:
            return ('core', f'max {self.max_satellites} satelliti detenuti → core')
        ret, above = ctx.ret, ctx.above
        ok = [t for t in ret.index
              if pd.notna(ret[t]) and bool(above.get(t, False))
              and t not in ctx.positions]
        if not ok:
            return ('core', 'nessun ETF sopra la SMA a 10 mesi → core')
        ok.sort(key=lambda t: ret[t], reverse=True)
        t = ok[0]
        return ('buy', t, float(ret[t]),
                f'Close > SMA 10 mesi, RS 12m {ret[t]:+.1%}')

    def exit_position(self, pos, ctx):
        if not bool(ctx.above.get(pos['ticker'], False)):
            return 'chiusura mensile < SMA 10 mesi'
        return None


PAC_STRATEGIES: Dict[str, _PACStrategy] = {
    s.key: s for s in (PacMomentumStrategy(), PacDipStrategy(),
                       PacValueAveragingStrategy(), PacFaberStrategy())}
PAC_STRATEGY_ORDER: List[str] = ['pac_momentum', 'pac_dip', 'pac_va', 'pac_faber']


def _run_pac_backtest_monthly(
    strategy_key: str,
    tickers: List[str],
    monthly_amount: float = 200.0,
    start_date: str = "2019-01-01",
    commission: float = 1.0,
    core_ticker: str = "VWCE.MI",
    core_fallback: bool = True,
    preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Optional[Dict]:
    """Backtest PAC con strategia NATIVA mensile. Stesso contratto e stesso
    formato dei risultati del percorso daily di `run_pac_backtest`."""
    pstrat = PAC_STRATEGIES[strategy_key]

    # --- dati: alle strategie mensili servono solo i Close (niente indicatori
    #     pesanti: run molto più veloci del percorso daily) ---
    if preloaded_data is not None:
        market_data = {t: df for t, df in preloaded_data.items()
                       if df is not None and not df.empty}
    else:
        market_data = prepare_market_data(
            list(tickers) + [core_ticker], period="max", allow_download=False)
    if not market_data:
        return None
    benchmark = CONFIG['SPY_TICKER']
    if benchmark not in market_data:
        proxy = build_benchmark_proxy(market_data)
        if proxy is not None:
            market_data[BENCH_PROXY_KEY] = proxy
            benchmark = BENCH_PROXY_KEY
    if benchmark not in market_data:
        return None

    core_available = core_ticker in market_data
    full_index = market_data[benchmark].index
    start_ts = pd.to_datetime(start_date)
    sim_mask = full_index >= start_ts
    if not sim_mask.any():
        return None
    sim_pos = np.where(sim_mask)[0]
    sim_dates = full_index[sim_pos]
    n_sim = len(sim_dates)

    universe = [t for t in dict.fromkeys(tickers)
                if t in market_data and t not in (benchmark, core_ticker)]

    # --- primo giorno utile di ogni mese = giorno di ribilanciamento ---
    first_of_month: List[int] = []
    seen_months = set()
    for p in sim_pos:
        d = full_index[p]
        mk = (d.year, d.month)
        if mk not in seen_months:
            seen_months.add(mk)
            first_of_month.append(int(p))
    mi_dates = full_index[first_of_month]
    mi_of_date = {d: mi for mi, d in enumerate(mi_dates)}
    sim_idx_of_date = {d: i for i, d in enumerate(sim_dates)}

    # --- storico allineato al calendario del benchmark + frame mensili ---
    # (stessa convenzione del motore daily: si decide e si esegue alla
    #  chiusura del giorno di ribilanciamento, senza lookahead)
    px_cols, ret_cols, dd_cols, above_cols = {}, {}, {}, {}
    for t in [benchmark] + universe + ([core_ticker] if core_available else []):
        s = market_data[t]['Close'].reindex(full_index).ffill()
        px_cols[t] = s
        ret_cols[t] = (s / s.shift(252) - 1.0).iloc[first_of_month]
        dd_cols[t] = (s / s.rolling(252, min_periods=252).max()
                      - 1.0).iloc[first_of_month]
        above_cols[t] = (s > s.rolling(210, min_periods=210).mean()) \
            .iloc[first_of_month]
    px = pd.DataFrame(px_cols)                     # close ffilled, storico pieno
    sim_px = px.iloc[sim_pos].values               # righe = giorni di simulazione
    col_idx = {t: j for j, t in enumerate(px.columns)}
    core_col = col_idx.get(core_ticker)
    m_ret = pd.DataFrame(ret_cols)[universe]       # righe = mesi, solo satelliti
    m_dd = pd.DataFrame(dd_cols)[universe]
    m_above = pd.DataFrame(above_cols)[universe]
    m_close = pd.DataFrame({t: px_cols[t].iloc[first_of_month] for t in px_cols})
    m_rank = m_ret.rank(ascending=False, axis=1)   # rank momentum 12m per mese

    cash = 0.0
    deposited = 0.0
    commissions_paid = 0.0
    positions: Dict[str, Dict] = {}
    core_qty = 0.0
    core_pure_qty = 0.0
    trades: List[Dict] = []
    monthly: List[Dict] = []
    nav_values = np.full(n_sim, np.nan)
    core_nav_values = np.full(n_sim, np.nan)
    deposit_values = np.zeros(n_sim)
    carry = 0.0
    n_buys = n_sells = 0
    months_total = months_no_signal = months_core = months_skipped = 0
    month_key = ''

    def _close_at(i: int, t: str) -> Optional[float]:
        v = sim_px[i, col_idx[t]]
        return None if np.isnan(v) else float(v)

    def _nav_now(i: int) -> float:
        nav = cash
        if core_available and core_col is not None:
            cp = _close_at(i, core_ticker)
            if cp:
                nav += core_qty * cp
        for t, pos in positions.items():
            cp = _close_at(i, t)
            nav += pos['qty'] * (cp if cp else pos['entry_price'])
        return nav

    def _buy_core(amount: float, i: int, date: pd.Timestamp, why: str) -> bool:
        nonlocal core_qty, commissions_paid, cash
        if not core_available or amount <= commission:
            return False
        cprice = _close_at(i, core_ticker)
        if not cprice or cprice <= 0:
            return False
        q = (amount - commission) / cprice
        core_qty += q
        cash -= q * cprice + commission
        commissions_paid += commission
        trades.append({
            'data': date.date(), 'tipo': 'ACQUISTO CORE', 'ticker': core_ticker,
            'qty': round(q, 4), 'prezzo': round(cprice, 4),
            'motivo': why, 'P&L (€)': 0.0, 'cash dopo (€)': round(cash, 2),
        })
        return True

    def _sell_to_core(t: str, pos: Dict, price: float, reason: str,
                      i: int, date: pd.Timestamp):
        """Vende il satellite e reinveste SUBITO i proventi nel core: è il
        fix strutturale del motore daily (lì i proventi restavano in cassa
        a rendimento zero)."""
        nonlocal cash, commissions_paid, n_sells
        qty = pos['qty']
        pnl = (price - pos['entry_price']) * qty
        proceeds = qty * price - commission
        cash += proceeds
        commissions_paid += commission
        n_sells += 1
        trades.append({
            'data': date.date(), 'tipo': 'VENDITA', 'ticker': t,
            'qty': round(qty, 4), 'prezzo': round(price, 4),
            'motivo': reason, 'P&L (€)': round(pnl, 2),
            'cash dopo (€)': round(cash, 2),
        })
        del positions[t]
        if proceeds > commission:
            _buy_core(proceeds, i, date, f'provento vendita {t} → core')

    for i in range(n_sim):
        date = sim_dates[i]
        mi = mi_of_date.get(date)

        if mi is not None:
            # --- inizio mese: versamento + ribilanciamento ---
            months_total += 1
            cash += monthly_amount
            deposited += monthly_amount
            deposit_values[i] += monthly_amount
            budget = monthly_amount + carry
            carry = 0.0
            month_key = f"{date.year}-{date.month:02d}"

            # benchmark interno: PAC puro sul core, ogni mese (come il daily)
            if core_available:
                cprice = _close_at(i, core_ticker)
                if cprice and monthly_amount > commission:
                    core_pure_qty += (monthly_amount - commission) / cprice

            ctx = _PACMonthCtx()
            ctx.mi = mi
            ctx.date = date
            ctx.month_key = month_key
            ctx.ret = m_ret.iloc[mi]
            ctx.dd = m_dd.iloc[mi]
            ctx.above = m_above.iloc[mi]
            ctx.rank = m_rank.iloc[mi]
            brv = ret_cols[benchmark].iloc[mi]
            ctx.bench_ret = float(brv) if pd.notna(brv) else None
            bdv = dd_cols[benchmark].iloc[mi]
            ctx.bench_dd = float(bdv) if pd.notna(bdv) else None
            ctx.positions = dict(positions)
            ctx.budget = budget
            ctx.nav = None
            ctx.target = None

            # --- 1. USCITE (solo mensili: mai stop intramonth) ---
            if positions:
                for t in sorted(list(positions.keys())):
                    reason = pstrat.exit_position(positions[t], ctx)
                    if reason:
                        price = m_close.at[date, t]
                        price = float(price) if pd.notna(price) and price > 0 \
                            else positions[t]['entry_price']
                        _sell_to_core(t, positions[t], price, reason, i, date)
                ctx.positions = dict(positions)

            # --- 2. decisione sul budget del mese ---
            g = pstrat.growth
            ctx.nav = _nav_now(i)
            ctx.target = (monthly_amount * (((1.0 + g) ** months_total - 1.0) / g)
                          if g else float(months_total) * monthly_amount)
            decision = pstrat.decide_month(ctx)
            kind = decision[0]

            if kind == 'buy':
                _, t, score, reason = decision
                price = m_close.at[date, t]
                investable = budget - commission
                if pd.notna(price) and price > 0 and investable > 0:
                    qty = investable / float(price)
                    positions[t] = {
                        'ticker': t, 'qty': qty, 'entry_price': float(price),
                        'entry_date': date, 'entry_mi': mi,
                        'entry_score': float(score),
                    }
                    cash -= qty * price + commission
                    commissions_paid += commission
                    n_buys += 1
                    monthly.append({
                        'mese': month_key, 'azione': '🛰️ acquistato',
                        'ticker': t, 'score': round(float(score), 4),
                        'motivo': reason, 'qty': round(qty, 4),
                        'prezzo': round(float(price), 2),
                    })
                else:
                    months_no_signal += 1
                    ok = core_fallback and _buy_core(
                        budget, i, date, 'dati insufficienti → core')
                    if ok:
                        months_core += 1
                    monthly.append({
                        'mese': month_key,
                        'azione': '⛔ dati mancanti → 🛡️ core' if ok
                                  else '⛔ dati mancanti → cassa',
                        'ticker': t, 'score': '', 'motivo': reason,
                        'qty': '', 'prezzo': '',
                    })
            elif kind == 'skip':
                months_no_signal += 1
                months_skipped += 1
                carry = budget
                action = '⏭️ sospeso (budget accumulato)'
                if pstrat.max_carry_months \
                        and carry > pstrat.max_carry_months * monthly_amount \
                        and core_fallback \
                        and _buy_core(carry, i, date,
                                      'sospensioni accumulate → core'):
                    months_core += 1
                    carry = 0.0
                    action = '⏭️ sospeso → 🛡️ core (budget accumulato)'
                monthly.append({
                    'mese': month_key, 'azione': action, 'ticker': '',
                    'score': '', 'motivo': decision[1], 'qty': '', 'prezzo': '',
                })
            else:  # 'core'
                months_no_signal += 1
                ok = core_fallback and _buy_core(budget, i, date, decision[1])
                if ok:
                    months_core += 1
                    action = '🛡️ core'
                else:
                    action = '⛔ → cassa'
                monthly.append({
                    'mese': month_key, 'azione': action, 'ticker': '',
                    'score': '', 'motivo': decision[1], 'qty': '', 'prezzo': '',
                })

        # --- NAV giornaliero mark-to-market ---
        nav_values[i] = _nav_now(i)
        if core_available and core_col is not None:
            cp = _close_at(i, core_ticker)
            core_nav_values[i] = core_pure_qty * cp if cp else np.nan

        if progress_callback is not None and i % max(1, n_sim // 50) == 0:
            try:
                progress_callback(i / n_sim)
            except Exception:
                pass

    nav_series = pd.Series(nav_values, index=sim_dates).dropna()
    core_nav = pd.Series(core_nav_values, index=sim_dates).dropna()
    deposits = pd.Series(np.cumsum(deposit_values), index=sim_dates)

    # colonne numeriche omogenee (come nel percorso daily)
    monthly_df = pd.DataFrame(monthly)
    if not monthly_df.empty:
        for col in ('score', 'qty', 'prezzo'):
            monthly_df[col] = pd.to_numeric(monthly_df[col], errors='coerce')

    # rendimenti giornalieri corretti per i versamenti (deposito a inizio
    # giornata): stessa logica del percorso daily
    dep_arr = deposit_values[[sim_idx_of_date[d] for d in nav_series.index]]
    vals = nav_series.values
    prev = np.concatenate([[np.nan], vals[:-1]])
    okm = prev > 0
    adj = np.where(okm, (vals - dep_arr) / np.where(okm, prev, np.nan) - 1.0, np.nan)
    ret = pd.Series(adj, index=nav_series.index)
    ret = ret[ret.notna() & np.isfinite(ret)]
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) \
        if len(ret) > 1 and ret.std() > 0 else None

    dd = nav_series / nav_series.cummax() - 1.0
    final = float(nav_series.iloc[-1]) if len(nav_series) else 0.0
    core_final = float(core_nav.iloc[-1]) if len(core_nav) else None

    return {
        'strategy': pstrat.name, 'strategy_key': strategy_key,
        'engine': 'pac-native',
        'final': final, 'invested': float(deposited),
        'net': final - deposited,
        'multiple': (final / deposited) if deposited > 0 else None,
        'max_dd': float(dd.min()) if len(dd) else None,
        'sharpe': sharpe,
        'n_buys': n_buys, 'n_sells': n_sells,
        'months_total': months_total, 'months_no_signal': months_no_signal,
        'months_core': months_core, 'months_skipped': months_skipped,
        'commissions': float(commissions_paid),
        'nav_daily': nav_series, 'core_nav': core_nav, 'deposits': deposits,
        'trades': pd.DataFrame(trades),
        'monthly': monthly_df,
        'yearly': _yearly_returns(ret),
        'core_available': core_available, 'core_final': core_final,
        'universe_size': len(universe),
    }


def run_pac_backtest(
    strategy_key: str,
    tickers: List[str],
    monthly_amount: float = 200.0,
    start_date: str = "2019-01-01",
    commission: float = 1.0,
    core_ticker: str = "VWCE.MI",
    core_fallback: bool = True,
    preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Optional[Dict]:
    """Backtest PAC per UNA strategia (PAC-native mensile o replay della
    strategia di trading giornaliera). Ritorna None se i dati non bastano."""
    if strategy_key in PAC_STRATEGIES:
        # strategia PAC-native: motore mensile dedicato
        return _run_pac_backtest_monthly(
            strategy_key, tickers, monthly_amount=monthly_amount,
            start_date=start_date, commission=commission,
            core_ticker=core_ticker, core_fallback=core_fallback,
            preloaded_data=preloaded_data, progress_callback=progress_callback)
    strat = get_strategy(strategy_key)
    cfg = dict(CONFIG)
    cfg.update(strat.defaults)
    cfg['COMMISSION'] = float(commission)
    benchmark = cfg['SPY_TICKER']

    if preloaded_data is not None:
        market_data, _bk = _ensure_processed(
            {t: df for t, df in preloaded_data.items()
             if df is not None and not df.empty}, benchmark)
    else:
        market_data = prepare_market_data(
            list(tickers) + [core_ticker], period="max", allow_download=False)
    if not market_data:
        return None
    if benchmark not in market_data and BENCH_PROXY_KEY in market_data:
        benchmark = BENCH_PROXY_KEY
    if benchmark not in market_data:
        return None

    spy_df = market_data.get(benchmark)
    market_data = _prepare_strategy_frames(strat, market_data, spy_df, cfg)

    core_available = core_ticker in market_data
    ref = benchmark if benchmark in market_data else list(market_data.keys())[0]
    start_ts = pd.to_datetime(start_date)
    sim_dates = market_data[ref].index[market_data[ref].index >= start_ts]
    n_sim = len(sim_dates)
    if n_sim == 0:
        return None

    price_matrix = pd.DataFrame(
        {t: d['Close'] for t, d in market_data.items()}).ffill()
    regime_series = _precompute_regimes(market_data.get(benchmark), sim_dates)
    panel = _FastRowPanel(market_data, sim_dates)
    universe = [t for t in tickers
                if t in market_data and t not in (benchmark, core_ticker)]

    ctx = _MarketCtx()
    ctx.cfg = cfg
    ctx.benchmark = benchmark
    ctx.market_data = market_data
    ctx.price_matrix = price_matrix

    cash = 0.0
    deposited = 0.0
    commissions_paid = 0.0
    positions: Dict[str, Dict] = {}
    core_qty = 0.0            # core comprato nei mesi senza segnale (fallback)
    core_pure_qty = 0.0       # benchmark: PAC puro sul core, ogni mese
    trades: List[Dict] = []
    monthly: List[Dict] = []
    nav_values = np.full(n_sim, np.nan)
    core_nav_values = np.full(n_sim, np.nan)
    deposit_values = np.zeros(n_sim)
    current_month = None
    pending = 0.0             # versamento del mese non ancora investito
    pending_month = None      # chiave 'YYYY-MM' del versamento in sospeso
    n_buys = n_sells = 0
    months_total = months_no_signal = months_core = 0

    def _sell(t: str, pos: Dict, qty: float, price: float, reason: str,
              date: pd.Timestamp):
        nonlocal cash, n_sells, commissions_paid
        qty = min(qty, pos['qty'])
        if qty <= 1e-9:
            return
        pnl = (price - pos['entry_price']) * qty
        cash += qty * price - commission
        commissions_paid += commission
        n_sells += 1
        pos['qty'] -= qty
        trades.append({
            'data': date.date(), 'tipo': 'VENDITA', 'ticker': t,
            'qty': round(qty, 4), 'prezzo': round(price, 4),
            'motivo': reason, 'P&L (€)': round(pnl, 2),
            'cash dopo (€)': round(cash, 2),
        })
        if pos['qty'] <= 1e-9:
            del positions[t]

    def _buy_core(amount: float, i: int, date: pd.Timestamp, why: str):
        nonlocal core_qty, commissions_paid, cash
        cprice = panel.close_ff(core_ticker, i)
        if cprice is None or cprice <= 0 or amount <= commission:
            return False
        q = (amount - commission) / cprice
        core_qty += q
        cash -= q * cprice + commission
        commissions_paid += commission
        trades.append({
            'data': date.date(), 'tipo': 'ACQUISTO CORE', 'ticker': core_ticker,
            'qty': round(q, 4), 'prezzo': round(cprice, 4),
            'motivo': why, 'P&L (€)': 0.0,
            'cash dopo (€)': round(cash, 2),
        })
        return True

    for i, current_date in enumerate(sim_dates):
        ctx.current_date = current_date
        if regime_series is not None:
            ctx.regime = regime_series.iloc[i]
        else:
            ctx.regime = _assess_market_regime(
                price_matrix, current_date, market_data, benchmark)
        ctx.is_crash = ctx.regime in ("DANGER", "VOLATILE")

        m = (current_date.year, current_date.month)
        new_month = (m != current_month)
        if new_month:
            # il mese precedente non ha dato segnali: il versamento in sospeso
            # va al core (come il consiglio live) oppure resta in cassa
            if pending > 0:
                months_no_signal += 1
                action = {
                    'mese': pending_month, 'azione': '⛔ nessun segnale nel mese',
                    'ticker': '', 'score': '', 'motivo': '', 'qty': '', 'prezzo': '',
                }
                if core_fallback and core_available \
                        and _buy_core(pending, i, current_date,
                                      "mese senza segnale → core"):
                    months_core += 1
                    action['azione'] += " → 🛡️ core"
                monthly.append(action)
                pending = 0.0
            current_month = m
            months_total += 1
            cash += monthly_amount
            deposited += monthly_amount
            deposit_values[i] += monthly_amount
            pending = monthly_amount
            pending_month = f"{current_date.year}-{current_date.month:02d}"
            # benchmark: PAC puro sul core, ogni mese, senza vendite
            if core_available:
                cprice = panel.close_ff(core_ticker, i)
                if cprice and cprice > 0 and monthly_amount > commission:
                    core_pure_qty += (monthly_amount - commission) / cprice

        # --- USCITE giornaliere sulle posizioni satellite ---
        for t in sorted(list(positions.keys())):
            pos = positions[t]
            row = panel.row(t, i)
            if row is None:
                continue
            stop = pos.get('stop_loss')
            if stop is not None and pd.notna(stop) and row['Low'] < stop:
                _sell(t, pos, pos['qty'], float(stop), "STOP LOSS", current_date)
                continue
            try:
                sig = strat.exit_signal(pos, row, ctx)
            except Exception:
                sig = None
            if sig:
                reason, portion, price = sig
                _sell(t, pos, pos['qty'] * float(portion), float(price),
                      str(reason), current_date)
            else:
                try:
                    strat.update_position(pos, row, ctx)
                except Exception:
                    pass

        # --- ACQUISTO: scansione giornaliera finché il versamento del mese
        #     non viene investito (max 1 satellite al mese, come il consiglio
        #     live "chiedi all'algoritmo quando hai i soldi") ---
        if pending > 0 and strat.can_buy(ctx):
            candidates = []
            for t in universe:
                if t in positions:
                    continue
                row = panel.row(t, i)
                if row is None:
                    continue
                try:
                    score, reason = strat.entry(row, ctx)
                except Exception:
                    continue
                if score is not None and pd.notna(score) \
                        and score >= strat.min_entry_score:
                    candidates.append({'t': t, 'score': float(score),
                                       'reason': reason, 'row': row})
            candidates.sort(key=lambda x: x['score'], reverse=True)
            pick = None
            blocked = 0
            for cand in candidates[:50]:
                is_safe, _ = check_correlation_strict(
                    cand['t'], sorted(positions.keys()),
                    price_matrix, current_date)
                if is_safe:
                    pick = cand
                    break
                blocked += 1
            if pick is not None:
                row = pick['row']
                price = float(row['Close'])
                investable = pending - commission
                if investable > 0 and price > 0:
                    qty = investable / price
                    stop = strat.initial_stop(row, ctx, cfg)
                    positions[pick['t']] = {
                        'qty': qty, 'entry_price': price,
                        'entry_date': current_date,
                        'stop_loss': (float(stop)
                                      if stop is not None and pd.notna(stop)
                                      else None),
                        'max_close': price, 'tp1_taken': False,
                        'regime_at_entry': ctx.regime,
                        'entry_score': pick['score'],
                    }
                    cash -= qty * price + commission
                    commissions_paid += commission
                    n_buys += 1
                    pending = 0.0
                    monthly.append({
                        'mese': pending_month, 'azione': '🛰️ acquistato',
                        'ticker': pick['t'], 'score': round(pick['score'], 1),
                        'motivo': pick['reason'],
                        'qty': round(qty, 4), 'prezzo': round(price, 2),
                    })

        # --- NAV mark-to-market ---
        cprice = panel.close_ff(core_ticker, i) if core_available else None
        nav = cash + (core_qty * cprice if cprice else 0.0)
        for t, pos in positions.items():
            cp = panel.close_ff(t, i)
            nav += pos['qty'] * (cp if cp is not None else pos['entry_price'])
        nav_values[i] = nav
        core_nav_values[i] = (core_pure_qty * cprice) if cprice else np.nan

        if progress_callback is not None and (i % max(1, n_sim // 50) == 0):
            try:
                progress_callback(i / n_sim)
            except Exception:
                pass

    # ultimo mese: risolvi il versamento ancora in sospeso
    if pending > 0:
        months_no_signal += 1
        action = {
            'mese': pending_month, 'azione': '⛔ nessun segnale nel mese',
            'ticker': '', 'score': '', 'motivo': '', 'qty': '', 'prezzo': '',
        }
        if core_fallback and core_available \
                and _buy_core(pending, n_sim - 1, sim_dates[-1],
                              "mese senza segnale → core"):
            months_core += 1
            action['azione'] += " → 🛡️ core"
        monthly.append(action)
        pending = 0.0

    nav_series = pd.Series(nav_values, index=sim_dates).dropna()
    core_nav = pd.Series(core_nav_values, index=sim_dates).dropna()
    deposits = pd.Series(np.cumsum(deposit_values), index=sim_dates)

    # colonne numeriche omogenee (i mesi senza azione hanno campi vuoti:
    # serve dtype coerente per la serializzazione arrow/st.dataframe)
    monthly_df = pd.DataFrame(monthly)
    if not monthly_df.empty:
        for col in ('score', 'qty', 'prezzo'):
            monthly_df[col] = pd.to_numeric(monthly_df[col], errors='coerce')

    # Rendimenti giornalieri corretti per i versamenti (il deposito entra a
    # inizio giornata, prima del movimento di mercato)
    pos_in_sim = {d: j for j, d in enumerate(sim_dates)}
    dep_arr = deposit_values[[pos_in_sim[d] for d in nav_series.index]]
    vals = nav_series.values
    prev = np.concatenate([[np.nan], vals[:-1]])
    ok = prev > 0
    adj = np.where(ok, (vals - dep_arr) / np.where(ok, prev, np.nan) - 1.0, np.nan)
    ret = pd.Series(adj, index=nav_series.index)
    ret = ret[ret.notna() & np.isfinite(ret)]
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) \
        if len(ret) > 1 and ret.std() > 0 else None

    dd = nav_series / nav_series.cummax() - 1.0
    final = float(nav_series.iloc[-1]) if len(nav_series) else 0.0
    core_final = float(core_nav.iloc[-1]) if len(core_nav) else None

    return {
        'strategy': strat.name, 'strategy_key': strategy_key,
        'final': final, 'invested': float(deposited),
        'net': final - deposited,
        'multiple': (final / deposited) if deposited > 0 else None,
        'max_dd': float(dd.min()) if len(dd) else None,
        'sharpe': sharpe,
        'n_buys': n_buys, 'n_sells': n_sells,
        'months_total': months_total, 'months_no_signal': months_no_signal,
        'months_core': months_core,
        'commissions': float(commissions_paid),
        'nav_daily': nav_series, 'core_nav': core_nav, 'deposits': deposits,
        'trades': pd.DataFrame(trades),
        'monthly': monthly_df,
        'yearly': _yearly_returns(ret),
        'core_available': core_available, 'core_final': core_final,
        'universe_size': len(universe),
    }


def run_pac_comparison(
    tickers: List[str],
    strategies: Optional[List[str]] = None,
    monthly_amount: float = 200.0,
    start_date: str = "2019-01-01",
    commission: float = 1.0,
    core_ticker: str = "VWCE.MI",
    core_fallback: bool = True,
    preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, Optional[Dict]]:
    """Confronto PAC su più strategie (native e/o di trading giornaliero),
    stessi dati e stessi versamenti.

    Ritorna {strategy_key: risultato_di_run_pac_backtest} (valore None per
    le strategie senza dati sufficienti).
    """
    default_keys = PAC_STRATEGY_ORDER + [k for k in STRATEGY_ORDER
                                         if k in STRATEGIES]
    keys = [k for k in (strategies or default_keys)
            if k in PAC_STRATEGIES or k in STRATEGIES]
    results: Dict[str, Optional[Dict]] = {}
    for idx, key in enumerate(keys):
        results[key] = run_pac_backtest(
            key, tickers, monthly_amount=monthly_amount, start_date=start_date,
            commission=commission, core_ticker=core_ticker,
            core_fallback=core_fallback, preloaded_data=preloaded_data)
        if progress_callback is not None:
            try:
                progress_callback((idx + 1) / len(keys))
            except Exception:
                pass
    return results
