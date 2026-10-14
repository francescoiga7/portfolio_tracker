# -*- coding: utf-8 -*-
"""Pagina UI: 🧭 Confronto Strategie PAC (backtest del consiglio mensile).

Replay storico del consiglio mensile: ogni mese versa l'importo scelto e
compra il miglior ETF dell'universo secondo la strategia. Due famiglie:

- **🧭 PAC-native** (default): decidono SOLO al ribilanciamento mensile,
  niente stop giornalieri, e reinvestiscono subito i proventi delle vendite
  nel core (momentum, buy-the-dip, value averaging, Faber);
- **strategie di trading giornaliero** (in coda, per confronto): replay
  con stop ATR, filtro di regime e proventi delle vendite in cassa.

Il motore è `core/pac_backtest.py` (`run_pac_comparison`): questa UI ne è il
rendering, e specchia la struttura core/automated_backtest.py + pac_backtest.py.

Viene renderizzata come modalità "🧭 Confronto Strategie PAC" della pagina
💹 Backtest (ui/automated_backtest.py) e ha una sidebar propria (importo PAC,
strategie, core, commissioni), indipendente dai parametri comuni del backtest.
"""
import streamlit as st
import pandas as pd

try:
    import matplotlib  # noqa: F401 — richiesto dal grafico NAV
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from etf_metrics.core.automated_backtest import (
    prepare_market_data, STRATEGIES, STRATEGY_ORDER)
from etf_metrics.core.pac_backtest import (
    run_pac_comparison, etf_universe_in_db, PAC_STRATEGIES, PAC_STRATEGY_ORDER)

# strategie selezionabili nel confronto: prima le PAC-native, poi quelle di
# trading giornaliero (mantenute per confronto)
_ALL_PAC_KEYS = PAC_STRATEGY_ORDER + [k for k in STRATEGY_ORDER if k in STRATEGIES]


def _pac_strategy_name(k: str) -> str:
    return PAC_STRATEGIES[k].name if k in PAC_STRATEGIES else STRATEGIES[k].name


def _render_pac_comparison_results(run):
    """Rendering dell'ultima run del Confronto Strategie PAC (cache di sessione)."""
    results = {k: r for k, r in run['results'].items() if r is not None}
    if not results:
        st.warning("Nessuna strategia ha prodotto risultati (dati insufficienti).")
        return
    amount = run['amount']

    # --- Grafico NAV ---
    if HAS_MATPLOTLIB:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for r in results.values():
            ax.plot(r['nav_daily'].index, r['nav_daily'].values,
                    label=r['strategy'], lw=1.3)
        ref = next(iter(results.values()))
        if ref['core_nav'] is not None and len(ref['core_nav']):
            ax.plot(ref['core_nav'].index, ref['core_nav'].values,
                    label="🛡️ Core PAC (buy & hold)", ls="--", color="black", lw=1.2)
        ax.plot(ref['deposits'].index, ref['deposits'].values,
                label="💵 Versato", ls=":", color="gray", lw=1.2)
        ax.set_ylabel("NAV (€)")
        ax.set_title(f"PAC satelliti — {amount:g}€/mese")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    # --- Tabella riassuntiva ---
    rows = []
    for r in results.values():
        rows.append({
            "Strategia": r['strategy'],
            "Finale (€)": r['final'],
            "Versato (€)": r['invested'],
            "Utile (€)": r['net'],
            "Multiplo": r['multiple'],
            "IRR anno": r.get('irr'),
            "Max DD %": (r['max_dd'] * 100) if r['max_dd'] is not None else None,
            "Sharpe": r['sharpe'],
            "Acquisti": r['n_buys'],
            "Mesi senza segnale": f"{r['months_no_signal']}/{r['months_total']}",
            "Commissioni (€)": r['commissions'],
        })
    ref = next(iter(results.values()))
    if ref['core_final'] is not None:
        rows.append({
            "Strategia": "🛡️ Core PAC (buy & hold)",
            "Finale (€)": ref['core_final'],
            "Versato (€)": ref['invested'],
            "Utile (€)": ref['core_final'] - ref['invested'],
            "Multiplo": ref['core_final'] / ref['invested'],
            "IRR anno": ref.get('core_irr'),
            "Max DD %": (ref['core_max_dd'] * 100)
                        if ref.get('core_max_dd') is not None else None,
            "Sharpe": ref.get('core_sharpe'),
            "Acquisti": ref['months_total'],
            "Mesi senza segnale": "—", "Commissioni (€)": ref['months_total'] * run['commission'],
        })
    df_sum = pd.DataFrame(rows).set_index("Strategia").sort_values("Finale (€)", ascending=False)
    st.markdown("### 🏆 Classifica")
    st.dataframe(
        df_sum.style.format({
            "Finale (€)": "€{:,.0f}", "Versato (€)": "€{:,.0f}", "Utile (€)": "€{:,.0f}",
            "Multiplo": "{:.3f}", "IRR anno": "{:+.1%}", "Max DD %": "{:.1f}",
            "Sharpe": "{:.2f}",
            "Acquisti": "{:.0f}", "Commissioni (€)": "€{:,.0f}",
        }, na_rep="—"),
        width="stretch")
    st.caption(
        "💼 **IRR anno** = rendimento annuo effettivo ponderato per i versamenti "
        "(money-weighted): in un PAC è il numero giusto da guardare, perché il "
        "capitale arriva gradualmente (il multiplo dipende dalla durata del piano, "
        "l'IRR dal rendimento reale del capitale in linea).")

    best = df_sum.iloc[0]
    core_row = df_sum.loc["🛡️ Core PAC (buy & hold)"] \
        if "🛡️ Core PAC (buy & hold)" in df_sum.index else None
    if core_row is not None:
        if best.name != "🛡️ Core PAC (buy & hold)":
            if best["Finale (€)"] > core_row["Finale (€)"]:
                st.success(f"🥇 **{best.name}** batte il core PAC: €{best['Finale (€)']:,.0f} "
                           f"vs €{core_row['Finale (€)']:,.0f} "
                           f"({best['Multiplo']:.3f}x vs {core_row['Multiplo']:.3f}x).")
            else:
                st.info(f"🛡️ In questo periodo il **core PAC** avrebbe reso di più "
                        f"(€{core_row['Finale (€)']:,.0f} vs €{best['Finale (€)']:,.0f} "
                        f"della migliore strategia): i satelliti non hanno pagato.")

    # --- Rendimenti per anno civile ---
    years = sorted({int(y) for r in results.values() for y in r['yearly'].index})
    if years:
        tbl = {}
        for r in results.values():
            tbl[r['strategy']] = {y: r['yearly'].get(y) for y in years}
        ydf = pd.DataFrame(tbl).T
        ydf.columns = [str(y) for y in years]
        with st.expander("📅 Rendimenti per anno civile (time-weighted)"):
            st.dataframe(ydf.style.format("{:+.1%}", na_rep="—"), width="stretch")

    # --- Log mensile e operazioni per strategia ---
    sel = st.selectbox(
        "📋 Log di quale strategia?",
        list(results.keys()), format_func=lambda k: results[k]['strategy'])
    r = results[sel]
    st.markdown("**🗓️ Log mensile** — cosa avrebbe fatto l'algoritmo ogni mese:")
    st.dataframe(r['monthly'], width="stretch", hide_index=True)
    if r['trades'] is not None and not r['trades'].empty:
        with st.expander(f"💱 Operazioni ({len(r['trades'])})"):
            st.dataframe(r['trades'], width="stretch", hide_index=True)


def render_pac_backtest_ui():
    """Modalità 🧭 della pagina 💹 Backtest: Confronto Strategie PAC
    (backtest del consiglio mensile dei Satelliti PAC)."""
    st.subheader("🏁 Confronto Strategie PAC (satelliti)")
    st.caption(
        "Replay storico del consiglio mensile: ogni mese versa l'importo scelto e "
        "compra il miglior ETF secondo la strategia. Le strategie 🧭 PAC-native "
        "(in cima) decidono al ribilanciamento mensile e reinvestiscono i proventi "
        "delle vendite nel core; le strategie di trading giornaliero (in coda) "
        "mantengono stop e filtri giornalieri, con i proventi in cassa.")

    with st.expander("📖 Strategie PAC-native (come funzionano)"):
        for k in PAC_STRATEGY_ORDER:
            st.markdown(f"**{PAC_STRATEGIES[k].name}**")
            st.markdown(PAC_STRATEGIES[k].description)
        st.caption(
            "Le strategie di trading giornaliero (in coda nella lista) restano "
            "selezionabili per confronto: decidono ogni giorno con stop ATR e "
            "filtro di regime, e i proventi delle vendite restano in cassa.")

    with st.expander("⚠️ Come leggere i risultati senza farsi ingannare"):
        st.markdown(
            "**Survivorship bias:** l'universo contiene solo gli ETF quotati **oggi**: "
            "i prodotti chiusi o falliti mancano e i risultati sono sistematicamente "
            "**ottimistici**.\n\n"
            "**Pochi trade:** una decisione al mese = ~12 trade l'anno: il risultato è "
            "dominato da pochi mesi fortunati (path-dependence). Guarda più anni e più "
            "strategie, non il singolo numero finale.\n\n"
            "**Costi e tasse:** commissioni per operazione sì, **nessuna tassa** sulle "
            "plusvalenze (26% in Italia al realizzo, non simulata)."
        )

    universe = etf_universe_in_db()
    if not universe:
        st.warning(
            "Universo ETF non disponibile: esegui la discovery in 📥 Gestione Dati "
            "→ 🌍 Universo ETF (gli ETF devono avere anche lo storico nel DB).")
        return
    st.caption(f"🌍 Universo: **{len(universe)}** ETF/ETC/ETP dal DB locale (discovery).")

    st.sidebar.header("🏁 Confronto PAC")
    monthly_amount = st.sidebar.number_input(
        "💰 Importo PAC mensile (€)", 50.0, 100000.0, 200.0, 50.0, format="%.2f")
    default_start = (pd.Timestamp.now() - pd.DateOffset(years=5)).date()
    start_date = st.sidebar.date_input("📅 Data inizio", value=default_start)
    selected = st.sidebar.multiselect(
        "🧬 Strategie", _ALL_PAC_KEYS, default=PAC_STRATEGY_ORDER,
        format_func=_pac_strategy_name,
        help="In cima le strategie PAC-native (decisione mensile, proventi "
             "reinvestiti nel core); in coda le strategie di trading "
             "giornaliero, mantenute per confronto.")
    core_ticker = st.sidebar.text_input(
        "🛡️ Core (buy & hold)", value="VWCE.MI",
        help="Il benchmark: stesso PAC, tutto sul core, nessuna vendita. Deve essere "
             "nel DB.").strip().upper()
    core_fallback = st.sidebar.checkbox(
        "Mesi senza segnale → PAC al core", value=True,
        help="Quando la strategia non compra un satellite (base della strategia, "
             "regime o nessun candidato), l'importo del mese finisce nel core "
             "invece che in cassa.")
    commission = st.sidebar.number_input(
        "💸 Commissione per operazione (€)", 0.0, 20.0, 1.0, 0.5)

    sig = (tuple(selected), str(start_date), float(monthly_amount), core_ticker,
           bool(core_fallback), float(commission), tuple(universe))

    if st.button("▶️ Avvia Confronto PAC", type="primary", disabled=not selected):
        with st.spinner("Caricamento dati dal DB..."):
            md = prepare_market_data(list(universe) + [core_ticker],
                                     period="max", allow_download=False)
        if not md or not any(t in md for t in universe):
            st.error("Nessun dato disponibile: aggiorna lo storico in 📥 Gestione Dati.")
        else:
            bar = st.progress(0.0, text="Simulazione strategie...")
            try:
                results = run_pac_comparison(
                    universe, strategies=selected,
                    monthly_amount=float(monthly_amount),
                    start_date=str(start_date),
                    commission=float(commission),
                    core_ticker=core_ticker, core_fallback=core_fallback,
                    preloaded_data=md,
                    progress_callback=lambda f: bar.progress(
                        f, text=f"Simulazione strategie... {int(f * 100)}%"))
            finally:
                bar.empty()
            st.session_state['pac_cmp_run'] = {
                'sig': sig, 'results': results,
                'amount': float(monthly_amount),
                'commission': float(commission),
            }

    run = st.session_state.get('pac_cmp_run')
    if run and run.get('sig') == sig and run.get('results'):
        _render_pac_comparison_results(run)
