# -*- coding: utf-8 -*-
"""Componenti di rendering Streamlit condivisi tra le varie pagine UI."""
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from etf_metrics.core.metrics import compute_metrics_from_series, compute_sharpe_ratio


# =====================================================================
# Selettore universo ticker (watchlist manuale o tutto il DB locale)
# =====================================================================

UNIVERSE_WATCHLIST = "✍️ Ticker che inserisco"
UNIVERSE_WHOLE_DB = "📦 Tutto il DB locale"


@st.cache_data(ttl=60, show_spinner=False)
def count_db_tickers() -> int:
    """Numero TOTALE di ticker nel DB locale (senza filtro min_rows)."""
    try:
        from etf_metrics.core.data_manager import MarketDataManager
        return len(MarketDataManager().get_all_tickers())
    except Exception:
        return 0


@st.cache_data(ttl=60, show_spinner=False)
def list_db_tickers(min_rows: int = 200) -> List[str]:
    """Ticker presenti nel DB locale con almeno `min_rows` righe di prezzi
    (storico sufficiente per scanner e backtest). Funziona con entrambi i
    backend (SQLite e DuckDB)."""
    try:
        from etf_metrics.core.data_manager import MarketDataManager
        dm = MarketDataManager()
        return dm.get_tickers_with_min_rows(min_rows)
    except Exception:
        return []


def render_universe_selector(
    default_watchlist: str = "",
    textarea_height: int = 180,
    show_download: bool = False,
    allow_download_default: bool = True,
    min_rows: int = 200,
) -> Tuple[List[str], str, bool]:
    """Selettore condiviso dell'universo ticker (sidebar).

    Due modalità:
    - "✍️ Ticker che inserisco": watchlist manuale, con validazione live contro
      il DB locale (✅ già scaricati / ⚠️ mancanti)
    - "📦 Tutto il DB locale": tutti i ticker già scaricati con storico sufficiente

    Con show_download=True mostra anche la checkbox di download/aggiornamento
    delta da Yahoo (applicabile a entrambe le modalità).

    Ritorna (tickers, universe_mode, allow_download).
    """
    db_tickers = list_db_tickers(min_rows=min_rows)

    mode = st.radio(
        "Quali ticker usare",
        [UNIVERSE_WATCHLIST, UNIVERSE_WHOLE_DB],
        help=("Watchlist: solo i ticker scritti qui sotto. Tutto il DB: l'intero "
              "universo già scaricato nel database locale."),
    )

    allow_download = False
    if mode == UNIVERSE_WHOLE_DB:
        tickers = list(db_tickers)
        if tickers:
            total = count_db_tickers()
            excluded = max(0, total - len(tickers))
            extra = (f"; {excluded} su {total} esclusi: storico troppo corto "
                     f"per gli indicatori (servono ≥ {min_rows} sedute per SMA200/"
                     "Donchian/momentum)") if excluded else ""
            st.caption(
                f"📦 {len(tickers)} ticker già nel DB (≥ {min_rows} righe di "
                f"storico{extra}). "
                "Nessuna lista da scrivere: si usa direttamente ciò che è salvato."
            )
        else:
            st.warning(
                "Il DB locale non contiene ticker con storico sufficiente: "
                "scarica i dati dalla pagina 📥 **Gestione Dati** e riprova.")
    else:
        txt = st.text_area(
            "Watchlist (un ticker per riga)", default_watchlist, height=textarea_height)
        tickers = [t.strip().upper() for t in txt.split('\n') if t.strip()]

        # Validazione live contro il DB
        if tickers and db_tickers:
            db_set = set(db_tickers)
            in_db = [t for t in tickers if t in db_set]
            missing = [t for t in tickers if t not in db_set]
            parts = [f"✅ {len(in_db)}/{len(tickers)} già nel DB."]
            if missing:
                shown = ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
                parts.append(f"⚠️ Non nel DB: {shown}")
            st.caption(" ".join(parts))
        elif tickers and not db_tickers:
            st.caption("ℹ️ Il DB locale è vuoto: i ticker verranno scaricati da Yahoo.")

    if show_download:
        allow_download = st.checkbox(
            "📥 Aggiorna da Yahoo i dati mancanti o vecchi",
            value=allow_download_default,
            help=("Download con logica delta: completo solo per i ticker nuovi, "
                  "aggiornamento incrementale per gli altri. I ticker in cache negativa "
                  "non vengono ritentati; con rate limit (429) si usa il DB locale. "
                  "Disattivalo per lavorare solo offline."),
        )

    return tickers, mode, allow_download


def get_portfolio_store():
    """Restituisce il PortfolioStore conservato in session_state (caricato da JSON al primo uso)."""
    from etf_metrics.core.portfolio_tracker import PortfolioStore
    if "portfolio_store" not in st.session_state:
        st.session_state["portfolio_store"] = PortfolioStore.load_from_json()
    return st.session_state["portfolio_store"]


def style_pnl_columns(val):
    """Colore verde per valori positivi, rosso per negativi."""
    if isinstance(val, (int, float)):
        color = '#28a745' if val > 0 else '#dc3545' if val < 0 else '#333333'
        return f'color: {color}; font-weight: 600;'
    return ''


def style_trend_signal(val: str) -> str:
    """Applica uno stile colorato alla colonna dei segnali di trend."""
    val_lower = val.lower()
    if "mantieni" in val_lower:
        return 'background-color: #28a745; color: white; font-weight: bold;'
    elif "monitora" in val_lower:
        return 'background-color: #ffc107; color: black; font-weight: bold;'
    elif val_lower.startswith("vendi"):
        return 'background-color: #dc3545; color: white; font-weight: bold;'
    return ''


def format_dataframe(
    df: pd.DataFrame,
    column_config: Dict,
    pnl_cols: List[str] = [],
    bar_cols: List[str] = [],
    trend_cols: List[str] = [],
):
    """Applica formattazione completa a un DataFrame per la visualizzazione."""
    df_display = df.rename(columns=column_config)

    styler = df_display.style

    renamed_pnl_cols = [column_config.get(col) for col in pnl_cols if column_config.get(col) in df_display.columns]
    renamed_bar_cols = [column_config.get(col) for col in bar_cols if column_config.get(col) in df_display.columns]
    renamed_trend_cols = [column_config.get(col) for col in trend_cols if column_config.get(col) in df_display.columns]

    format_dict = {
        name: '€{:,.2f}' for col, name in column_config.items() if '€' in name
    }
    format_dict.update({
        name: '{:+.2f}%' for col, name in column_config.items() if '%' in name
    })
    format_dict.update({
        name: '{:.6f}' for col, name in column_config.items() if 'Quantità' in name
    })
    styler = styler.format(format_dict, na_rep="n.d.")

    if renamed_pnl_cols:
        styler = styler.apply(lambda x: x.map(style_pnl_columns), subset=renamed_pnl_cols)

    if renamed_trend_cols:
        for col_name in renamed_trend_cols:
            styler = styler.map(style_trend_signal, subset=[col_name])

    if renamed_bar_cols:
        for col_name in renamed_bar_cols:
            styler = styler.background_gradient(cmap='viridis_r', subset=[col_name])

    styler = styler.set_properties(**{'text-align': 'right'}).set_properties(
        subset=[column_config.get('isin', 'isin')], **{'text-align': 'left'}
    )

    return styler


def render_allocation_pie(labels: List[str], values: List[float], title: str, key="alloc_pie"):
    """Renderizza un grafico a torta dell'allocazione."""
    if not labels or not values or sum(values) <= 0:
        st.info("Aggiungi almeno una posizione per vedere l'allocazione.")
        return
    if HAS_PLOTLY:
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo="label+percent", pull=[0.05] * len(labels))]
        )
        fig.update_layout(
            title_text=title,
            margin=dict(t=50, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        )
        st.plotly_chart(fig, width="stretch", key=key)


def display_backtest_results(all_series: Dict[str, pd.Series], rf_ann: float, key_prefix="backtest"):
    """Visualizza l'andamento e le metriche di un set di serie di portafoglio."""
    if not all_series:
        st.error("Nessun dato da visualizzare.")
        return
    period_map = {"1M": 1, "3M": 3, "6M": 6, "YTD": "ytd", "1A": 12, "3A": 36, "5A": 60, "Max": None}
    selected_period_label = st.radio(
        "Seleziona periodo di analisi",
        list(period_map.keys()),
        index=len(period_map) - 1,
        horizontal=True,
        key=f"{key_prefix}_period_radio",
    )
    lookback_months = period_map[selected_period_label]
    end_date = pd.to_datetime('today').normalize()
    if lookback_months == "ytd":
        start_date = pd.to_datetime(f"{end_date.year}-01-01")
    elif lookback_months is not None:
        start_date = end_date - pd.DateOffset(months=lookback_months)
    else:
        start_date = min(s.index.min() for s in all_series.values() if s is not None and not s.empty)

    metrics_list, series_to_plot = [], {}
    for name, series in all_series.items():
        if series is None or series.empty:
            continue
        filtered = series[series.index >= start_date]
        if filtered.shape[0] < 2:
            continue
        series_to_plot[name] = filtered
        metrics = compute_metrics_from_series(filtered)
        metrics["sharpe"] = compute_sharpe_ratio(filtered, rf_ann)
        metrics["name"] = name
        metrics_list.append(metrics)

    st.subheader(f"Andamento Portafogli ({selected_period_label})")
    st.line_chart(series_to_plot)

    st.subheader(f"Metriche di Performance ({selected_period_label})")
    if metrics_list:
        df = pd.DataFrame(metrics_list).set_index("name")[["cagr", "vol_ann", "mdd", "sharpe"]]
        df.columns = ["CAGR %", "Volatilità Ann. %", "Max Drawdown %", "Sharpe Ratio"]
        st.dataframe(df.style.format("{:.2f}", na_rep="n.d."), width="stretch")
