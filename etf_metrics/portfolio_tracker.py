# -*- coding: utf-8 -*-
"""
PortfolioTracker: utility per gestire salvataggi, P&L e confronto benchmark
per il Portfolio Tracker della UI Streamlit.

API attese dalla UI (vedi ui_portfolio_tracker_app.py):
- init_session_from_json_once(filename="saved_portfolio.json")
- get_saved_portfolio_names()
- load_portfolio_from_session(name)
- save_portfolio_to_session(portfolio_data)
- save_portfolio_to_json(portfolio_data, filename="saved_portfolio.json")
- calculate_portfolio_pnl(holdings)
- get_performance_vs_benchmark(positions)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Any

import pandas as pd
import streamlit as st

from .yahoo_client import resolve_isin_one, get_series

logger = logging.getLogger(__name__)


class PortfolioTracker:
    # --- Gestione storage in sessione/file ---------------------------------

    @staticmethod
    def init_session_from_json_once(filename: str = "saved_portfolio.json") -> None:
        """
        Carica una volta i portafogli salvati da file in sessione.
        La sessione mantiene un dict: {'nome_portafoglio': {...}, ...}
        """
        if st.session_state.get("_pt_loaded_once"):
            return

        st.session_state.setdefault("_pt_portfolios", {})
        path = Path(filename)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    st.session_state["_pt_portfolios"].update(data)
            except Exception as e:
                logger.warning("Impossibile leggere %s: %s", filename, e)

        st.session_state["_pt_loaded_once"] = True

    @staticmethod
    def get_saved_portfolio_names() -> List[str]:
        return list(st.session_state.get("_pt_portfolios", {}).keys())

    @staticmethod
    def load_portfolio_from_session(name: str) -> Optional[Dict[str, Any]]:
        return st.session_state.get("_pt_portfolios", {}).get(name)

    @staticmethod
    def save_portfolio_to_session(portfolio_data: Dict[str, Any]) -> None:
        """
        Salva/aggiorna il portafoglio nella sessione, indicizzato per nome.
        Atteso formato:
        {
          'name': str,
          'holdings': [{'isin':..., 'quantity':..., 'purchase_price':..., 'purchase_date': date}, ...],
          'created_date': date
        }
        """
        name = (portfolio_data or {}).get("name") or "Portafoglio"
        st.session_state.setdefault("_pt_portfolios", {})
        st.session_state["_pt_portfolios"][name] = portfolio_data

    @staticmethod
    def save_portfolio_to_json(portfolio_data: Dict[str, Any], filename: str = "saved_portfolio.json") -> None:
        """
        Salva il portafoglio anche su file (merge con l'esistente).
        """
        try:
            path = Path(filename)
            current = {}
            if path.exists():
                try:
                    current = json.loads(path.read_text(encoding="utf-8")) or {}
                except Exception:
                    current = {}
            name = (portfolio_data or {}).get("name") or "Portafoglio"
            current[name] = portfolio_data
            path.write_text(json.dumps(current, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Errore salvataggio JSON %s: %s", filename, e)

    # --- Calcolo P&L --------------------------------------------------------

    @staticmethod
    def _to_date(d: Any) -> date:
        if isinstance(d, date):
            return d
        try:
            return pd.to_datetime(d).date()
        except Exception:
            return datetime.now().date()

    @staticmethod
    def calculate_portfolio_pnl(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcola P&L complessivo e dettaglio posizioni.
        Ogni holding attesa come:
        {'isin': str, 'quantity': float, 'purchase_price': float, 'purchase_date': date|str}
        """
        if not holdings:
            return {
                'total_invested': 0.0,
                'current_value': 0.0,
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
                'holdings_detail': []
            }

        details: List[Dict[str, Any]] = []
        total_invested = 0.0
        total_current = 0.0
        today = datetime.now().date()

        for h in holdings:
            isin = (h.get("isin") or "").strip().upper()
            qty = float(h.get("quantity", 0.0))
            buy_price = float(h.get("purchase_price", 0.0))
            buy_date = PortfolioTracker._to_date(h.get("purchase_date"))

            if not isin or qty <= 0 or buy_price <= 0:
                continue
            ticker = resolve_isin_one(isin) or ""
            # per il prezzo corrente basta un periodo breve
            series = get_series(ticker, "6mo") if ticker else None
            current_price = float(series.iloc[-1]) if (series is not None and not series.empty) else buy_price

            invested_amount = qty * buy_price
            current_amount = qty * current_price
            pnl = current_amount - invested_amount
            pnl_pct = (pnl / invested_amount * 100.0) if invested_amount > 0 else 0.0
            days_held = (today - buy_date).days if buy_date else 0
            details.append({
                "isin": isin,
                "quantity": qty,
                "purchase_price": buy_price,
                "current_price": current_price,
                "invested_amount": invested_amount,
                "current_amount": current_amount,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "days_held": days_held,
            })

            total_invested += invested_amount
            total_current += current_amount

        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100.0) if total_invested > 0 else 0.0

        return {
            'total_invested': round(total_invested, 2),
            'current_value': round(total_current, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'holdings_detail': details
        }

    # --- Confronto vs benchmark (S&P 500) -----------------------------------

    @staticmethod
    def get_performance_vs_benchmark(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcola una misura semplice di performance del portafoglio rispetto a S&P 500.
        - Portfolio performance: (Valore attuale / Investito - 1) * 100
        - Benchmark performance: variazione ^GSPC da min(purchase_date) a oggi
        """
        try:
            # Portfolio performance come P&L complessivo
            pnl = PortfolioTracker.calculate_portfolio_pnl(positions)
            portfolio_perf = float(pnl.get("total_pnl_pct", 0.0))

            # Intervallo temporale di riferimento
            if not positions:
                return {
                    "portfolio_performance": portfolio_perf,
                    "benchmark_performance": 0.0,
                    "outperformance": portfolio_perf,
                }
            start = min(PortfolioTracker._to_date(p.get("purchase_date")) for p in positions)
            end = datetime.now().date()

            # Serie S&P 500 (^GSPC) e calcolo performance
            bench_series = get_series("^GSPC", "max")
            if bench_series is None or bench_series.empty:
                return {"error": "Dati benchmark non disponibili"}

            bench_window = bench_series.loc[pd.to_datetime(start): pd.to_datetime(end)]
            if bench_window.shape[0] < 2:
                return {"error": "Dati benchmark insufficienti per l'intervallo selezionato"}

            bench_perf = (float(bench_window.iloc[-1]) / float(bench_window.iloc[0]) - 1.0) * 100.0
            outperf = portfolio_perf - bench_perf

            return {
                "portfolio_performance": round(portfolio_perf, 2),
                "benchmark_performance": round(bench_perf, 2),
                "outperformance": round(outperf, 2),
            }
        except Exception as e:
            logger.warning("Errore nel confronto con benchmark: %s", e)
            return {"error": "Impossibile calcolare il benchmark."}
