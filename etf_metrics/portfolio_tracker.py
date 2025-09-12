# -*- coding: utf-8 -*-
"""
Portfolio Tracker - Traccia plusvalenze e minusvalenze del portafoglio personale
"""
# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional
from datetime import date
import json
import os
import streamlit as st

from .yahoo_client import get_series, resolve_isin_one
logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Tracker per portafoglio personale con calcolo P&L."""

    @staticmethod
    def save_portfolio_to_session(portfolio_data: Dict) -> None:
        """Salva il portafoglio nella sessione Streamlit."""
        if 'saved_portfolios' not in st.session_state:
            st.session_state.saved_portfolios = {}
        portfolio_name = portfolio_data.get('name', f'Portfolio_{len(st.session_state.saved_portfolios)}')
        st.session_state.saved_portfolios[portfolio_name] = portfolio_data

    @staticmethod
    def load_portfolio_from_session(portfolio_name: str) -> Optional[Dict]:
        """Carica un portafoglio dalla sessione."""
        if 'saved_portfolios' in st.session_state:
            return st.session_state.saved_portfolios.get(portfolio_name)
        return None

    @staticmethod
    def get_saved_portfolio_names() -> List[str]:
        """Restituisce i nomi dei portafogli salvati."""
        if 'saved_portfolios' in st.session_state:
            return list(st.session_state.saved_portfolios.keys())
        return []

    @staticmethod
    def save_portfolio_to_json(portfolio_data: Dict, filename: str = "saved_portfolio.json") -> None:
        """Salva il portafoglio in un file JSON."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(portfolio_data, f, default=str, indent=4)
        except Exception as e:
            logger.error(f"Errore nel salvataggio del portafoglio su JSON: {e}")

    @staticmethod
    def load_portfolio_from_json(filename: str = "saved_portfolio.json") -> Optional[Dict]:
        """Carica il portafoglio da un file JSON."""
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Errore nel caricamento del portafoglio da JSON: {e}")
        return None

    @staticmethod
    def init_session_from_json_once(filename: str = "saved_portfolio.json") -> None:
        """
        Inizializza la sessione Streamlit da file JSON una sola volta.
        (Evita side-effect all'import del modulo.)
        """
        if "current_portfolio" not in st.session_state:
            loaded_portfolio = PortfolioTracker.load_portfolio_from_json(filename)
            if loaded_portfolio:
                st.session_state.current_portfolio = loaded_portfolio
                st.session_state.sb_name = loaded_portfolio.get("name", "Il Mio Portafoglio")
                st.session_state.sb_npos = len(loaded_portfolio.get("holdings", []))
                for i, h in enumerate(loaded_portfolio.get("holdings", [])):
                    st.session_state[f"sb_isin_{i}"] = h.get("isin", "")
                    st.session_state[f"sb_qty_{i}"] = h.get("quantity", 0.0)
                    st.session_state[f"sb_price_{i}"] = h.get("purchase_price", 0.0)
                    # Se purchase_date è stringa ISO
                    pd_str = h.get("purchase_date")
                    if isinstance(pd_str, str):
                        try:
                            st.session_state[f"sb_date_{i}"] = date.fromisoformat(pd_str)
                        except Exception:
                            st.session_state[f"sb_date_{i}"] = date.today()

    @staticmethod
    def calculate_portfolio_pnl(holdings: List[Dict], current_date: date = None) -> Dict:
        """
        Calcola P&L del portafoglio.
        Args:
            holdings: Lista di {'isin': str, 'quantity': float, 'purchase_price': float, 'purchase_date': date}
            current_date: Data di valutazione (default: oggi)
        """
        if current_date is None:
            current_date = date.today()

        results = {
            'total_invested': 0.0,
            'current_value': 0.0,
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0,
            'holdings_detail': []
        }

        for holding in holdings:
            try:
                isin = holding['isin']
                quantity = float(holding['quantity'])
                purchase_price = float(holding['purchase_price'])
                purchase_date = holding['purchase_date']

                ticker = resolve_isin_one(isin)
                if not ticker:
                    logger.warning(f"Ticker non trovato per ISIN: {isin}")
                    continue

                current_series = None
                periods_to_try = ("5d", "1mo", "3mo", "6mo")
                for period in periods_to_try:
                    current_series = get_series(ticker, period)
                    if current_series is not None and len(current_series) >= 2:
                        break

                if current_series is None or current_series.empty or len(current_series) < 2:
                    logger.warning(f"Prezzo corrente non disponibile o insufficiente per {ticker}")
                    continue

                current_price = float(current_series.iloc[-1])

                invested_amount = quantity * purchase_price
                current_amount = quantity * current_price
                pnl = current_amount - invested_amount
                pnl_pct = (pnl / invested_amount) * 100 if invested_amount > 0 else 0
                days_held = (current_date - purchase_date).days if isinstance(purchase_date, date) else 0

                holding_detail = {
                    'isin': isin,
                    'ticker': ticker,
                    'quantity': quantity,
                    'purchase_price': purchase_price,
                    'current_price': current_price,
                    'invested_amount': invested_amount,
                    'current_amount': current_amount,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held,
                    'purchase_date': purchase_date
                }
                results['holdings_detail'].append(holding_detail)
                results['total_invested'] += invested_amount
                results['current_value'] += current_amount
            except Exception as e:
                logger.error(f"Errore nel calcolo P&L per holding {holding}: {e}")
                continue
        results['total_pnl'] = results['current_value'] - results['total_invested']
        if results['total_invested'] > 0:
            results['total_pnl_pct'] = (results['total_pnl'] / results['total_invested']) * 100
        return results

    @staticmethod
    def get_performance_vs_benchmark(holdings: List[Dict], benchmark_ticker: str = "^GSPC") -> Dict:
        """Confronta la performance del portafoglio vs benchmark."""
        try:
            min_date = min(h['purchase_date'] for h in holdings if isinstance(h['purchase_date'], date))
            total_weight = 0.0
            weighted_performance = 0.0

            for holding in holdings:
                ticker = resolve_isin_one(holding['isin'])
                if not ticker:
                    continue
                period = "max"
                series = get_series(ticker, period)
                if series is None:
                    continue
                series_from_purchase = series[series.index.date >= holding['purchase_date']]
                if len(series_from_purchase) < 2:
                    continue

                performance = ((series_from_purchase.iloc[-1] / series_from_purchase.iloc[0]) - 1) * 100.0
                weight = float(holding['quantity']) * float(holding['purchase_price'])
                weighted_performance += performance * weight
                total_weight += weight

            if total_weight == 0:
                return {'error': 'Impossibile calcolare performance portafoglio'}

            portfolio_performance = weighted_performance / total_weight

            benchmark_series = get_series(benchmark_ticker, "max")
            if benchmark_series is not None:
                benchmark_from_date = benchmark_series[benchmark_series.index.date >= min_date]
                if len(benchmark_from_date) >= 2:
                    benchmark_performance = ((benchmark_from_date.iloc[-1] / benchmark_from_date.iloc[0]) - 1) * 100.0
                    return {
                        'portfolio_performance': portfolio_performance,
                        'benchmark_performance': benchmark_performance,
                        'outperformance': portfolio_performance - benchmark_performance,
                        'benchmark_name': benchmark_ticker
                    }
            return {'portfolio_performance': portfolio_performance}
        except Exception as e:
            logger.error(f"Errore nel calcolo performance vs benchmark: {e}")
            return {'error': str(e)}