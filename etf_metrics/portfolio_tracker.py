# -*- coding: utf-8 -*-
"""
    # Ensure the date range is valid for yfinance to avoid fetching only 1 observation
    if start_date >= end_date:
        start_date = end_date - pd.DateOffset(days=1)

Portfolio Tracker - Traccia plusvalenze e minusvalenze del portafoglio personale
"""
import logging
from typing import Dict, List, Optional
from datetime import date
import streamlit as st
from .yahoo_client import get_series, resolve_isin_one
import logger

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

                # Risolvi ISIN in ticker
                ticker = resolve_isin_one(isin)
                if not ticker:
                    logger.warning(f"Ticker non trovato per ISIN: {isin}")
                    continue

                # Ottieni prezzo corrente con periodi di fallback
                current_series = None
                periods_to_try = ["5d", "1mo", "3mo", "6mo"]  # Prova periodi più lunghi se necessario

                for period in periods_to_try:
                    current_series = get_series(ticker, period)
                    if current_series is not None and len(current_series) >= 2:
                        break

                if current_series is None or current_series.empty:
                    logger.warning(f"Prezzo corrente non disponibile per {ticker}")
                    continue

                # Verifica che abbiamo abbastanza dati
                if len(current_series) < 2:
                    logger.warning(f"Dati insufficienti per {ticker}: {len(current_series)} osservazioni")
                    continue

                current_price = float(current_series.iloc[-1])

                # Calcola P&L per questa posizione
                invested_amount = quantity * purchase_price
                current_amount = quantity * current_price
                pnl = current_amount - invested_amount
                pnl_pct = (pnl / invested_amount) * 100 if invested_amount > 0 else 0

                # Calcola giorni di holding
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

        # Calcola totali
        results['total_pnl'] = results['current_value'] - results['total_invested']
        if results['total_invested'] > 0:
            results['total_pnl_pct'] = (results['total_pnl'] / results['total_invested']) * 100

        return results

    @staticmethod
    def get_performance_vs_benchmark(holdings: List[Dict], benchmark_ticker: str = "^GSPC") -> Dict:
        """Confronta la performance del portafoglio vs benchmark."""
        try:
            # Calcola data minima di acquisto
            min_date = min(h['purchase_date'] for h in holdings if isinstance(h['purchase_date'], date))

            # Simula portafoglio pesato
            total_weight = 0
            weighted_performance = 0

            for holding in holdings:
                ticker = resolve_isin_one(holding['isin'])
                if not ticker:
                    continue

                # Ottieni serie dal purchase_date
                period = "max"  # Prendi tutto lo storico disponibile
                series = get_series(ticker, period)
                if series is None:
                    continue

                # Filtra dalla data di acquisto
                series_from_purchase = series[series.index.date >= holding['purchase_date']]
                if len(series_from_purchase) < 2:
                    continue

                # Calcola performance
                performance = ((series_from_purchase.iloc[-1] / series_from_purchase.iloc[0]) - 1) * 100
                weight = holding['quantity'] * holding['purchase_price']

                weighted_performance += performance * weight
                total_weight += weight

            if total_weight == 0:
                return {'error': 'Impossibile calcolare performance portafoglio'}

            portfolio_performance = weighted_performance / total_weight

            # Performance benchmark
            benchmark_series = get_series(benchmark_ticker, period)
            if benchmark_series is not None:
                benchmark_from_date = benchmark_series[benchmark_series.index.date >= min_date]
                if len(benchmark_from_date) >= 2:
                    benchmark_performance = ((benchmark_from_date.iloc[-1] / benchmark_from_date.iloc[0]) - 1) * 100

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