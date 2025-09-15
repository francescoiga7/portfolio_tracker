# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from datetime import date
from typing import Dict, List, Any
import pandas as pd
import streamlit as st
from collections import defaultdict

from .yahoo_client import resolve_isin_one, get_series

logger = logging.getLogger(__name__)


class PortfolioTracker:
    # --- Gestione storage in sessione/file ---
    @staticmethod
    def init_session_from_json_once(filename: str = "saved_portfolio.json") -> None:
        if st.session_state.get("_pt_loaded_once"): return
        st.session_state.setdefault("_pt_portfolios", {})
        path = Path(filename)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8")) or {}
                st.session_state["_pt_portfolios"].update(data)
            except Exception as e:
                logger.warning("Impossibile leggere %s: %s", filename, e)
        st.session_state["_pt_loaded_once"] = True

    @staticmethod
    def get_saved_portfolio_names() -> List[str]:
        return list(st.session_state.get("_pt_portfolios", {}).keys())

    @staticmethod
    def load_portfolio_from_session(name: str) -> Dict[str, Any]:
        return st.session_state.get("_pt_portfolios", {}).get(name, {"name": name, "transactions": []})

    @staticmethod
    def add_transaction_to_portfolio(name: str, transaction: Dict[str, Any]):
        portfolios = st.session_state.get("_pt_portfolios", {})
        if name not in portfolios:
            portfolios[name] = {"name": name, "transactions": []}
        # Converte la data per la serializzazione JSON
        transaction['date'] = transaction['date'].isoformat()
        portfolios[name]["transactions"].append(transaction)
        st.session_state["_pt_portfolios"] = portfolios

    @staticmethod
    def save_all_portfolios_to_json(filename: str = "saved_portfolio.json") -> None:
        try:
            path = Path(filename)
            all_data = st.session_state.get("_pt_portfolios", {})
            path.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Errore salvataggio JSON %s: %s", filename, e)

    # --- Calcolo P&L basato su transazioni ---
    @staticmethod
    def calculate_portfolio_pnl(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not transactions:
            return defaultdict(float, {'open_positions': []})

        # Strutture dati per il calcolo
        positions = defaultdict(lambda: {'quantity': 0, 'total_cost': 0, 'realized_pnl': 0})
        realized_pnl = 0
        net_invested = 0

        # Ordina le transazioni per data
        sorted_trans = sorted(transactions, key=lambda x: x['date'])

        for t in sorted_trans:
            isin = t['isin']
            qty = float(t['quantity'])
            price = float(t['price'])

            if t['type'] == 'buy':
                positions[isin]['quantity'] += qty
                positions[isin]['total_cost'] += qty * price
                net_invested += qty * price

            elif t['type'] == 'sell':
                if positions[isin]['quantity'] < qty:
                    logger.warning(f"Vendita allo scoperto non supportata per {isin}.")
                    continue

                avg_buy_price = positions[isin]['total_cost'] / positions[isin]['quantity']
                cost_of_sold_shares = qty * avg_buy_price

                sale_pnl = (qty * price) - cost_of_sold_shares
                realized_pnl += sale_pnl

                positions[isin]['quantity'] -= qty
                positions[isin]['total_cost'] -= cost_of_sold_shares
                net_invested -= qty * price  # Il capitale ritorna dalla vendita

        # Calcolo P&L non realizzato sulle posizioni aperte
        open_positions_details = []
        current_total_value = 0

        for isin, data in positions.items():
            if data['quantity'] > 0.001:  # Tolleranza per floating point
                ticker = resolve_isin_one(isin)
                series = get_series(ticker, "1mo") if ticker else None
                current_price = series.iloc[-1] if (series is not None and not series.empty) else 0

                current_amount = data['quantity'] * current_price
                invested_amount = data['total_cost']
                unrealized_pnl = current_amount - invested_amount

                open_positions_details.append({
                    'isin': isin,
                    'quantity': data['quantity'],
                    'avg_buy_price': invested_amount / data['quantity'],
                    'current_price': current_price,
                    'invested_amount': invested_amount,
                    'current_amount': current_amount,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': (unrealized_pnl / invested_amount * 100) if invested_amount > 0 else 0
                })
                current_total_value += current_amount

        total_pnl = realized_pnl + sum(p['unrealized_pnl'] for p in open_positions_details)
        initial_investment = sum(t['quantity'] * t['price'] for t in sorted_trans if t['type'] == 'buy')

        return {
            'net_invested': net_invested,
            'current_value': current_total_value,
            'realized_pnl': realized_pnl,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / initial_investment * 100) if initial_investment > 0 else 0,
            'open_positions': open_positions_details
        }