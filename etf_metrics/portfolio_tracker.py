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
        # Salva automaticamente dopo ogni transazione
        PortfolioTracker.save_all_portfolios_to_json()

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
    def calculate_capital_gains(transactions: List[Dict[str, Any]], commission: float = 1.0, tax_rate: float = 26.0) -> \
    List[Dict[str, Any]]:
        """Calcola plusvalenze e minusvalenze per le vendite con la logica corretta."""
        sales_gains = []
        positions = defaultdict(lambda: {'quantity': 0, 'total_cost': 0})
        sorted_trans = sorted(transactions, key=lambda x: x['date'])

        for t in sorted_trans:
            isin = t['isin']
            qty = float(t['quantity'])
            price = float(t['price'])

            if t['type'] == 'buy':
                positions[isin]['quantity'] += qty
                positions[isin]['total_cost'] += qty * price
            elif t['type'] == 'sell':
                if positions[isin]['quantity'] < qty:
                    continue

                avg_buy_price = positions[isin]['total_cost'] / positions[isin]['quantity']
                cost_of_sold_shares = qty * avg_buy_price
                sale_revenue = qty * price

                # 1. Calcola la plusvalenza lorda (base imponibile)
                capital_gain = sale_revenue - cost_of_sold_shares

                # 2. Calcola le imposte su questa plusvalenza
                taxable_amount = capital_gain if capital_gain > 0 else 0
                tax_paid = taxable_amount * (tax_rate / 100.0)

                # 3. Calcola il P&L lordo (dopo commissioni ma prima delle tasse)
                gross_pnl_after_commission = capital_gain - commission

                # 4. Calcola il P&L netto finale
                net_pnl = gross_pnl_after_commission - tax_paid

                sales_gains.append({
                    'date': t['date'],
                    'isin': isin,
                    'quantity': qty,
                    'sale_price': price,
                    'avg_buy_price': avg_buy_price,
                    'gross_pnl': gross_pnl_after_commission,
                    'commission': commission,
                    'taxable_amount': taxable_amount,
                    'tax_paid': tax_paid,
                    'net_pnl': net_pnl
                })

                positions[isin]['quantity'] -= qty
                positions[isin]['total_cost'] -= cost_of_sold_shares

        return sales_gains

    @staticmethod
    def calculate_portfolio_pnl(transactions: List[Dict[str, Any]], commission: float = 1.0, tax_rate: float = 26.0) -> \
    Dict[str, Any]:
        if not transactions:
            return defaultdict(float, {'open_positions': [], 'aggregated_positions': [], 'capital_gains': []})

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

        # Calcolo posizioni aggregate
        aggregated_positions = []
        all_isins = set(t['isin'] for t in transactions)
        for isin in all_isins:
            total_bought_qty = sum(t['quantity'] for t in transactions if t['isin'] == isin and t['type'] == 'buy')
            total_bought_value = sum(
                t['quantity'] * t['price'] for t in transactions if t['isin'] == isin and t['type'] == 'buy')
            total_sold_qty = sum(t['quantity'] for t in transactions if t['isin'] == isin and t['type'] == 'sell')
            total_sold_value = sum(
                t['quantity'] * t['price'] for t in transactions if t['isin'] == isin and t['type'] == 'sell')

            if total_bought_qty > 0:
                avg_buy_price = total_bought_value / total_bought_qty
            else:
                avg_buy_price = 0

            current_position = next((p for p in open_positions_details if p['isin'] == isin), None)

            if current_position:
                current_qty = current_position['quantity']
                current_value = current_position['current_amount']
                unrealized_pnl = current_position['unrealized_pnl']
            else:
                current_qty = 0
                current_value = 0
                unrealized_pnl = 0

            realized_pnl_isin = total_sold_value - (total_sold_qty * avg_buy_price if total_bought_qty > 0 else 0)

            aggregated_positions.append({
                'isin': isin,
                'total_bought_qty': total_bought_qty,
                'total_sold_qty': total_sold_qty,
                'current_qty': current_qty,
                'avg_buy_price': avg_buy_price,
                'realized_pnl': realized_pnl_isin,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': realized_pnl_isin + unrealized_pnl,
                'current_value': current_value,
            })

        capital_gains = PortfolioTracker.calculate_capital_gains(transactions, commission, tax_rate)

        return {
            'net_invested': net_invested,
            'current_value': current_total_value,
            'realized_pnl': realized_pnl,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / initial_investment * 100) if initial_investment > 0 else 0,
            'open_positions': open_positions_details,
            'aggregated_positions': aggregated_positions,
            'capital_gains': capital_gains
        }