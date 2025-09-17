# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
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

        # --- 1. Calcolo posizioni aperte e P&L non realizzato ---
        positions = defaultdict(lambda: {'quantity': 0, 'total_cost': 0})
        for t in sorted(transactions, key=lambda x: x['date']):
            isin = t['isin']
            qty = float(t['quantity'])
            price = float(t['price'])
            if t['type'] == 'buy':
                positions[isin]['quantity'] += qty
                positions[isin]['total_cost'] += qty * price
            elif t['type'] == 'sell':
                if positions[isin]['quantity'] > 0:
                    avg_buy_price = positions[isin]['total_cost'] / positions[isin]['quantity']
                    cost_of_sold_shares = qty * avg_buy_price
                    positions[isin]['quantity'] -= qty
                    positions[isin]['total_cost'] -= cost_of_sold_shares

        open_positions_details = []
        for isin, data in positions.items():
            if data['quantity'] > 0.0000001:
                ticker = resolve_isin_one(isin)
                series = get_series(ticker, "1mo") if ticker else None
                current_price = series.iloc[-1] if (series is not None and not series.empty) else 0
                current_amount = data['quantity'] * current_price
                invested_amount = data['total_cost']
                unrealized_pnl = current_amount - invested_amount
                open_positions_details.append({
                    'isin': isin, 'quantity': data['quantity'],
                    'avg_buy_price': invested_amount / data['quantity'] if data['quantity'] > 0 else 0,
                    'current_price': current_price, 'invested_amount': invested_amount,
                    'current_amount': current_amount, 'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': (unrealized_pnl / invested_amount * 100) if invested_amount > 0 else 0
                })

        # --- 2. Calcolo P&L realizzato dal Cassetto Fiscale ---
        capital_gains = PortfolioTracker.calculate_capital_gains(transactions, commission, tax_rate)
        realized_pnl_net = sum(cg['net_pnl'] for cg in capital_gains)

        # --- 3. Calcolo metriche di riepilogo secondo la nuova logica ---
        invested_net_open = sum(p['invested_amount'] for p in open_positions_details)
        current_value_open = sum(p['current_amount'] for p in open_positions_details)
        unrealized_pnl_open = sum(p['unrealized_pnl'] for p in open_positions_details)

        total_pnl = unrealized_pnl_open + realized_pnl_net

        total_capital_invested = sum(t['quantity'] * t['price'] for t in transactions if t['type'] == 'buy')

        # --- 4. Calcolo posizioni aggregate (mantiene la logica precedente per la tabella di dettaglio) ---
        aggregated_positions = []
        all_isins = set(t['isin'] for t in transactions)
        for isin in all_isins:
            total_bought_qty = sum(t['quantity'] for t in transactions if t['isin'] == isin and t['type'] == 'buy')
            total_bought_value = sum(
                t['quantity'] * t['price'] for t in transactions if t['isin'] == isin and t['type'] == 'buy')
            total_sold_qty = sum(t['quantity'] for t in transactions if t['isin'] == isin and t['type'] == 'sell')
            total_sold_value = sum(
                t['quantity'] * t['price'] for t in transactions if t['isin'] == isin and t['type'] == 'sell')
            avg_buy_price = total_bought_value / total_bought_qty if total_bought_qty > 0 else 0
            current_pos = next((p for p in open_positions_details if p['isin'] == isin), None)

            realized_pnl_isin_gross = total_sold_value - (total_sold_qty * avg_buy_price)

            aggregated_positions.append({
                'isin': isin, 'total_bought_qty': total_bought_qty, 'total_sold_qty': total_sold_qty,
                'current_qty': current_pos['quantity'] if current_pos else 0,
                'avg_buy_price': avg_buy_price,
                'realized_pnl': realized_pnl_isin_gross,
                'unrealized_pnl': current_pos['unrealized_pnl'] if current_pos else 0,
                'total_pnl': realized_pnl_isin_gross + (current_pos['unrealized_pnl'] if current_pos else 0),
                'current_value': current_pos['current_amount'] if current_pos else 0,
            })

        return {
            'net_invested': invested_net_open,
            'current_value': current_value_open,
            'realized_pnl': realized_pnl_net,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / total_capital_invested * 100) if total_capital_invested > 0 else 0,
            'open_positions': open_positions_details,
            'aggregated_positions': aggregated_positions,
            'capital_gains': capital_gains
        }