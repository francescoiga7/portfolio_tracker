# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, DefaultDict
from collections import defaultdict
import pandas as pd
import streamlit as st

from etf_metrics.clients.yahoo_client import resolve_isin_one, get_series
from etf_metrics.core.trading import check_satellite_status

logger = logging.getLogger(__name__)


def _calculate_open_positions(transactions: List[Dict[str, Any]]) -> DefaultDict[str, Dict[str, Any]]:
    """Calcola le quantità e i costi totali. Tiene traccia della data del primo acquisto attivo."""
    positions = defaultdict(lambda: {'quantity': 0, 'total_cost': 0, 'satellite': False, 'first_buy_date': None})
    sorted_trans = sorted(transactions, key=lambda x: x['date'])

    for t in sorted_trans:
        isin = t['isin']
        qty = float(t['quantity'])
        price = float(t['price'])
        t_date = pd.to_datetime(t['date'])

        if t['type'] == 'buy':
            if positions[isin]['quantity'] == 0:
                positions[isin]['first_buy_date'] = t_date

            positions[isin]['quantity'] += qty
            positions[isin]['total_cost'] += qty * price
            if t.get('satellite', False):
                positions[isin]['satellite'] = True

        elif t['type'] == 'sell':
            if positions[isin]['quantity'] > 0:
                avg_buy_price = positions[isin]['total_cost'] / positions[isin]['quantity']
                sell_qty = min(qty, positions[isin]['quantity'])
                cost_of_sold_shares = sell_qty * avg_buy_price
                positions[isin]['quantity'] -= sell_qty

                if positions[isin]['quantity'] < 1e-9:
                    positions[isin]['quantity'] = 0
                    positions[isin]['total_cost'] = 0
                    positions[isin]['first_buy_date'] = None
                else:
                    positions[isin]['total_cost'] -= cost_of_sold_shares
                    if positions[isin]['total_cost'] < 0: positions[isin]['total_cost'] = 0

    return positions


def _calculate_realized_pnl(transactions: List[Dict[str, Any]], commission: float, tax_rate: float) -> List[
    Dict[str, Any]]:
    """Calcola plusvalenze e minusvalenze per ogni operazione di vendita."""
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

            capital_gain = sale_revenue - cost_of_sold_shares
            taxable_amount = max(0, capital_gain)
            tax_paid = taxable_amount * (tax_rate / 100.0)

            gross_pnl_after_commission = capital_gain - commission
            net_pnl = gross_pnl_after_commission - tax_paid

            sales_gains.append({
                'date': t['date'], 'isin': isin, 'quantity': qty,
                'sale_price': price, 'avg_buy_price': avg_buy_price,
                'gross_pnl': gross_pnl_after_commission, 'commission': commission,
                'taxable_amount': taxable_amount, 'tax_paid': tax_paid, 'net_pnl': net_pnl
            })

            positions[isin]['quantity'] -= qty
            positions[isin]['total_cost'] -= cost_of_sold_shares

    return sales_gains


def _fetch_current_prices_and_series(isins: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Recupera dati OHLCV a 10 anni per calcolare correttamente ATR e massimi.
    """
    data = {}
    fetch_period = "10y"

    for isin in isins:
        ticker = resolve_isin_one(isin)
        series_df = get_series(ticker, fetch_period, as_dataframe=True) if ticker else None

        if series_df is not None and not series_df.empty and 'Close' in series_df.columns:
            current_price = series_df['Close'].iloc[-1]
            data[isin] = {
                "price": current_price,
                "series_df": series_df,
            }
        else:
            data[isin] = {"price": 0, "series_df": None}
            logger.warning(f"Dati incompleti per {isin}")
    return data


def _enrich_open_positions_with_pnl(open_positions: DefaultDict[str, Dict[str, Any]],
                                    market_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Arricchisce posizioni con P&L e logica Satellite/Core.
    """
    enriched = []
    for isin, data in open_positions.items():
        if data['quantity'] > 1e-9:
            mkt = market_data.get(isin, {})
            curr_price = mkt.get("price", 0)
            df_hist = mkt.get("series_df", None)

            invested = data['total_cost']
            curr_val = data['quantity'] * curr_price
            unrealized = curr_val - invested
            unrealized_pct = (unrealized / invested * 100) if invested > 0 else 0

            signal_text = "N/D"
            is_sat = data.get('satellite', False)

            if df_hist is not None:
                if is_sat:
                    purchase_date = data.get('first_buy_date')
                    if purchase_date:
                        res = check_satellite_status(df_hist, purchase_date, data['quantity'])
                        signal_text = res['action']
                        if res['action'] == "SELL":
                            signal_text += f" (Stop: {res.get('stop_price', 0):.2f})"
                    else:
                        signal_text = "HOLD (Data acq. ignota)"
                else:
                    sma200 = df_hist['Close'].rolling(200).mean().iloc[-1]
                    if curr_price > sma200:
                        signal_text = "MANTIENI (Trend Core)"
                    else:
                        signal_text = "MONITORA (Sotto SMA200)"

            enriched.append({
                'isin': isin,
                'quantity': data['quantity'],
                'avg_buy_price': invested / data['quantity'],
                'current_price': curr_price,
                'invested_amount': invested,
                'current_amount': curr_val,
                'unrealized_pnl': unrealized,
                'unrealized_pnl_pct': unrealized_pct,
                'trend_signal': signal_text,
                'satellite': is_sat
            })
    return enriched


def _aggregate_portfolio_summary(open_positions_details, realized_pnl_net, transactions):
    invested_net_open = sum(p['invested_amount'] for p in open_positions_details)
    current_value_open = sum(p['current_amount'] for p in open_positions_details)
    unrealized_pnl_open = sum(p['unrealized_pnl'] for p in open_positions_details)
    total_pnl = unrealized_pnl_open + realized_pnl_net
    total_capital_invested = sum(t['quantity'] * t['price'] for t in transactions if t['type'] == 'buy')
    return {
        'net_invested': invested_net_open,
        'current_value': current_value_open,
        'realized_pnl': realized_pnl_net,
        'total_pnl': total_pnl,
        'total_pnl_pct': (total_pnl / total_capital_invested * 100) if total_capital_invested > 0 else 0,
    }


class PortfolioTracker:
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
        transaction['date'] = transaction['date'].isoformat()
        portfolios[name]["transactions"].append(transaction)
        st.session_state["_pt_portfolios"] = portfolios
        PortfolioTracker.save_all_portfolios_to_json()

    @staticmethod
    def save_all_portfolios_to_json(filename: str = "saved_portfolio.json") -> None:
        try:
            path = Path(filename)
            all_data = st.session_state.get("_pt_portfolios", {})
            path.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Errore salvataggio JSON %s: %s", filename, e)

    @staticmethod
    def calculate_portfolio_pnl(transactions: List[Dict[str, Any]], commission: float = 1.0, tax_rate: float = 26.0) -> \
    Dict[str, Any]:
        if not transactions:
            return {'open_positions': [], 'aggregated_positions': [], 'capital_gains': [], 'net_invested': 0,
                    'current_value': 0, 'realized_pnl': 0, 'total_pnl': 0, 'total_pnl_pct': 0}

        capital_gains_realized = _calculate_realized_pnl(transactions, commission, tax_rate)
        realized_pnl_net = sum(cg['net_pnl'] for cg in capital_gains_realized)
        open_positions_base = _calculate_open_positions(transactions)
        isins_to_fetch = [isin for isin, data in open_positions_base.items() if data['quantity'] > 1e-9]
        market_data = _fetch_current_prices_and_series(isins_to_fetch)
        open_positions_details = _enrich_open_positions_with_pnl(open_positions_base, market_data)
        summary = _aggregate_portfolio_summary(open_positions_details, realized_pnl_net, transactions)
        aggregated_positions = PortfolioTracker.calculate_aggregated_positions(transactions, open_positions_details)

        return {**summary, 'open_positions': open_positions_details, 'aggregated_positions': aggregated_positions,
                'capital_gains': capital_gains_realized}

    @staticmethod
    def calculate_aggregated_positions(transactions, open_positions_details):
        aggregated = []
        all_isins = set(t['isin'] for t in transactions)
        for isin in all_isins:
            trans_isin = [t for t in transactions if t['isin'] == isin]
            total_bought_qty = sum(t['quantity'] for t in trans_isin if t['type'] == 'buy')
            total_bought_value = sum(t['quantity'] * t['price'] for t in trans_isin if t['type'] == 'buy')
            total_sold_qty = sum(t['quantity'] for t in trans_isin if t['type'] == 'sell')
            total_sold_value = sum(t['quantity'] * t['price'] for t in trans_isin if t['type'] == 'sell')
            avg_buy_price = total_bought_value / total_bought_qty if total_bought_qty > 0 else 0
            realized_pnl_gross = total_sold_value - (total_sold_qty * avg_buy_price)
            current_pos = next((p for p in open_positions_details if p['isin'] == isin), None)
            unrealized_pnl = current_pos['unrealized_pnl'] if current_pos else 0
            aggregated.append({
                'isin': isin, 'total_bought_qty': total_bought_qty, 'total_sold_qty': total_sold_qty,
                'current_qty': current_pos['quantity'] if current_pos else 0, 'avg_buy_price': avg_buy_price,
                'realized_pnl': realized_pnl_gross, 'unrealized_pnl': unrealized_pnl,
                'total_pnl': realized_pnl_gross + unrealized_pnl,
                'current_value': current_pos['current_amount'] if current_pos else 0,
            })
        return aggregated