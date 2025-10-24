# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, DefaultDict
import streamlit as st
from collections import defaultdict
import pandas as pd

from etf_metrics.clients.yahoo_client import resolve_isin_one, get_series
from .metrics import get_trend_signal, get_satellite_signal

logger = logging.getLogger(__name__)

def _calculate_open_positions(transactions: List[Dict[str, Any]]) -> DefaultDict[str, Dict[str, Any]]:
    """
    Calcola le quantità, i costi totali e lo stato 'satellite' delle posizioni aperte.
    """
    positions = defaultdict(lambda: {'quantity': 0, 'total_cost': 0, 'satellite': False})
    sorted_trans = sorted(transactions, key=lambda x: x['date'])

    for t in sorted_trans:
        isin = t['isin']
        qty = float(t['quantity'])
        price = float(t['price'])

        if t['type'] == 'buy':
            positions[isin]['quantity'] += qty
            positions[isin]['total_cost'] += qty * price
            if t.get('satellite', False):
                positions[isin]['satellite'] = True
        elif t['type'] == 'sell':
            if positions[isin]['quantity'] > 0:
                if positions[isin]['quantity'] < 1e-9: continue
                avg_buy_price = positions[isin]['total_cost'] / positions[isin]['quantity']

                sell_qty = min(qty, positions[isin]['quantity'])

                cost_of_sold_shares = sell_qty * avg_buy_price
                positions[isin]['quantity'] -= sell_qty
                if positions[isin]['quantity'] < 1e-9:
                    positions[isin]['quantity'] = 0
                    positions[isin]['total_cost'] = 0
                else:
                    positions[isin]['total_cost'] -= cost_of_sold_shares
                    if positions[isin]['total_cost'] < 0: positions[isin]['total_cost'] = 0

    return positions

def _calculate_realized_pnl(transactions: List[Dict[str, Any]], commission: float, tax_rate: float) -> \
        List[Dict[str, Any]]:
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
                logger.warning(
                    f"Vendita di {isin} per una quantità ({qty}) superiore a quella posseduta. Transazione saltata.")
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
    Recupera il prezzo più recente, la serie storica 'Close', il DataFrame OHLCV
    e la serie VIX per una lista di ISIN.
    """
    data = {}
    fetch_period = "2y" # Potrebbe servire uno storico adeguato per l'ATR e il max recente

    # Recupera la serie VIX una sola volta
    try:
        vix_series = get_series("^VIX", fetch_period)
        if vix_series is None or vix_series.empty:
            logger.warning("Impossibile recuperare i dati del VIX.")
            vix_series = pd.Series(dtype=float) # Serie vuota per evitare errori dopo
    except Exception as e:
        logger.error(f"Errore nel recupero dati VIX: {e}")
        vix_series = pd.Series(dtype=float)

    for isin in isins:
        ticker = resolve_isin_one(isin)
        series_df = get_series(ticker, fetch_period, as_dataframe=True) if ticker else None

        if series_df is not None and not series_df.empty and 'Close' in series_df.columns:
            close_series = series_df['Close'].copy()
            close_series.name = ticker
            data[isin] = {
                "price": close_series.iloc[-1],
                "series": close_series,
                "series_df": series_df,
                "vix_series": vix_series # Aggiungi la serie VIX ai dati per ogni ISIN
            }
        else:
            data[isin] = {
                "price": 0,
                "series": None,
                "series_df": None,
                "vix_series": vix_series # Aggiungi comunque la serie VIX
            }
            logger.warning(f"Impossibile recuperare i dati OHLCV completi per {isin} ({ticker})")
    return data


def _enrich_open_positions_with_pnl(open_positions: DefaultDict[str, Dict[str, Any]],
                                    market_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Arricchisce i dati delle posizioni aperte con P&L e segnale di trend/satellite.
    Modificato per passare la serie VIX a get_satellite_signal.
    """
    enriched_positions = []
    for isin, data in open_positions.items():
        if data['quantity'] > 1e-9: # Usa una tolleranza piccola per evitare errori floating point
            market_info = market_data.get(isin, {})
            current_price = market_info.get("price", 0)
            series = market_info.get("series", None)
            series_df = market_info.get("series_df", None)
            vix_series = market_info.get("vix_series", pd.Series(dtype=float)) # Recupera VIX

            invested_amount = data['total_cost']
            current_amount = data['quantity'] * current_price
            unrealized_pnl = current_amount - invested_amount
            avg_buy_price = invested_amount / data['quantity'] if data['quantity'] > 1e-9 else 0
            unrealized_pnl_pct = (unrealized_pnl / invested_amount * 100) if invested_amount > 1e-9 else 0.0

            is_satellite_asset = data.get('satellite', False)

            trend_signal_info = {"signal": "N/D", "reason": "Dati insufficienti"} # Default

            if is_satellite_asset:
                if series_df is not None and vix_series is not None:
                     # Passa vix_series alla funzione
                    trend_signal_info = get_satellite_signal(series_df, vix_series)
                else:
                    reason = []
                    if series_df is None: reason.append("Mancano dati OHLCV.")
                    if vix_series is None or vix_series.empty : reason.append("Mancano dati VIX.")
                    trend_signal_info = {"signal": "Dati Insufficienti", "reason": " ".join(reason)}
            else: # Asset Core
                if series is not None:
                    original_signal_string = get_trend_signal(series)
                    trend_signal_info = {"signal": original_signal_string, "reason": ""}
                # else: mantiene il default "Dati insufficienti"

            enriched_positions.append({
                'isin': isin,
                'quantity': data['quantity'],
                'avg_buy_price': avg_buy_price,
                'current_price': current_price,
                'invested_amount': invested_amount,
                'current_amount': current_amount,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'trend_signal': trend_signal_info.get("signal", "N/D"),
                'trend_reason': trend_signal_info.get("reason", ""),
                'satellite': is_satellite_asset
            })
    return enriched_positions

def _aggregate_portfolio_summary(
        open_positions_details: List[Dict[str, Any]],
        realized_pnl_net: float,
        transactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calcola le metriche di riepilogo del portafoglio.
    """
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
        """Orchestra il calcolo completo del P&L del portafoglio."""
        if not transactions:
            return {
                'open_positions': [], 'aggregated_positions': [], 'capital_gains': [],
                'net_invested': 0, 'current_value': 0, 'realized_pnl': 0,
                'total_pnl': 0, 'total_pnl_pct': 0
            }

        capital_gains_realized = _calculate_realized_pnl(transactions, commission, tax_rate)
        realized_pnl_net = sum(cg['net_pnl'] for cg in capital_gains_realized)

        open_positions_base = _calculate_open_positions(transactions)

        isins_to_fetch = [isin for isin, data in open_positions_base.items() if data['quantity'] > 1e-9]
        market_data = _fetch_current_prices_and_series(isins_to_fetch)

        open_positions_details = _enrich_open_positions_with_pnl(open_positions_base, market_data)

        summary = _aggregate_portfolio_summary(open_positions_details, realized_pnl_net, transactions)

        aggregated_positions = PortfolioTracker.calculate_aggregated_positions(transactions, open_positions_details)

        return {
            **summary,
            'open_positions': open_positions_details,
            'aggregated_positions': aggregated_positions,
            'capital_gains': capital_gains_realized
        }

    @staticmethod
    def calculate_aggregated_positions(transactions: List[Dict[str, Any]],
                                       open_positions_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calcola una vista aggregata per ISIN (acquisti totali, vendite totali, P&L, etc.)."""
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
                'isin': isin,
                'total_bought_qty': total_bought_qty,
                'total_sold_qty': total_sold_qty,
                'current_qty': current_pos['quantity'] if current_pos else 0,
                'avg_buy_price': avg_buy_price,
                'realized_pnl': realized_pnl_gross,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': realized_pnl_gross + unrealized_pnl,
                'current_value': current_pos['current_amount'] if current_pos else 0,
            })
        return aggregated