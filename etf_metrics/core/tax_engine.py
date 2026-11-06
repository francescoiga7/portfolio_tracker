# -*- coding: utf-8 -*-
"""Registro fiscale italiano per plusvalenze/minusvalenze (art. 68 TUIR).

Un'unica implementazione condivisa da TUTTI i motori (backtest daily,
backtest PAC mensile/daily, trading live, tracker): lo stesso trade paga
le stesse tasse ovunque, coerentemente col principio del progetto
"quello che si valida in backtest è quello che il portafoglio reale esegue".

Regole modellate (fiscalità italiana, redditi diversi di natura finanziaria):
  - aliquota sul plusvalore realizzato (default 26%, parametrica);
  - le minusvalenze compensano le plusvalenze dello stesso periodo e dei
    periodi successivi, entro il 4° periodo d'imposta successivo
    (art. 68, comma 9, TUIR): una perdita realizzata nel 2022 copre
    fino al 31/12/2026;
  - il credito si consuma FIFO (prima le perdite più vecchie, che scadono
    prima);
  - `carryforward_years`:
      4     -> regola italiana (default)
      0     -> compensazione solo nello stesso anno solare
      None  -> recupero illimitato (comportamento storico del backtest
               daily, mantenuto per confronto)

Semplificazioni dichiarate:
  - aliquota unica per strumento: la quota al 12,5% (titoli di Stato
    white-list) degli ETF obbligazionari non è modellata;
  - l'imposta è detratta alla vendita (in dichiarazione si salda l'anno
    successivo, ma il capitale reinvestibile è lo stesso);
  - minusvalenze da fonte estera compensabili con plusvalenze italiane
    (dal 2019): nessuna distinzione di fonte;
  - la base imponibile è il gain come definito dal motore chiamante
    (al netto delle commissioni di vendita dove disponibile).
"""
from typing import Any, Dict, List, Optional, Tuple


def _year_of(date: Any) -> Optional[int]:
    """Anno solare di una data (None se la data non è interpretabile).

    Accetta oggetti con attributo `.year` (datetime, pd.Timestamp, date) o
    stringhe tipo 'YYYY-MM-DD' / 'YYYY-MM-DD HH:MM' (i primi 4 caratteri
    devono essere un anno plausibile, 1900-2199).
    """
    if date is None:
        return None
    for attr in ('year',):
        y = getattr(date, attr, None)
        if isinstance(y, int):
            return y
    try:
        s = str(date)[:4]
        y = int(s)
        return y if 1900 < y < 2200 else None
    except (ValueError, TypeError):
        return None


class ItalianTaxLedger:
    """Cassetto fiscale delle minusvalenze, condiviso da tutti i motori.

    Stato interno:
      - `buckets`: cestini annuali delle perdite ancora recuperabili, nella
        forma `[{'year': 2023, 'amount': 120.0}, ...]`;
      - contatori aggregati (`total_tax_paid`, `total_losses_registered`,
        `total_credit_used`, `total_losses_expired`) per la reportistica.

    Metodi:
      - `register_sale(gain, date)`: logica core — registra un realizzo e
        ritorna (imposta dovuta, credito usato);
      - `available_credit(date)` / `credit_by_expiry(date)`: interrogazione
        per la reportistica (credito residuo, con scadenze);
      - `to_state()` / `from_state()`: serializzazione per lo stato live JSON.

    Usage tipico (motore di backtest o live):
        ledger = ItalianTaxLedger(tax_rate=26.0, carryforward_years=4)
        ...
        tax, credit_used = ledger.register_sale(gain, trade_date)
        cash += proceeds - tax
    """

    def __init__(self, tax_rate: float = 26.0,
                 carryforward_years: Optional[int] = 4):
        self.tax_rate = float(tax_rate)
        self.carryforward_years = carryforward_years
        self.buckets: List[Dict[str, Any]] = []
        self.total_tax_paid = 0.0
        self.total_losses_registered = 0.0
        self.total_credit_used = 0.0
        self.total_losses_expired = 0.0

    def _is_expired(self, bucket: Dict[str, Any], year: Optional[int]) -> bool:
        if self.carryforward_years is None or year is None:
            return False
        return year > bucket['year'] + self.carryforward_years

    def _sweep_expired(self, year: Optional[int]) -> None:
        """Rimuove i cestini scaduti (lazily: si attivano al primo uso)."""
        if self.carryforward_years is None or year is None:
            return
        alive = []
        for b in self.buckets:
            if self._is_expired(b, year):
                self.total_losses_expired += b['amount']
            else:
                alive.append(b)
        self.buckets = alive

    def register_sale(self, gain: float, date: Any = None) -> Tuple[float, float]:
        """Registra il realizzo di una vendita e ritorna (imposta, credito usato).

        gain: plusvalore/minusvalore della vendita (segno compreso), come
        definito dal motore chiamante. Le minusvalenze alimentano il credito;
        le plusvalenze consumano prima il credito non scaduto (FIFO).
        """
        year = _year_of(date)
        self._sweep_expired(year)

        if gain > 0:
            credit_used = self._consume_credit(gain, year)
            taxable = gain - credit_used
            tax = taxable * self.tax_rate / 100.0
            self.total_tax_paid += tax
            self.total_credit_used += credit_used
            return tax, credit_used

        if gain < 0:
            amount = -gain
            if year is not None:
                self.buckets.append({'year': year, 'amount': amount})
                self.total_losses_registered += amount
            return 0.0, 0.0

        return 0.0, 0.0

    def _consume_credit(self, gain: float, year: Optional[int]) -> float:
        """Consuma il credito FIFO (primi i cestini più vecchi) fino a coprire
        `gain`; scarta i cestini svuotati. Ritorna il credito effettivamente
        usato."""
        remaining = gain
        used = 0.0
        for b in self.buckets:
            if remaining <= 1e-12:
                break
            take = min(b['amount'], remaining)
            if take > 0:
                b['amount'] -= take
                remaining -= take
                used += take
        self.buckets = [b for b in self.buckets if b['amount'] > 1e-9]
        return used

    def available_credit(self, date: Any = None) -> float:
        """Minusvalenze residue recuperabili alla data indicata."""
        year = _year_of(date)
        self._sweep_expired(year)
        return float(sum(b['amount'] for b in self.buckets))

    def credit_by_expiry(self, date: Any = None) -> List[Dict[str, Any]]:
        """Credito residuo raggruppato per anno di scadenza, crescente.

        Ritorna [{'expires': 2027, 'amount': 250.0}, ...]; con recupero
        illimitato la chiave 'expires' vale None.
        """
        year = _year_of(date)
        self._sweep_expired(year)
        agg: Dict[Optional[int], float] = {}
        for b in self.buckets:
            exp = None
            if self.carryforward_years is not None:
                exp = b['year'] + self.carryforward_years
            agg[exp] = agg.get(exp, 0.0) + b['amount']
        return [{'expires': exp, 'amount': amt}
                for exp, amt in sorted(agg.items(), key=lambda kv: (kv[0] is None, kv[0]))]

    def to_state(self) -> Dict[str, Any]:
        """Serializza il ledger (cestini + contatori) per la persistenza JSON
        dello stato live."""
        return {
            'tax_rate': self.tax_rate,
            'carryforward_years': self.carryforward_years,
            'buckets': [dict(b) for b in self.buckets],
            'total_tax_paid': self.total_tax_paid,
            'total_losses_registered': self.total_losses_registered,
            'total_credit_used': self.total_credit_used,
            'total_losses_expired': self.total_losses_expired,
        }

    @classmethod
    def from_state(cls, state: Optional[Dict[str, Any]],
                   tax_rate: float = 26.0,
                   carryforward_years: Optional[int] = 4) -> "ItalianTaxLedger":
        """Ricostruisce il ledger dallo stato persistito (o nuovo se assente
        o malformato: mai crashare per via del cassetto fiscale)."""
        ledger = cls(tax_rate=tax_rate, carryforward_years=carryforward_years)
        if not isinstance(state, dict):
            return ledger
        try:
            if state.get('carryforward_years') is not None:
                ledger.carryforward_years = int(state['carryforward_years'])
            for b in (state.get('buckets') or []):
                amt = float(b.get('amount', 0.0))
                try:
                    yr = int(b.get('year'))
                except (TypeError, ValueError):
                    yr = None
                if amt > 0 and yr is not None:
                    ledger.buckets.append({'year': yr, 'amount': amt})
            ledger.total_tax_paid = float(state.get('total_tax_paid', 0.0) or 0.0)
            ledger.total_losses_registered = float(
                state.get('total_losses_registered', 0.0) or 0.0)
            ledger.total_credit_used = float(state.get('total_credit_used', 0.0) or 0.0)
            ledger.total_losses_expired = float(
                state.get('total_losses_expired', 0.0) or 0.0)
        except (TypeError, ValueError, KeyError, AttributeError, IndexError):
            return cls(tax_rate=tax_rate, carryforward_years=carryforward_years)
        return ledger
