# -*- coding: utf-8 -*-
from typing import List, Optional
import pandas as pd

from .config import PREFERRED_SUFFIXES


def pick_preferred_symbol(candidates: List[str], preferred_suffixes=None) -> Optional[str]:
    """
    Sceglie il ticker preferito da una lista di candidati in base a un elenco di suffissi.
    """
    preferred_suffixes = preferred_suffixes or PREFERRED_SUFFIXES
    cands = [c for c in candidates if c]
    if not cands:
        return None

    def score(s: str) -> int:
        suf = s[s.find(".") :] if "." in s else ""
        return preferred_suffixes.index(suf) if suf in preferred_suffixes else 999

    return sorted(cands, key=score)[0]


def to_percent_index(s: pd.Series) -> pd.Series:
    """
    Converte una serie di prezzi in indice % dal primo valore.
    Restituisce (s / s.iloc[0] - 1) * 100, evitando errori su serie vuote/zero.
    """
    if s is None or len(s) == 0:
        return s
    base = s.iloc[0]
    if base == 0:
        # Evita divisione per zero: restituisce la serie originale
        return s
    return (s / base - 1.0) * 100.0