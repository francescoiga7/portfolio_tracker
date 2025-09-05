# -*- coding: utf-8 -*-
from typing import List, Optional
import pandas as pd
from .config import PREFERRED_SUFFIXES

def _suffix(symbol: str) -> str:
    return symbol[symbol.find("."):] if "." in symbol else ""


def pick_preferred_symbol(candidates: List[str], preferred_suffixes=None) -> Optional[str]:
    preferred_suffixes = preferred_suffixes or PREFERRED_SUFFIXES
    cands = [c for c in candidates if c]
    if not cands:
        return None

    def score(s: str) -> int:
        suf = _suffix(s)
        return preferred_suffixes.index(suf) if suf in preferred_suffixes else 999

    return sorted(cands, key=score)[0]


def to_percent_index(s: pd.Series) -> pd.Series:
    return (s / s.iloc[0] - 1.0) * 100.0