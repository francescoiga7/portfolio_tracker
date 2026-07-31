# -*- coding: utf-8 -*-
from typing import List, Optional

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
