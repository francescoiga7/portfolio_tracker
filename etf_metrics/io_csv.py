# -*- coding: utf-8 -*-
from typing import Dict, List
import os
import numpy as np
import pandas as pd


def upsert_results_csv(csv_path: str, rows: List[Dict]) -> None:
    new_df = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
        except Exception:
            old_df = pd.DataFrame(columns=new_df.columns)
        for c in new_df.columns:
            if c not in old_df.columns:
                old_df[c] = np.nan
        for c in old_df.columns:
            if c not in new_df.columns:
                new_df[c] = np.nan
        new_keys = set(zip(new_df["isin"], new_df["period"]))
        mask = ~old_df.apply(lambda r: (r.get("isin"), r.get("period")) in new_keys, axis=1)
        merged = pd.concat([old_df[mask], new_df], ignore_index=True)
        merged.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)