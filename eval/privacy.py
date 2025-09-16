from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def exact_match_rate(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Fraction of synthetic rows that exactly match a real row (string compare)."""
    common = [c for c in real.columns if c in synth.columns]                                                                        # normalize column order/types
    r = real[common].copy()
    s = synth[common].copy()

    r_str = r.astype(str).agg("||".join, axis=1)
    s_str = s.astype(str).agg("||".join, axis=1)
    rate = s_str.isin(set(r_str)).mean()
    return float(rate)

def uniqueness_rate(df: pd.DataFrame) -> float:
    """Share of rows that are unique within df (the higher, the riskier if too close to 1.0 for small data)."""
    sig = df.astype(str).agg("||".join, axis=1)
    vc = sig.value_counts()
    unique_share = (vc == 1).sum() / len(sig) if len(sig) else 0.0
    return float(unique_share)

def knn_min_distance(real: pd.DataFrame, synth: pd.DataFrame, k: int = 1) -> dict:
    """Compute kNN distance of each synthetic row to the real set (numeric-only for simplicity)."""
    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c]) and c in synth.columns]
    if not num_cols:
        return {"median": np.nan, "p05": np.nan, "p95": np.nan}

    R = real[num_cols].fillna(real[num_cols].median())
    S = synth[num_cols].fillna(real[num_cols].median())

    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(R.values)
    dists, _ = nbrs.kneighbors(S.values, n_neighbors=k)
    d = dists[:, -1]  # distance to kth neighbor
    return {"median": float(np.median(d)), "p05": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95))}

def basic_privacy_report(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    emr = exact_match_rate(real, synth)
    uniq = uniqueness_rate(synth)
    knn = knn_min_distance(real, synth, k=1)
    flags = {
        "exact_match_ok": emr == 0.0,
        "knn_min_ok": (not np.isnan(knn["p05"])) and knn["p05"] > 0.0,
    }
    return {
        "exact_match_rate": emr,
        "synthetic_uniqueness_rate": uniq,
        "knn_min_distance": knn,
        "flags": flags,
    }