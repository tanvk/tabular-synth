from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _tvd(p: pd.Series, q: pd.Series) -> float:
    """Total Variation Distance between two discrete distributions."""
    p = p.fillna(0.0)
    q = q.fillna(0.0)
    # align categories
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=0.0)
    q = q.reindex(idx, fill_value=0.0)
    return 0.5 * np.abs(p - q).sum()

def univariate_similarity(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """
    For numeric cols: KS test statistic (lower is better).
    For categorical cols: TVD over normalized value counts (lower is better).
    """
    common_cols = [c for c in real.columns if c in synth.columns]
    rows = []
    for c in common_cols:
        r, s = real[c], synth[c]
        if _is_numeric(r):
            r_clean = r.dropna()                                                                            # drop NaNs; if too few values, skip
            s_clean = s.dropna()
            if len(r_clean) > 5 and len(s_clean) > 5:
                stat = ks_2samp(r_clean, s_clean).statistic
            else:
                stat = np.nan
            metric = "KS"
            value = stat
        else:
            r_norm = (r.astype("object").value_counts(normalize=True))                                      # categorical: compare normalized histograms
            s_norm = (s.astype("object").value_counts(normalize=True))
            stat = _tvd(r_norm, s_norm)
            metric = "TVD"
            value = stat
        rows.append({"column": c, "metric": metric, "value": float(value)})
    return pd.DataFrame(rows).sort_values("value")

def correlation_delta(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Pearson correlations for numeric columns present in both.
    Returns pairwise abs delta; lower is better.
    """
    num_cols = [c for c in real.columns if _is_numeric(real[c]) and c in synth.columns]
    if len(num_cols) < 2:
        return pd.DataFrame(columns=["col_i", "col_j", "abs_delta"])
    r_corr = real[num_cols].corr(method="pearson")
    s_corr = synth[num_cols].corr(method="pearson")
    rows = []
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i+1:]:
            delta = abs((r_corr.loc[c1, c2] or 0.0) - (s_corr.loc[c1, c2] or 0.0))
            rows.append({"col_i": c1, "col_j": c2, "abs_delta": float(delta)})
    df = pd.DataFrame(rows).sort_values("abs_delta")
    return df

def basic_fidelity_report(real: pd.DataFrame, synth: pd.DataFrame, top_pairs: int = 10) -> dict:
    uni = univariate_similarity(real, synth)
    corr = correlation_delta(real, synth)
    headline = {
        "univariate_ok_%": float((uni["value"] <= 0.1).mean() * 100.0),                                       # rough heuristic
        "median_KS_TVD": float(uni["value"].median(skipna=True)),
        "median_corr_delta": float(corr["abs_delta"].median(skipna=True)) if not corr.empty else np.nan,
    }
    return {
        "headline": headline,
        "univariate": uni.to_dict(orient="records"),
        "corr_top": corr.head(top_pairs).to_dict(orient="records"),
        "corr_worst": corr.tail(top_pairs).to_dict(orient="records") if not corr.empty else [],
    }