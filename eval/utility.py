from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
import numpy as np

def _split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def _pipeline(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
    ])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num, num_cols),
        ("cat", cat, cat_cols),
    ])

def _is_binary(y: pd.Series) -> bool:
    return y.nunique(dropna=True) == 2

def _positive_label(y: pd.Series):
    classes = list(y.dropna().unique())
    if len(classes) != 2:
        return None
    for cand in [">50K", "yes", "true", 1, "1", "Y", "Positive", "pos", True]:
        if cand in classes:
            return cand
    vc = y.value_counts(dropna=True)
    return vc.index[-1]  # minority as positive

def _best_threshold(y_true, y_proba, pos_label) -> float:
    """Grid-search a threshold that maximizes F1 on a validation set."""
    y_true_bin = (y_true == pos_label).astype(int)
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true_bin, preds, average="binary")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)

def utility_transfer_report(real: pd.DataFrame, synth: pd.DataFrame, target_col: str) -> dict:
    """Train on synthetic, validate threshold on a slice of real, test on a held-out real set."""
    assert target_col in real.columns and target_col in synth.columns, "target_col missing"

    Xr, yr = _split_xy(real, target_col)
    Xs, ys = _split_xy(synth, target_col)

    # Split real into train/val/test for fair thresholding & evaluation
    strat = yr if _is_binary(yr) else None
    Xr_train, Xr_temp, yr_train, yr_temp = train_test_split(
        Xr, yr, test_size=0.35, random_state=42, stratify=strat
    )
    Xr_val, Xr_test, yr_val, yr_test = train_test_split(
        Xr_temp, yr_temp, test_size=0.5, random_state=42,
        stratify=(yr_temp if _is_binary(yr_temp) else None)
    )

    pre = _pipeline(pd.concat([Xr_train, Xs], axis=0, ignore_index=True))

    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=3000,
    )

    out = {}

    if _is_binary(yr):
        pos = _positive_label(yr)

        # ===== Train on synthetic =====
        pipe_synth = Pipeline([("pre", pre), ("clf", clf)])
        pipe_synth.fit(Xs, ys)

        # Pick threshold on real validation set
        proba_val = pipe_synth.predict_proba(Xr_val)[:, 1]
        thr = _best_threshold(yr_val, proba_val, pos)

        # Evaluate on real test
        proba_test_s = pipe_synth.predict_proba(Xr_test)[:, 1]
        preds_test_s = (proba_test_s >= thr).astype(int)
        out["threshold_used"] = thr
        out["synth_to_real_AUROC"] = float(roc_auc_score(yr_test, proba_test_s))
        out["synth_to_real_PRAUC"] = float(average_precision_score(yr_test == pos, proba_test_s))
        out["synth_to_real_F1@tuned"] = float(f1_score(yr_test == pos, preds_test_s))

        # ===== Train =====
        pipe_real = Pipeline([("pre", pre), ("clf", LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=3000
        ))])
        pipe_real.fit(Xr_train, yr_train)

        proba_val_r = pipe_real.predict_proba(Xr_val)[:, 1]
        thr_r = _best_threshold(yr_val, proba_val_r, pos)

        proba_test_r = pipe_real.predict_proba(Xr_test)[:, 1]
        preds_test_r = (proba_test_r >= thr_r).astype(int)

        out["real_to_real_AUROC"] = float(roc_auc_score(yr_test, proba_test_r))
        out["real_to_real_PRAUC"] = float(average_precision_score(yr_test == pos, proba_test_r))
        out["real_to_real_F1@tuned"] = float(f1_score(yr_test == pos, preds_test_r))

        # Deltas
        out["delta_AUROC"] = float(out["real_to_real_AUROC"] - out["synth_to_real_AUROC"])
        out["delta_PRAUC"] = float(out["real_to_real_PRAUC"] - out["synth_to_real_PRAUC"])
        out["delta_F1@tuned"] = float(out["real_to_real_F1@tuned"] - out["synth_to_real_F1@tuned"])

    else:
        pipe_synth = Pipeline([("pre", pre), ("clf", clf)])
        pipe_synth.fit(Xs, ys)
        preds_s = pipe_synth.predict(Xr_test)

        pipe_real = Pipeline([("pre", pre), ("clf", clf)])
        pipe_real.fit(Xr_train, yr_train)
        preds_r = pipe_real.predict(Xr_test)

        out["synth_to_real_F1_macro"] = float(f1_score(yr_test, preds_s, average="macro"))
        out["real_to_real_F1_macro"] = float(f1_score(yr_test, preds_r, average="macro"))
        out["delta_F1_macro"] = float(out["real_to_real_F1_macro"] - out["synth_to_real_F1_macro"])

    return out