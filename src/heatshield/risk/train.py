"""Risk models: heat-index rule, XGBoost without twin, twin alone, XGBoost with twin.

Patient-level split (no patient appears in two splits), Platt (sigmoid) calibration on the
validation patients, all metrics on the untouched test patients. `multi_split()` repeats this over
several random patient splits so one lucky split cannot mislead.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

from ..config import cfg, MODELS
from .features import EHR_COLS, WEAR_COLS, WX_COLS, TWIN_COLS

FEATURE_SETS = {"xgb_no_twin": EHR_COLS + WEAR_COLS + WX_COLS,
                "xgb_with_twin": EHR_COLS + WEAR_COLS + WX_COLS + TWIN_COLS}
LABELS = {"heat_index_rule": "Heat-index rule", "xgb_no_twin": "XGBoost, no twin",
          "twin_only": "Twin forecast alone", "xgb_with_twin": "HEATSHIELD (XGBoost + twin)"}


def split_patients(pids: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    u = rng.permutation(np.unique(pids))
    n = len(u)
    return {"train": set(u[:int(.7 * n)]), "val": set(u[int(.7 * n):int(.85 * n)]), "test": set(u[int(.85 * n):])}


def _metrics(y, p, thr=0.5) -> dict:
    pred = p >= thr
    tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
    return {"n": len(y), "events": int(y.sum()),
            "auroc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
            "auprc": average_precision_score(y, p) if y.sum() else np.nan,
            "brier": brier_score_loss(y, np.clip(p, 0, 1)),
            "sensitivity": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1)}


def _scale_rule(hi: np.ndarray) -> np.ndarray:
    """Heat index -> 0..1 score. 41 C ('danger') maps to 0.5."""
    return 1 / (1 + np.exp(-(hi - 41) / 1.5))


def train_and_evaluate(df: pd.DataFrame, seed: int | None = None, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = cfg()["seed"] if seed is None else seed
    sp = split_patients(df["pid"].to_numpy(), seed)
    part = {k: df["pid"].isin(v).to_numpy() for k, v in sp.items()}
    preds = pd.DataFrame({"pid": df["pid"], "evening": df["evening"], "label": df["label"],
                          "beta_blocker": df["beta_blocker"], "home_ac": df["home_ac"],
                          "split": np.select([part["train"], part["val"]], ["train", "val"], "test")})
    preds["heat_index_rule"] = _scale_rule(df["fc_heat_index_max"].to_numpy())
    preds["twin_only"] = df["twin_p_event"].to_numpy()
    for name, cols in FEATURE_SETS.items():
        m = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, min_child_weight=5, subsample=0.9,
                          colsample_bytree=0.9, random_state=seed, eval_metric="logloss")
        m.fit(df.loc[part["train"], cols], df.loc[part["train"], "label"])
        raw = np.clip(m.predict_proba(df[cols])[:, 1], 1e-4, 1 - 1e-4)
        logit = np.log(raw / (1 - raw))[:, None]
        platt = LogisticRegression().fit(logit[part["val"]], df.loc[part["val"], "label"])
        preds[name] = platt.predict_proba(logit)[:, 1]
        if save:
            m.save_model(MODELS / f"{name}.json")
            imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
            imp.to_csv(MODELS / f"{name}_importance.csv", header=["importance"])
    rows = []
    test = preds[preds["split"] == "test"]
    for name in LABELS:
        for group, mask in (("all", np.ones(len(test), bool)), ("beta-blocked", test["beta_blocker"] == 1),
                            ("not beta-blocked", test["beta_blocker"] == 0), ("no AC", test["home_ac"] == 0)):
            r = _metrics(test.loc[mask, "label"].to_numpy(), test.loc[mask, name].to_numpy())
            rows.append({"model": LABELS[name], "group": group, **r})
    if save:
        (MODELS / "split.json").write_text(json.dumps({k: sorted(v) for k, v in sp.items()}, indent=1))
    return pd.DataFrame(rows), preds


def multi_split(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Mean and sd of test metrics over several random patient splits."""
    rows = []
    for k in range(n_splits):
        res, _ = train_and_evaluate(df, seed=cfg()["seed"] + k, save=False)
        rows.append(res.assign(split=k))
    allr = pd.concat(rows)
    g = allr.groupby(["model", "group"], sort=False)[["auroc", "auprc", "brier", "sensitivity", "precision"]]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
    return out
