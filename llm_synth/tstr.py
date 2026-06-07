"""
Train a gradient-boosted classifier (sklearn HistGradientBoostingClassifier).
Target: 'readmitted' (NO / >30 / <30).

Three evaluation modes run sequentially:
  1. seed       — train on seed (80/20 split + 5-fold CV),  test on held-out seed
  2. synth-self — train on synthetic (80/20 split),         test on held-out synthetic
  3. synth-seed — train on full synthetic,                  test on full seed

Usage:
    python scripts/train_xgboost.py                 # runs all three in order
    python scripts/train_xgboost.py --mode seed
    python scripts/train_xgboost.py --mode synth-self
    python scripts/train_xgboost.py --mode synth-seed
"""

import argparse
import copy
import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _PlattLR
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

# ---------------------------------------------------------------------------
# Paths — resolved from config at import time; overridable for testing
# ---------------------------------------------------------------------------
from llm_synth.config import load_config, get_root as _get_root, get_domain_name as _get_domain_name

def _tstr_paths() -> tuple[Path, Path, Path]:
    try:
        cfg    = load_config()
        root   = _get_root()
        seed   = root / cfg.get("validation", {}).get("seed_csv",     "data/diab_seed.csv")
        synth  = root / cfg.get("validation", {}).get("synthetic_csv", "processed_data/synthetic_output.csv")
        domain = _get_domain_name()
        out    = root / "processed_data" / domain if domain else root / "processed_data"
        return seed, synth, out
    except FileNotFoundError:
        root = Path(__file__).resolve().parent.parent
        return root / "data/diab_seed.csv", root / "processed_data/synthetic_output.csv", root / "processed_data"

SEED_FILE, SYNTH_FILE, RESULTS_DIR = _tstr_paths()

# ---------------------------------------------------------------------------
# Schema — loaded from config.yaml; no hardcoded column names
# ---------------------------------------------------------------------------

def _tstr_schema() -> tuple[str, list[str] | None, dict[str, list], set]:
    """Read target column, class order, ordinal encodings, and expected cols from config."""
    try:
        cfg  = load_config().get("tstr", {})
        target      = cfg.get("target_column", "readmitted")
        target_order = cfg.get("target_order")          # may be None → inferred from data
        ordinals     = {col: v["order"] for col, v in cfg.get("ordinal_columns", {}).items()}
        schema_cols  = set(load_config().get("schema", {}).get("expected_columns", []))
        return target, target_order, ordinals, schema_cols
    except FileNotFoundError:
        return "readmitted", ["NO", ">30", "<30"], {}, set()

TARGET, TARGET_ORDER, _ORDINAL_COLS, EXPECTED_COLS = _tstr_schema()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ordinal columns: map to integer rank using the declared order
    for col, order in _ORDINAL_COLS.items():
        if col in df.columns:
            col_map = {v: i for i, v in enumerate(order)}
            df[col] = df[col].fillna("").map(col_map).fillna(-1).astype(int)

    # All remaining object columns: label-encode (domain-agnostic)
    for col in df.select_dtypes(include="object").columns:
        if col == TARGET:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))

    return df


def load_and_prepare(path: Path) -> pd.DataFrame:
    _peek = pd.read_csv(path, nrows=0)
    _first = str(_peek.columns[0]) if len(_peek.columns) else ""
    _idx = 0 if (_first in ("", "Unnamed: 0")) else None
    df = pd.read_csv(path, index_col=_idx)
    if not EXPECTED_COLS.issubset(df.columns):
        df = pd.read_csv(path)
    df = encode_features(df)
    target_map = {v: i for i, v in enumerate(TARGET_ORDER)}
    df[TARGET] = df[TARGET].map(target_map)
    return df.dropna(subset=[TARGET])


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    return (len(y) / (len(counts) * counts))[y]


def _print_section(title: str) -> None:
    bar = "=" * 60
    logger.info("\n%s", bar)
    logger.info("  %s", title)
    logger.info("%s", bar)


def _show_target_dist(y: np.ndarray, label: str) -> None:
    logger.info("\nTarget distribution — %s:", label)
    for i, name in enumerate(TARGET_ORDER):
        n = (y == i).sum()
        logger.info("  %s: %s  (%.1f%%)", name, f"{n:,}", n/len(y)*100)


def _report(y_true, y_pred) -> dict:
    f1 = f1_score(y_true, y_pred, average="macro")
    target_names_str = [str(v) for v in TARGET_ORDER]
    logger.info("\nF1-macro: %.4f", f1)
    logger.info("%s", classification_report(y_true, y_pred, target_names=target_names_str, digits=4))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=target_names_str, columns=target_names_str)
    logger.info("Confusion matrix:")
    logger.info("%s", cm.to_string())
    return {"f1_macro": round(float(f1), 4)}


class _CalibratedModel(BaseEstimator, ClassifierMixin):
    """
    Multiclass probability calibrator wrapping a pre-fitted base classifier.
    Fits one isotonic or Platt (sigmoid) regressor per class using OvR
    decomposition on the base model's raw probabilities.
    Compatible with pickle and sklearn >= 1.6 (no cv='prefit' dependency).
    """

    def __init__(self, base_model, method: str = "isotonic") -> None:
        self.base_model  = base_model
        self.method      = method
        self._calibrators: list = []

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "_CalibratedModel":
        proba   = self.base_model.predict_proba(X_cal)
        n_cls   = proba.shape[1]
        y_bin   = label_binarize(y_cal, classes=list(range(n_cls)))
        if n_cls == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])
        self._calibrators = []
        for i in range(n_cls):
            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(proba[:, i], y_bin[:, i])
            else:                                          # sigmoid / Platt
                cal = _PlattLR(C=1.0, solver="lbfgs", max_iter=1000)
                cal.fit(proba[:, i].reshape(-1, 1), y_bin[:, i])
            self._calibrators.append(cal)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba     = self.base_model.predict_proba(X)
        cal_proba = np.empty_like(proba)
        for i, cal in enumerate(self._calibrators):
            if self.method == "isotonic":
                cal_proba[:, i] = cal.predict(proba[:, i])
            else:
                cal_proba[:, i] = cal.predict_proba(proba[:, i].reshape(-1, 1))[:, 1]
        # Normalise rows to sum to 1
        row_sums = cal_proba.sum(axis=1, keepdims=True)
        return cal_proba / np.maximum(row_sums, 1e-10)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def calibrate_model(
    base_model: HistGradientBoostingClassifier,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> _CalibratedModel:
    """Calibrate a pre-fitted model on a held-out calibration set."""
    return _CalibratedModel(base_model, method=method).fit(X_cal, y_cal)


def _calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> dict:
    """
    Compute Expected Calibration Error (ECE) and macro-averaged Brier score
    across all classes using one-vs-rest decomposition.
    """
    n_classes = y_proba.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    # label_binarize returns (n, 1) for binary; expand to (n, 2) for uniform handling
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    n = len(y_true)

    brier = float(np.mean([
        brier_score_loss(y_bin[:, i], y_proba[:, i])
        for i in range(n_classes)
    ]))

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_classes):
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_proba[:, i] >= lo) & (y_proba[:, i] < hi)
            if not mask.any():
                continue
            ece += (mask.sum() / n) * abs(y_proba[mask, i].mean() - y_bin[mask, i].mean())
    ece /= n_classes

    return {"brier_score": round(brier, 4), "ece": round(ece, 4)}


def _pr_curve_data(y_true: np.ndarray, y_proba: np.ndarray, n_points: int = 60) -> dict:
    """
    Micro-averaged precision-recall curve (one-vs-rest over all classes, raveled)
    plus the micro-averaged average-precision (PR-AUC) score. Curve is
    downsampled to `n_points` for compact JSON storage / dashboard plotting.
    """
    n_classes = y_proba.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
    ap = average_precision_score(y_bin, y_proba, average="micro")

    # Downsample by recall for compact, evenly-spaced storage
    order = np.argsort(recall)
    recall_s, precision_s = recall[order], precision[order]
    if len(recall_s) > n_points:
        idx = np.linspace(0, len(recall_s) - 1, n_points).astype(int)
        recall_s, precision_s = recall_s[idx], precision_s[idx]

    return {
        "recall":    [round(float(r), 4) for r in recall_s],
        "precision": [round(float(p), 4) for p in precision_s],
        "ap_micro":  round(float(ap), 4),
    }


def _print_calibration_table(base: dict, cal: dict, n_cal: int, f1_base: float, f1_cal: float, method: str) -> None:
    logger.info("\n  Calibration results (%s, cal set = %s rows):", method, f"{n_cal:,}")
    logger.info("  %-18s %10s %12s", "Metric", "Base", "Calibrated")
    logger.info("  %s", '-'*42)
    logger.info("  %-18s %10.4f %12.4f", "F1-macro", f1_base, f1_cal)
    logger.info("  %-18s %10.4f %12.4f", "Brier score", base['brier_score'], cal['brier_score'])
    logger.info("  %-18s %10.4f %12.4f", "ECE", base['ece'], cal['ece'])


def _permutation_importance(model, X, y, feature_cols) -> dict:
    logger.info("\nComputing permutation importances …")
    perm = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring="f1_macro")
    ranked = sorted(zip(feature_cols, perm.importances_mean), key=lambda x: x[1], reverse=True)
    logger.info("Feature importances (F1-macro drop):")
    for feat, score in ranked:
        logger.info("  %-25s %.4f", feat, score)
    return {k: round(float(v), 4) for k, v in ranked}


# ---------------------------------------------------------------------------
# Mode 1: seed → seed  (80/20 split + CV)
# ---------------------------------------------------------------------------

def run_seed_seed(seed_df: pd.DataFrame, feature_cols: list, cv_folds: int,
                  cal_method: str = "isotonic") -> dict:
    _print_section("Mode 1 — Train on SEED / Test on SEED")

    X = seed_df[feature_cols].values
    y = seed_df[TARGET].values.astype(int)

    # 80% train | 20% test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    # 80% base-train | 20% calibration  (from the 80% train split)
    X_base, X_cal, y_base, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=42
    )
    sw_base = sample_weights(y_base)
    sw_tr   = sample_weights(y_tr)

    _show_target_dist(y_base, f"seed base-train ({len(y_base):,} rows)")
    _show_target_dist(y_cal,  f"seed calibration ({len(y_cal):,} rows)")
    _show_target_dist(y_te,   f"seed test  ({len(y_te):,} rows)")

    # --- CV on full train split (evaluation only — hyperparams are fixed) ---
    logger.info("\nRunning %d-fold stratified CV on train split …", cv_folds)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    for tr_idx, val_idx in cv.split(X_tr, y_tr):
        m = make_model()
        m.fit(X_tr[tr_idx], y_tr[tr_idx], sample_weight=sw_tr[tr_idx])
        cv_scores.append(f1_score(y_tr[val_idx], m.predict(X_tr[val_idx]), average="macro"))
    cv_scores = np.array(cv_scores)
    logger.info("  CV F1-macro: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # --- Base model (trained on base-train only, calibration set held out) ---
    logger.info("\nFitting base model on base-train split …")
    model = make_model()
    model.fit(X_base, y_base, sample_weight=sw_base)
    y_pred_base  = model.predict(X_te)
    y_proba_base = model.predict_proba(X_te)

    logger.info("\nClassification report — base model (seed → seed hold-out):")
    metrics = _report(y_te, y_pred_base)
    base_cal_metrics = _calibration_metrics(y_te, y_proba_base)

    # --- Calibration ---
    logger.info("\nCalibrating (%s) on %s held-out samples …", cal_method, f"{len(y_cal):,}")
    cal_model    = calibrate_model(model, X_cal, y_cal, method=cal_method)
    y_pred_cal   = cal_model.predict(X_te)
    y_proba_cal  = cal_model.predict_proba(X_te)
    cal_metrics  = _calibration_metrics(y_te, y_proba_cal)
    f1_cal       = f1_score(y_te, y_pred_cal, average="macro")

    _print_calibration_table(base_cal_metrics, cal_metrics, len(y_cal),
                             metrics["f1_macro"], f1_cal, cal_method)

    pr_curve = _pr_curve_data(y_te, y_proba_cal)
    logger.info("  PR-AUC (micro, calibrated): %.4f", pr_curve["ap_micro"])

    importances = _permutation_importance(cal_model, X_te, y_te, feature_cols)

    # --- Save both models ---
    for tag_suffix, obj in (("", model), ("_calibrated", cal_model)):
        path = RESULTS_DIR / f"xgboost_seed_seed{tag_suffix}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Model saved → %s", path)

    results = {
        "mode": "seed_seed",
        "train_data": "seed (64% base-train + 16% calibration)",
        "test_data":  "seed (20% hold-out)",
        "base_train_rows": int(len(y_base)),
        "cal_rows":        int(len(y_cal)),
        "test_rows":       int(len(y_te)),
        "cv_folds":            cv_folds,
        "cv_f1_macro_mean":    round(float(cv_scores.mean()), 4),
        "cv_f1_macro_std":     round(float(cv_scores.std()),  4),
        "calibration_method":  cal_method,
        **metrics,
        "base_brier":    base_cal_metrics["brier_score"],
        "base_ece":      base_cal_metrics["ece"],
        "cal_f1_macro":  round(float(f1_cal), 4),
        "cal_brier":     cal_metrics["brier_score"],
        "cal_ece":       cal_metrics["ece"],
        "feature_importances": importances,
        "pr_curve": pr_curve,
    }
    _save_results(results, "seed_seed")
    return results


# ---------------------------------------------------------------------------
# Mode 2: synthetic → synthetic  (80/20 split)
# ---------------------------------------------------------------------------

def run_synth_synth(synth_df: pd.DataFrame, feature_cols: list,
                    cal_method: str = "isotonic") -> dict:
    _print_section("Mode 2 — Train on SYNTHETIC / Test on SYNTHETIC")

    X = synth_df[feature_cols].values
    y = synth_df[TARGET].values.astype(int)

    # 80% train | 20% test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    # 80% base-train | 20% calibration
    X_base, X_cal, y_base, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=42
    )
    sw_base = sample_weights(y_base)

    _show_target_dist(y_base, f"synthetic base-train ({len(y_base):,} rows)")
    _show_target_dist(y_cal,  f"synthetic calibration ({len(y_cal):,} rows)")
    _show_target_dist(y_te,   f"synthetic test  ({len(y_te):,} rows)")

    logger.info("\nFitting base model on base-train split …")
    model = make_model()
    model.fit(X_base, y_base, sample_weight=sw_base)
    y_pred_base  = model.predict(X_te)
    y_proba_base = model.predict_proba(X_te)

    logger.info("\nClassification report — base model (synthetic → synthetic hold-out):")
    metrics = _report(y_te, y_pred_base)
    base_cal_metrics = _calibration_metrics(y_te, y_proba_base)

    # --- Calibration ---
    logger.info("\nCalibrating (%s) on %s held-out samples …", cal_method, f"{len(y_cal):,}")
    cal_model   = calibrate_model(model, X_cal, y_cal, method=cal_method)
    y_pred_cal  = cal_model.predict(X_te)
    y_proba_cal = cal_model.predict_proba(X_te)
    cal_metrics = _calibration_metrics(y_te, y_proba_cal)
    f1_cal      = f1_score(y_te, y_pred_cal, average="macro")

    _print_calibration_table(base_cal_metrics, cal_metrics, len(y_cal),
                             metrics["f1_macro"], f1_cal, cal_method)

    pr_curve = _pr_curve_data(y_te, y_proba_cal)
    logger.info("  PR-AUC (micro, calibrated): %.4f", pr_curve["ap_micro"])

    importances = _permutation_importance(cal_model, X_te, y_te, feature_cols)

    for tag_suffix, obj in (("", model), ("_calibrated", cal_model)):
        path = RESULTS_DIR / f"xgboost_synth_synth{tag_suffix}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Model saved → %s", path)

    results = {
        "mode": "synth_synth",
        "train_data": "synthetic (64% base-train + 16% calibration)",
        "test_data":  "synthetic (20% hold-out)",
        "base_train_rows": int(len(y_base)),
        "cal_rows":        int(len(y_cal)),
        "test_rows":       int(len(y_te)),
        "calibration_method":  cal_method,
        **metrics,
        "base_brier":    base_cal_metrics["brier_score"],
        "base_ece":      base_cal_metrics["ece"],
        "cal_f1_macro":  round(float(f1_cal), 4),
        "cal_brier":     cal_metrics["brier_score"],
        "cal_ece":       cal_metrics["ece"],
        "feature_importances": importances,
        "pr_curve": pr_curve,
    }
    _save_results(results, "synth_synth")
    return results


# ---------------------------------------------------------------------------
# Mode 3: synthetic → seed
# ---------------------------------------------------------------------------

def run_synth_seed(synth_df: pd.DataFrame, seed_holdout_df: pd.DataFrame, feature_cols: list,
                   cal_method: str = "isotonic") -> dict:
    _print_section("Mode 3 — Train on SYNTHETIC / Test on SEED hold-out")

    X_synth = synth_df[feature_cols].values
    y_synth = synth_df[TARGET].values.astype(int)

    # Split synthetic into 80% base-train | 20% calibration
    X_base, X_cal, y_base, y_cal = train_test_split(
        X_synth, y_synth, test_size=0.20, stratify=y_synth, random_state=42
    )
    sw_base = sample_weights(y_base)

    # Evaluate on the same 20% seed hold-out used in Mode 1
    X_te = seed_holdout_df[feature_cols].values
    y_te = seed_holdout_df[TARGET].values.astype(int)

    _show_target_dist(y_base, f"synthetic base-train ({len(y_base):,} rows)")
    _show_target_dist(y_cal,  f"synthetic calibration ({len(y_cal):,} rows)")
    _show_target_dist(y_te,   f"seed hold-out test  ({len(y_te):,} rows)")

    logger.info("\nFitting base model on synthetic base-train …")
    model = make_model()
    model.fit(X_base, y_base, sample_weight=sw_base)
    y_pred_base  = model.predict(X_te)
    y_proba_base = model.predict_proba(X_te)

    logger.info("\nClassification report — base model (synthetic → seed):")
    metrics = _report(y_te, y_pred_base)
    base_cal_metrics = _calibration_metrics(y_te, y_proba_base)

    # --- Calibration ---
    logger.info("\nCalibrating (%s) on %s held-out samples …", cal_method, f"{len(y_cal):,}")
    cal_model   = calibrate_model(model, X_cal, y_cal, method=cal_method)
    y_pred_cal  = cal_model.predict(X_te)
    y_proba_cal = cal_model.predict_proba(X_te)
    cal_metrics = _calibration_metrics(y_te, y_proba_cal)
    f1_cal      = f1_score(y_te, y_pred_cal, average="macro")

    _print_calibration_table(base_cal_metrics, cal_metrics, len(y_cal),
                             metrics["f1_macro"], f1_cal, cal_method)

    pr_curve = _pr_curve_data(y_te, y_proba_cal)
    logger.info("  PR-AUC (micro, calibrated): %.4f", pr_curve["ap_micro"])

    importances = _permutation_importance(cal_model, X_te, y_te, feature_cols)

    for tag_suffix, obj in (("", model), ("_calibrated", cal_model)):
        path = RESULTS_DIR / f"xgboost_synth_seed{tag_suffix}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Model saved → %s", path)

    results = {
        "mode": "synth_seed",
        "train_data": "synthetic (80% base-train + 20% calibration)",
        "test_data":  "seed (20% hold-out, same split as Mode 1)",
        "base_train_rows": int(len(y_base)),
        "cal_rows":        int(len(y_cal)),
        "test_rows":       int(len(y_te)),
        "calibration_method":  cal_method,
        **metrics,
        "base_brier":    base_cal_metrics["brier_score"],
        "base_ece":      base_cal_metrics["ece"],
        "cal_f1_macro":  round(float(f1_cal), 4),
        "cal_brier":     cal_metrics["brier_score"],
        "cal_ece":       cal_metrics["ece"],
        "feature_importances": importances,
        "pr_curve": pr_curve,
    }
    _save_results(results, "synth_seed")
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_results(results: dict, tag: str) -> None:
    path = RESULTS_DIR / f"xgboost_{tag}_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", path)


def _print_summary(r1: dict, r2: dict, r3: dict) -> None:
    _print_section("Summary — F1-macro across all three modes")
    rows = [
        ("seed → seed (hold-out)",       r1.get("f1_macro", "—")),
        ("synthetic → synthetic (hold-out)", r2.get("f1_macro", "—")),
        ("synthetic → seed",             r3.get("f1_macro", "—")),
    ]
    for label, score in rows:
        bar = "█" * int(float(score) * 40) if isinstance(score, float) else ""
        logger.info("  %-40s  %.4f  %s", label, score, bar)

    fidelity = r3.get("f1_macro", 0) / r1.get("f1_macro", 1)
    logger.info("\n  Synthetic fidelity (mode 3 / mode 1): %.2f%%", fidelity * 100)
    summary = {"mode_1_seed_seed": r1.get("f1_macro"),
               "mode_2_synth_synth": r2.get("f1_macro"),
               "mode_3_synth_seed": r3.get("f1_macro"),
               "fidelity_ratio": round(fidelity, 4)}
    path = RESULTS_DIR / "xgboost_comparison.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\n  Comparison saved → %s", path)
    # Also write the condition-specific file the dashboard reads
    domain = _get_domain_name()
    if domain:
        cond_path = RESULTS_DIR / f"{domain}_tstr_comparison.json"
        with open(cond_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("  Dashboard sync  → %s", cond_path)


# ---------------------------------------------------------------------------
# Extended analysis — multi-model AUROC, shift robustness, subgroup parity
# ---------------------------------------------------------------------------

def _make_extended_models() -> dict:
    """Three classifiers for extended multi-model AUROC analysis."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, random_state=42),
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("clf", _LR(max_iter=1000)),
        ]),
        "MLP (3-layer)": Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500)),
        ]),
    }


def compute_all_aurocs(
    models: dict,
    real_clfs: dict,
    synth_clfs: dict,
    X_te_real: np.ndarray,
    y_te_real: np.ndarray,
    X_te_synth: np.ndarray,
    y_te_synth: np.ndarray,
) -> dict:
    """TRTR / TSTR / TRTS / TSTS AUROCs for all models.

    TRTR: Train Real,  Test Real   — baseline upper bound
    TSTR: Train Synth, Test Real   — synthetic utility
    TRTS: Train Real,  Test Synth  — model generalisation to synthetic
    TSTS: Train Synth, Test Synth  — synthetic self-consistency
    """
    from sklearn.metrics import roc_auc_score
    results = {}
    for name in models:
        results[name] = {
            "TRTR": roc_auc_score(y_te_real,  real_clfs[name].predict_proba(X_te_real)[:, 1]),
            "TSTR": roc_auc_score(y_te_real,  synth_clfs[name].predict_proba(X_te_real)[:, 1]),
            "TRTS": roc_auc_score(y_te_synth, real_clfs[name].predict_proba(X_te_synth)[:, 1]),
            "TSTS": roc_auc_score(y_te_synth, synth_clfs[name].predict_proba(X_te_synth)[:, 1]),
        }
    return results


def distribution_shift_robustness(
    models: dict,
    real_clfs: dict,
    synth_clfs: dict,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed_df: pd.DataFrame,
    feature_cols: list,
    shift_pct: float = 0.10,
) -> list:
    """Measure AUROC degradation under Gaussian noise shift on numeric features."""
    from sklearn.metrics import roc_auc_score
    # from llm_synth.utils.plots import plot_shift_robustness as _plot_shift

    numeric_cols = seed_df[feature_cols].select_dtypes(include="number").columns.tolist()
    numeric_idx  = [feature_cols.index(c) for c in numeric_cols]

    rng = np.random.default_rng(0)
    X_te_shifted = X_te.astype(float).copy()
    for i in numeric_idx:
        X_te_shifted[:, i] += rng.normal(0, shift_pct * X_tr[:, i].std(), size=len(X_te_shifted))

    logger.info("\nDistribution Shift Robustness  (shift_pct=%s)", shift_pct)
    logger.info("Shifted columns (%d): %s", len(numeric_cols), numeric_cols)
    logger.info("%s", '='*90)
    logger.info("%-25s %14s %15s %10s  %15s %16s %11s %14s",
                "Model", "AUC real→orig", "AUC real→shift", "ΔAUC real",
                "AUC synth→orig", "AUC synth→shift", "ΔAUC synth", "Ratio(Δs/Δr)")
    logger.info("%s", '-'*90)

    rows = []
    for name in models:
        auc_r_orig  = roc_auc_score(y_te, real_clfs[name].predict_proba(X_te)[:, 1])
        auc_r_shift = roc_auc_score(y_te, real_clfs[name].predict_proba(X_te_shifted)[:, 1])
        auc_s_orig  = roc_auc_score(y_te, synth_clfs[name].predict_proba(X_te)[:, 1])
        auc_s_shift = roc_auc_score(y_te, synth_clfs[name].predict_proba(X_te_shifted)[:, 1])
        d_real  = auc_r_shift - auc_r_orig
        d_synth = auc_s_shift - auc_s_orig
        shift_ratio = (d_synth / d_real) if abs(d_real) >= 0.005 else np.nan
        rows.append((name, auc_r_orig, auc_r_shift, d_real, auc_s_orig, auc_s_shift, d_synth, shift_ratio))
        ratio_str = f"{shift_ratio:>+14.4f}" if not np.isnan(shift_ratio) else f"{'nan':>14}"
        logger.info("%-25s %14.4f %15.4f %+10.4f  %15.4f %16.4f %+11.4f %s",
                    name, auc_r_orig, auc_r_shift, d_real,
                    auc_s_orig, auc_s_shift, d_synth, ratio_str)

    # _plot_shift(rows, shift_pct)
    return rows


def _ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for binary classification."""
    bins  = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    err   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum():
            err += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return err / total


def subgroup_auroc_parity(
    models: dict,
    real_clfs: dict,
    synth_clfs: dict,
    X_te_real: np.ndarray,
    y_te_real: np.ndarray,
    X_te_synth: np.ndarray,
    y_te_synth: np.ndarray,
    seed_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    idx_te_real: np.ndarray,
    idx_te_synth: np.ndarray,
    parity_cols: list | None = None,
    min_group_size: int = 1000,
) -> dict:
    """Per-subgroup AUROC parity across Baseline / TSTR / TRTS regimes."""
    from sklearn.metrics import roc_auc_score
    # from llm_synth.utils.plots import plot_subgroup_heatmap as _plot_heatmap

    if parity_cols is None:
        parity_cols = ["gender", "race"]

    seed_te  = seed_df.iloc[idx_te_real].reset_index(drop=True)
    synth_te = synth_df.iloc[idx_te_synth].reset_index(drop=True)

    def _sub_auc(clf, X_sub, y_sub):
        if len(y_sub) < 10 or len(np.unique(y_sub)) < 2:
            return np.nan
        return roc_auc_score(y_sub, clf.predict_proba(X_sub)[:, 1])

    logger.info("\nSubgroup AUROC Parity  (min_group_size=%d)\n%s", min_group_size, '='*80)
    parity_dfs = {}

    for col in parity_cols:
        if col not in seed_df.columns:
            continue
        counts   = seed_df[col].value_counts()
        eligible = counts[counts >= min_group_size].index.tolist()
        excluded = counts[counts <  min_group_size].index.tolist()

        logger.info("\n%s", '─'*80)
        logger.info("  Attribute : %s", col)
        logger.info("  Eligible  : %s", eligible)
        logger.info("  Excluded  : %s  (< %d members in seed)", excluded, min_group_size)

        parity_records = []
        for grp in eligible:
            mask_r = (seed_te[col]  == grp).values
            mask_s = (synth_te[col] == grp).values
            for name in models:
                parity_records.append({
                    col:        grp,
                    "model":    name,
                    "n_real":   int(mask_r.sum()),
                    "n_synth":  int(mask_s.sum()),
                    "Baseline": _sub_auc(real_clfs[name],  X_te_real[mask_r],  y_te_real[mask_r]),
                    "TSTR":     _sub_auc(synth_clfs[name], X_te_real[mask_r],  y_te_real[mask_r]),
                    "TRTS":     _sub_auc(real_clfs[name],  X_te_synth[mask_s], y_te_synth[mask_s]),
                })

        df_p = pd.DataFrame(parity_records)
        parity_dfs[col] = df_p

        for name in models:
            sub = df_p[df_p["model"] == name].copy()
            logger.info("\n  ── %s", name)
            logger.info("  %-22s %7s %8s %10s %8s %8s", "Group", "N_real", "N_synth", "Baseline", "TSTR", "TRTS")
            logger.info("  %s", '-'*65)
            for _, row in sub.iterrows():
                logger.info("  %-22s %7d %8d %10.4f %8.4f %8.4f",
                            str(row[col]), row['n_real'], row['n_synth'],
                            row['Baseline'], row['TSTR'], row['TRTS'])
            for regime in ("Baseline", "TSTR", "TRTS"):
                vals = sub[regime].dropna()
                if len(vals) >= 2:
                    gap   = vals.max() - vals.min()
                    worst = sub.loc[vals.idxmin(), col]
                    best  = sub.loc[vals.idxmax(), col]
                    logger.info("  Parity gap [%s]: %.4f  best=%s (%.4f)  worst=%s (%.4f)",
                                regime, gap, best, vals.max(), worst, vals.min())

        # _plot_heatmap(df_p, col)

    return parity_dfs


def run_extended(seed_file: Path | None = None, synth_file: Path | None = None) -> None:
    """Multi-model AUROC + shift robustness + subgroup parity extended evaluation.

    Uses a binary target (any readmission vs none) and OrdinalEncoder so that
    Logistic Regression and MLP can run alongside HistGradientBoosting.
    """
    from sklearn.preprocessing import OrdinalEncoder
    # from llm_synth.utils.plots import (
    #     plot_calibration_panel,
    #     plot_roc_divergence_panel,
    #     plot_trts_panel,
    #     plot_tstr_panel,
    #     plot_tstr_trtr_auroc,
    # )

    _seed_file  = seed_file  or SEED_FILE
    _synth_file = synth_file or SYNTH_FILE

    _print_section("Extended Evaluation — Multi-model AUROC, Shift Robustness, Subgroup Parity")

    # ── Load data ──────────────────────────────────────────────────────────────
    seed_df_raw  = pd.read_csv(_seed_file)
    synth_df_raw = pd.read_csv(_synth_file)
    for _df in (seed_df_raw, synth_df_raw):
        _df.drop(columns=[c for c in _df.columns if c.startswith("Unnamed:")],
                 inplace=True, errors="ignore")

    feature_cols = [c for c in seed_df_raw.columns if c != TARGET]

    # Fit OrdinalEncoder on seed; transform both (needed for LR / MLP)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_seed  = enc.fit_transform(seed_df_raw[feature_cols])
    X_synth = enc.transform(synth_df_raw[feature_cols])

    # Binary target: any readmission vs no readmission
    no_idx  = TARGET_ORDER.index("NO") if TARGET_ORDER and "NO" in TARGET_ORDER else 0
    int_map = {v: i for i, v in enumerate(TARGET_ORDER)}
    y_seed  = (seed_df_raw[TARGET].map(int_map)  != no_idx).astype(int).values
    y_synth = (synth_df_raw[TARGET].map(int_map) != no_idx).astype(int).values

    # ── Train / test splits ───────────────────────────────────────────────────
    idx_seed  = np.arange(len(X_seed))
    idx_synth = np.arange(len(X_synth))
    _, idx_te_real  = train_test_split(idx_seed,  test_size=0.20, stratify=y_seed,  random_state=42)
    _, idx_te_synth = train_test_split(idx_synth, test_size=0.20, stratify=y_synth, random_state=42)
    X_tr_real,  X_te_real,  y_tr_real,  y_te_real  = train_test_split(
        X_seed,  y_seed,  test_size=0.20, stratify=y_seed,  random_state=42)
    X_tr_synth, X_te_synth, y_tr_synth, y_te_synth = train_test_split(
        X_synth, y_synth, test_size=0.20, stratify=y_synth, random_state=42)

    # ── Fit all models ────────────────────────────────────────────────────────
    models = _make_extended_models()
    real_probas, synth_probas = {}, {}
    real_clfs,  synth_clfs   = {}, {}
    for name, clf in models.items():
        rc = copy.deepcopy(clf)
        rc.fit(X_tr_real, y_tr_real)
        real_clfs[name]   = rc
        real_probas[name] = rc.predict_proba(X_te_real)[:, 1]
        sc = copy.deepcopy(clf)
        sc.fit(X_tr_synth, y_tr_synth)
        synth_clfs[name]   = sc
        synth_probas[name] = sc.predict_proba(X_te_real)[:, 1]

    # ── ROC panels ───────────────────────────────────────────────────────────
    DIVERGENCE_THRESHOLD = 0.05
    # max_divergence = plot_roc_divergence_panel(
    #     models, real_probas, synth_probas, y_te_real, DIVERGENCE_THRESHOLD)
    max_divergence = {name: np.nan for name in models}
    # plot_tstr_trtr_auroc(models, real_clfs, synth_clfs, X_te_real, y_te_real)
    # plot_tstr_panel(models, real_clfs, synth_clfs, X_te_real, y_te_real)

    # ── Four-regime AUROC table ───────────────────────────────────────────────
    auroc_results = compute_all_aurocs(
        models, real_clfs, synth_clfs, X_te_real, y_te_real, X_te_synth, y_te_synth)

    trts_rows = []
    logger.info("\nAUROC — All Regimes\n%s", '='*90)
    logger.info("%-25s %8s  %8s  %8s  %8s  %13s  %13s  %s",
                "Model", "TRTR", "TSTR", "TRTS", "TSTS", "Δ(TRTR-TRTS)", "Δ(TRTR-TSTR)", "Verdict")
    logger.info("%s", '-'*90)
    for name in models:
        r        = auroc_results[name]
        gap_trts = r["TRTR"] - r["TRTS"]
        gap_tstr = r["TRTR"] - r["TSTR"]
        verdict  = "faithful" if abs(gap_trts) < 0.05 else ("too easy" if gap_trts < 0 else "diverges")
        trts_rows.append((name, r["TRTR"], r["TSTR"], r["TRTS"], gap_trts, gap_tstr, verdict))
        logger.info("%-25s %8.4f  %8.4f  %8.4f  %8.4f  %+13.4f  %+13.4f  %s",
                    name, r['TRTR'], r['TSTR'], r['TRTS'], r['TSTS'],
                    gap_trts, gap_tstr, verdict)

    # plot_trts_panel(models, real_clfs, trts_rows, X_te_real, y_te_real, X_te_synth, y_te_synth)

    # ── Shift robustness ──────────────────────────────────────────────────────
    shift_rows = distribution_shift_robustness(
        models, real_clfs, synth_clfs,
        X_tr_real, X_te_real, y_te_real,
        seed_df_raw, feature_cols,
    )

    # ── Subgroup parity ───────────────────────────────────────────────────────
    parity_dfs = subgroup_auroc_parity(
        models, real_clfs, synth_clfs,
        X_te_real, y_te_real, X_te_synth, y_te_synth,
        seed_df_raw, synth_df_raw, idx_te_real, idx_te_synth,
    )

    # ── Calibration panel ─────────────────────────────────────────────────────
    regimes_cal = {
        "Baseline": (real_clfs,  X_te_real,  y_te_real),
        "TSTR":     (synth_clfs, X_te_real,  y_te_real),
        "TRTS":     (real_clfs,  X_te_synth, y_te_synth),
        "TSTS":     (synth_clfs, X_te_synth, y_te_synth),
    }
    # cal_results = plot_calibration_panel(models, regimes_cal, 10, _ece_binary)
    cal_results = {}

    # ── Final CSV report ──────────────────────────────────────────────────────
    _trts_map   = {r[0]: r for r in trts_rows}
    _shift_map  = {r[0]: r for r in shift_rows}
    _REGIME_KEY = {"Baseline": "TRTR", "TSTR": "TSTR", "TRTS": "TRTS", "TSTS": "TSTS"}

    report_rows = []
    for name in models:
        tr = _trts_map[name]
        sr = _shift_map[name]
        for regime in ("Baseline", "TSTR", "TRTS", "TSTS"):
            auc = auroc_results[name][_REGIME_KEY[regime]]
            row = {
                "model":         name,
                "regime":        regime,
                "AUC":           round(auc, 4),
                "Brier":         round(cal_results[(name, regime)]["brier"], 4)
                                 if (name, regime) in cal_results else np.nan,
                "ECE":           round(cal_results[(name, regime)]["ece"],   4)
                                 if (name, regime) in cal_results else np.nan,
                "AUC_shift10":   round({"Baseline": sr[2], "TSTR": sr[5]}.get(regime, np.nan), 4),
                "delta_shift10": round({"Baseline": sr[3], "TSTR": sr[6]}.get(regime, np.nan), 4),
                "shift_ratio":   round(sr[7], 4)
                                 if regime in ("Baseline", "TSTR") and not np.isnan(sr[7]) else np.nan,
                "ROC_max_div":   round(max_divergence[name], 4),
                "TRTS_verdict":  tr[6] if regime == "TRTS" else np.nan,
            }
            for col in parity_dfs:
                df_p = parity_dfs[col]
                sub  = df_p[df_p["model"] == name]
                for grp in sub[col].tolist():
                    if regime in df_p.columns:
                        val = sub.loc[sub[col] == grp, regime].values
                        row[f"{col}_{grp}_AUC"] = (
                            round(float(val[0]), 4) if len(val) and not np.isnan(val[0]) else np.nan
                        )
                    else:
                        row[f"{col}_{grp}_AUC"] = np.nan
                vals = sub[regime].dropna() if regime in df_p.columns else pd.Series([], dtype=float)
                row[f"{col}_parity_gap"] = (
                    round(float(vals.max() - vals.min()), 4) if len(vals) >= 2 else np.nan
                )
            report_rows.append(row)

    report_df = pd.DataFrame(report_rows).set_index(["model", "regime"])
    logger.info("\n%s", '='*80)
    logger.info("FINAL REPORT — all metrics across all model × regime combinations")
    logger.info("%s\n", '='*80)
    with pd.option_context("display.max_columns", None, "display.width", 160,
                           "display.float_format", "{:.4f}".format):
        logger.info("%s", report_df.to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "eval_report.csv"
    report_df.to_csv(report_path)
    logger.info("\nSaved → %s", report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train HistGradientBoosting on seed/synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["seed", "synth-self", "synth-seed", "all"],
        default="all",
        help=(
            "seed: seed→seed | synth-self: synth→synth | "
            "synth-seed: synth→seed | all: run all three (default)"
        ),
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--cal-method",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Calibration method: isotonic (default) or sigmoid (Platt scaling)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help=(
            "Run extended multi-model analysis after the standard modes: "
            "TRTR/TSTR/TRTS/TSTS AUROC across HistGradientBoosting, Logistic "
            "Regression, and MLP; distribution shift robustness; subgroup AUROC "
            "parity by race/gender; calibration panel; and a final CSV report."
        ),
    )
    args, _ = parser.parse_known_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("\nLoading seed data      : %s", SEED_FILE)
    seed_df = load_and_prepare(SEED_FILE)
    feature_cols = [c for c in seed_df.columns if c != TARGET]
    logger.info("  %s rows, %d features", f"{len(seed_df):,}", len(feature_cols))

    # Reserve 20% of seed as a shared hold-out for Mode 1 and Mode 3
    seed_train_df, seed_holdout_df = train_test_split(
        seed_df, test_size=0.20, stratify=seed_df[TARGET], random_state=42
    )
    logger.info("  Hold-out split: %s train / %s test",
                f"{len(seed_train_df):,}", f"{len(seed_holdout_df):,}")

    synth_df = None
    if args.mode in ("synth-self", "synth-seed", "all"):
        logger.info("\nLoading synthetic data : %s", SYNTH_FILE)
        synth_df = load_and_prepare(SYNTH_FILE)
        logger.info("  %s rows", f"{len(synth_df):,}")

    if args.mode == "seed":
        run_seed_seed(seed_df, feature_cols, args.cv_folds, args.cal_method)

    elif args.mode == "synth-self":
        run_synth_synth(synth_df, feature_cols, args.cal_method)

    elif args.mode == "synth-seed":
        run_synth_seed(synth_df, seed_holdout_df, feature_cols, args.cal_method)

    else:  # all
        r1 = run_seed_seed(seed_df,    feature_cols, args.cv_folds, args.cal_method)
        r2 = run_synth_synth(synth_df, feature_cols, args.cal_method)
        r3 = run_synth_seed(synth_df,  seed_holdout_df, feature_cols, args.cal_method)
        _print_summary(r1, r2, r3)

    if args.extended:
        run_extended()


if __name__ == "__main__":
    main()
