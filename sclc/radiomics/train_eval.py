"""Phase 3: feature selection, nested-CV model training, test evaluation.

For each dataset (LPCT-Dx and BigLunge):
  1. Load stable features from Phase 2 (``stable_features_<dataset>.csv``).
  2. Z-score normalize on train fold.
  3. LASSO logistic-regression feature selection (capped at √N_train).
  4. SMOTE class rebalancing on train fold.
  5. Inner 5-fold CV across SVM-RBF, RandomForest, GradientBoosting; pick
     winner by mean macro-F1.
  6. Refit each algorithm on full train, predict on the held-out test set.

Six model_types are produced (3 algos × {LPCT-trained, BL-trained}):
  - ``radiomics_<algo>`` : trained on Lung-PET-CT-Dx train+val. Evaluated on
    LPCT-Dx test (``dapt_test`` phase) and BigLunge test (``test`` phase,
    cross-dataset transfer).
  - ``radiomics_<algo>_bl`` : trained on BigLunge train+val. Evaluated on
    BigLunge test (``test`` phase, in-sample).

Outputs match the deep-pipeline format (consumed by ``scripts/build_thesis_results.py``):
  - ``results/output/2d/<model_type>/<model_type>_<ts>[_dapt]_inference_probabilities.json``
  - ``results/output/2d/<model_type>/metrics.jsonl``  (phase=dapt_test, phase=test)
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
RADIOMICS_DIR = REPO_ROOT / "results" / "radiomics"
OUTPUT_ROOT = REPO_ROOT / "results" / "output" / "2d"
CLASS_NAMES = ["Adenocarcinoma", "Small Cell", "Squamous"]
META_COLS = {"patient_id", "class_idx", "class_name", "split"}
SEED = 42


# ---------- data loading -----------------------------------------------------

def load_stable(dataset: str) -> pd.DataFrame:
    p = RADIOMICS_DIR / f"stable_features_{dataset}.csv"
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found. Run sclc.radiomics.stability first."
        )
    df = pd.read_csv(p)
    return df


def split_xy(
    df: pd.DataFrame, splits: Tuple[str, ...]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    sub = df[df["split"].isin(splits)].reset_index(drop=True)
    feat_cols = [c for c in sub.columns if c not in META_COLS]
    X = sub[feat_cols].to_numpy(dtype=float)
    y = sub["class_idx"].to_numpy(dtype=int)
    return X, y, feat_cols


def patients_for_split(df: pd.DataFrame, splits: Tuple[str, ...]) -> List[str]:
    return df[df["split"].isin(splits)]["patient_id"].tolist()


# ---------- LASSO feature selection ------------------------------------------

def lasso_select(
    X: np.ndarray, y: np.ndarray, feat_names: List[str], cap: int,
) -> List[int]:
    """Multiclass L1-LR feature selection. Returns indices into feat_names.

    Strategy: sweep C ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0}; pick the smallest C
    that yields ≤ cap nonzero features, falling back to whichever yields the
    most ≤cap if none does. Standard tabular-ML radiomics selection (Liu 2020).
    """
    cs = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    best_idx: List[int] = []
    for c in cs:
        clf = LogisticRegression(
            penalty="l1", solver="saga", C=c, max_iter=10_000,
            random_state=SEED, n_jobs=1,
        )
        clf.fit(X, y)
        # Nonzero across any class.
        nz = np.any(np.abs(clf.coef_) > 0.0, axis=0)
        idx = list(np.where(nz)[0])
        if len(idx) <= cap and len(idx) > 0:
            best_idx = idx
            break
        # Otherwise track the largest selection that's still ≤cap.
        if len(idx) <= cap and len(idx) > len(best_idx):
            best_idx = idx
    if not best_idx:
        # No C produced ≤cap: take top-cap by max-|coef| at C=0.01.
        clf = LogisticRegression(
            penalty="l1", solver="saga", C=0.01, max_iter=10_000,
            multi_class="multinomial", random_state=SEED, n_jobs=1,
        ).fit(X, y)
        scores = np.abs(clf.coef_).max(axis=0)
        order = np.argsort(scores)[::-1]
        best_idx = list(order[:cap])
    return sorted(int(i) for i in best_idx)


# ---------- model factory ----------------------------------------------------

def make_models() -> Dict[str, Any]:
    """Return {algo_id: (estimator, hyperparam_grid)}.

    Hyperparameter search is intentionally compact — N is small, the goal is
    a defensible baseline, not a hyperparam sweep.
    """
    return {
        "svm": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
            [
                {"C": 0.5, "gamma": "scale"},
                {"C": 1.0, "gamma": "scale"},
                {"C": 2.0, "gamma": "scale"},
                {"C": 1.0, "gamma": 0.01},
                {"C": 1.0, "gamma": 0.1},
            ],
        ),
        "rf": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
            [
                {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
                {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 2},
                {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 1},
            ],
        ),
        "gb": (
            GradientBoostingClassifier(random_state=SEED),
            [
                {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
                {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
                {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
            ],
        ),
    }


# ---------- inner CV (model + hyperparam selection) --------------------------

def inner_cv_select(
    X_train: np.ndarray, y_train: np.ndarray,
) -> Dict[str, Any]:
    """5-fold stratified inner CV. For each algo × hyperparam combo, compute
    mean macro-F1 over folds. Returns:
        {algo_id: {"best_params": {...}, "best_score": float}}
    plus a top-level "winner" key naming the best algo.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    summary: Dict[str, Any] = {}
    best_overall: Tuple[str, float, Dict[str, Any]] = ("", -1.0, {})

    for algo, (estimator_proto, grid) in make_models().items():
        algo_best = (-1.0, {})
        for hp in grid:
            scores: List[float] = []
            for fold, (tr, va) in enumerate(cv.split(X_train, y_train)):
                X_tr, X_va = X_train[tr], X_train[va]
                y_tr, y_va = y_train[tr], y_train[va]
                # Z-score on inner-train, apply to inner-val (no leakage).
                mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0)
                sigma = np.where(sigma == 0, 1.0, sigma)
                X_tr_s = (X_tr - mu) / sigma
                X_va_s = (X_va - mu) / sigma
                # SMOTE the inner-train fold only.
                try:
                    Xb, yb = SMOTE(random_state=SEED, k_neighbors=min(5, max(1, np.bincount(y_tr).min() - 1))).fit_resample(X_tr_s, y_tr)
                except Exception:
                    Xb, yb = X_tr_s, y_tr
                clf = estimator_proto.__class__(**{**estimator_proto.get_params(), **hp})
                clf.fit(Xb, yb)
                pred = clf.predict(X_va_s)
                scores.append(f1_score(y_va, pred, average="macro", labels=[0, 1, 2], zero_division=0))
            mean_score = float(np.mean(scores))
            if mean_score > algo_best[0]:
                algo_best = (mean_score, hp)
        summary[algo] = {"best_params": algo_best[1], "best_score": algo_best[0]}
        if algo_best[0] > best_overall[1]:
            best_overall = (algo, algo_best[0], algo_best[1])

    summary["winner"] = best_overall[0]
    summary["winner_score"] = best_overall[1]
    return summary


# ---------- final fit + test eval --------------------------------------------

def fit_final(
    algo: str,
    hp: Dict[str, Any],
    X_train: np.ndarray, y_train: np.ndarray,
    feat_names: List[str],
) -> Tuple[Any, np.ndarray, np.ndarray, List[str]]:
    """Fit an algorithm on full train (LASSO-selected + SMOTE-balanced).

    Returns (fitted_model, mu, sigma, selected_features) for downstream eval.
    """
    cap = max(5, int(round(np.sqrt(len(y_train)))))
    print(f"  [feature selection] LASSO cap=√{len(y_train)}≈{cap}")

    # Z-score on train.
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    X_train_z = (X_train - mu) / sigma

    sel_idx = lasso_select(X_train_z, y_train, feat_names, cap=cap)
    sel_names = [feat_names[i] for i in sel_idx]
    print(f"  [feature selection] selected {len(sel_idx)} / {len(feat_names)} features")
    X_train_sel = X_train_z[:, sel_idx]

    # SMOTE final train.
    counts = np.bincount(y_train)
    print(f"  [SMOTE] pre={counts.tolist()}")
    try:
        Xb, yb = SMOTE(
            random_state=SEED,
            k_neighbors=min(5, max(1, counts.min() - 1)),
        ).fit_resample(X_train_sel, y_train)
        print(f"  [SMOTE] post={np.bincount(yb).tolist()}")
    except Exception as e:
        print(f"  [SMOTE] failed ({e}); using raw train")
        Xb, yb = X_train_sel, y_train

    estimator_proto, _ = make_models()[algo]
    clf = estimator_proto.__class__(**{**estimator_proto.get_params(), **hp})
    clf.fit(Xb, yb)
    return clf, mu, sigma, sel_names


def predict_proba(
    clf: Any, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    sel_names: List[str], all_feat_names: List[str],
) -> np.ndarray:
    """Standardize + select + predict_proba. Returns (N, 3) probability matrix."""
    X_z = (X - mu) / sigma
    sel_idx = [all_feat_names.index(n) for n in sel_names]
    X_sel = X_z[:, sel_idx]
    proba = clf.predict_proba(X_sel)
    # Map to canonical (Adeno, SmallCell, Squamous) order. clf.classes_ may
    # reorder if training has fewer than 3 classes; pad/expand.
    classes = list(getattr(clf, "classes_", [0, 1, 2]))
    if classes != [0, 1, 2]:
        full = np.zeros((proba.shape[0], 3), dtype=float)
        for col, c in enumerate(classes):
            if 0 <= int(c) < 3:
                full[:, int(c)] = proba[:, col]
        # Re-normalize defensively.
        s = full.sum(axis=1, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        proba = full / s
    return proba


# ---------- I/O: probs JSON + metrics.jsonl ----------------------------------

def _build_probs_payload(
    patient_ids: List[str], y_true: np.ndarray, proba: np.ndarray,
) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    for i, pid in enumerate(patient_ids):
        p = [float(proba[i, c]) for c in range(3)]
        pred = int(np.argmax(p))
        samples.append({
            "sample_index": i,
            "patient_id": pid,
            "volume_id": None,
            "true_label": int(y_true[i]),
            "true_name": CLASS_NAMES[int(y_true[i])],
            "pred_label": pred,
            "pred_name": CLASS_NAMES[pred],
            "confidence": float(p[pred]),
            "probabilities": {
                CLASS_NAMES[0]: p[0],
                CLASS_NAMES[1]: p[1],
                CLASS_NAMES[2]: p[2],
            },
        })

    proba_mean = proba.mean(axis=0)
    pred_counts = np.bincount(np.argmax(proba, axis=1), minlength=3)
    return {
        "num_samples": len(samples),
        "class_names": CLASS_NAMES,
        "mean_probability_per_class": {CLASS_NAMES[i]: float(proba_mean[i]) for i in range(3)},
        "predicted_class_counts": {CLASS_NAMES[i]: int(pred_counts[i]) for i in range(3)},
        "predicted_class_fractions": {CLASS_NAMES[i]: float(pred_counts[i] / max(1, len(samples))) for i in range(3)},
        "samples": samples,
    }


def _stratified_bootstrap_indices(y: np.ndarray, n_boot: int, seed: int = 0) -> List[np.ndarray]:
    """Resample indices stratified by class label so all classes stay present."""
    rng = np.random.default_rng(seed)
    by_class: Dict[int, np.ndarray] = {c: np.where(y == c)[0] for c in np.unique(y)}
    out: List[np.ndarray] = []
    for _ in range(n_boot):
        idx_parts = []
        for cls, idxs in by_class.items():
            if len(idxs) == 0:
                continue
            sample = rng.choice(idxs, size=len(idxs), replace=True)
            idx_parts.append(sample)
        out.append(np.concatenate(idx_parts))
    return out


def _f1_bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 0,
) -> Tuple[List[float], List[List[float]]]:
    """Stratified bootstrap CIs for per-class F1 + macro F1.

    Returns (per_class_f1_ci95, [[lo0, hi0], [lo1, hi1], [lo2, hi2]],
             macro_f1_ci95 [lo, hi]).
    """
    n_classes = 3
    pc_samples: List[List[float]] = [[] for _ in range(n_classes)]
    macro_samples: List[float] = []
    for idx in _stratified_bootstrap_indices(y_true, n_boot=n_boot, seed=seed):
        yt = y_true[idx]; yp = y_pred[idx]
        pc = f1_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0)
        macro = f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0)
        for c in range(n_classes):
            pc_samples[c].append(float(pc[c]))
        macro_samples.append(float(macro))
    pc_ci = [[float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))] for s in pc_samples]
    macro_ci = [float(np.percentile(macro_samples, 2.5)), float(np.percentile(macro_samples, 97.5))]
    return pc_ci, macro_ci


def _build_metrics_row(
    phase: str, y_true: np.ndarray, proba: np.ndarray, timestamp: str,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    pred = np.argmax(proba, axis=1)
    n = int(len(y_true))
    pcf1 = f1_score(y_true, pred, average=None, labels=[0, 1, 2], zero_division=0).tolist()
    pcf1_ci, macro_ci = _f1_bootstrap_ci(y_true, pred, n_boot=n_boot)
    return {
        "phase": phase,
        "epoch": 1,
        "timestamp": timestamp,
        "test_patient": {
            "num_patients": n,
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
            "macro_f1_ci95": macro_ci,
            "per_class_f1": pcf1,
            "per_class_f1_ci95": pcf1_ci,
        },
    }


def _atomic_write_json(payload: Dict[str, Any], out_path: Path) -> None:
    """Match sclc/main.py atomic-replace pattern (so ctrl-C mid-write
    doesn't leave a truncated JSON like the swin_unetr DAPT bug)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out_path)


def _append_metrics_jsonl(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite each run; build_thesis_results just reads the latest test row.
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


# ---------- per-algorithm driver ---------------------------------------------

def train_and_eval_algo(
    algo: str,
    inner_summary: Dict[str, Any],
    *,
    train_df: pd.DataFrame,
    test_dfs: List[Tuple[str, str, pd.DataFrame]],   # (phase, dataset_label, df)
    model_type: str,
    feat_names: List[str],
    output_subdir: Path,
) -> Dict[str, Any]:
    """Train ``algo`` on ``train_df`` train+val rows, eval on each ``test_dfs``.

    Writes inference_probabilities JSON per phase + a single metrics.jsonl.
    Returns a small report dict for the provenance.
    """
    print(f"[train_and_eval] {model_type} (algo={algo})")
    # Fit on TRAIN ONLY so val stays held-out for honest val scoring.
    # Earlier behavior (fit on train+val) leaked val into the final model
    # and produced bogus val_acc=1.0 numbers on in-sample evaluation.
    # 15% less training data is the right tradeoff for a fair val curve.
    X_train, y_train, _ = split_xy(train_df, ("train",))

    hp = inner_summary[algo]["best_params"]
    print(f"  [hyperparams] {hp}")
    clf, mu, sigma, sel_names = fit_final(algo, hp, X_train, y_train, feat_names)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics_rows: List[Dict[str, Any]] = []
    eval_outputs: Dict[str, Any] = {}

    for phase, dataset_label, df, split_filter in test_dfs:
        sub = df[df["split"] == split_filter].reset_index(drop=True)
        if sub.empty:
            print(f"  [eval][{phase}] no {split_filter} rows; skipping")
            continue
        # Align columns to train feature space (in case test_df uses a
        # different stable_features file from a different dataset).
        for c in feat_names:
            if c not in sub.columns:
                sub[c] = 0.0
        X_te = sub[feat_names].to_numpy(dtype=float)
        y_te = sub["class_idx"].to_numpy(dtype=int)
        pids = sub["patient_id"].tolist()
        proba = predict_proba(clf, X_te, mu, sigma, sel_names, feat_names)

        # Filename + directory routing.
        # ``build_thesis_results.py`` globs <out_dir>/<model>_*_inference_probabilities.json
        # at the top level only and excludes _dapt_ for biglunge_test discovery.
        # We keep the headline-relevant files (test phases) at the top level
        # and tuck val phases into ``<out_dir>/val/`` so they don't pollute
        # the headline discovery.
        if phase == "dapt_test":
            target_dir = output_subdir
            suffix = "_dapt"
        elif phase == "test":
            target_dir = output_subdir
            suffix = ""
        elif phase == "dapt_val":
            target_dir = output_subdir / "val"
            suffix = "_dapt_val"
        elif phase == "val":
            target_dir = output_subdir / "val"
            suffix = "_val"
        else:
            raise ValueError(f"Unknown phase: {phase}")
        probs_path = target_dir / f"{model_type}_{timestamp}{suffix}_inference_probabilities.json"
        _atomic_write_json(_build_probs_payload(pids, y_te, proba), probs_path)

        metrics_rows.append(_build_metrics_row(phase, y_te, proba, timestamp))

        pred = np.argmax(proba, axis=1)
        # Macro AUC (one-vs-rest); guard for missing classes in test.
        try:
            macro_auc = float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=[0, 1, 2]))
        except Exception:
            macro_auc = float("nan")

        print(f"  [eval][{phase}/{dataset_label}] n={len(y_te)} "
              f"acc={accuracy_score(y_te, pred):.3f} "
              f"macro_f1={f1_score(y_te, pred, average='macro', labels=[0,1,2], zero_division=0):.3f} "
              f"macro_auc={macro_auc:.3f}")
        eval_outputs[phase] = {
            "probs_json": str(probs_path),
            "n_patients": int(len(y_te)),
            "macro_f1": float(f1_score(y_te, pred, average="macro", labels=[0,1,2], zero_division=0)),
            "macro_auc": macro_auc,
        }

    if metrics_rows:
        _append_metrics_jsonl(metrics_rows, output_subdir / "metrics.jsonl")

    return {
        "model_type": model_type,
        "algo": algo,
        "hyperparams": hp,
        "n_features_selected": len(sel_names),
        "selected_features": sel_names,
        "evals": eval_outputs,
    }


# ---------- top-level ---------------------------------------------------------

def run() -> Dict[str, Any]:
    """Train all 6 model_types and write outputs.

    Returns a dict suitable for serialization into per-model provenance.json.
    """
    print("[radiomics] loading stable features")
    lpcd = load_stable("lpcd")
    bl = load_stable("biglunge")
    print(f"  lpcd: {lpcd.shape}  biglunge: {bl.shape}")

    feat_lpcd = [c for c in lpcd.columns if c not in META_COLS]
    feat_bl = [c for c in bl.columns if c not in META_COLS]
    common = sorted(set(feat_lpcd) & set(feat_bl))
    print(f"  feature counts — lpcd: {len(feat_lpcd)}, bl: {len(feat_bl)}, intersection: {len(common)}")

    # Restrict both datasets to the intersecting feature set so cross-dataset
    # transfer (LPCT-trained → BL-test) operates on the same feature axes.
    keep_cols_lpcd = [c for c in lpcd.columns if c in META_COLS or c in common]
    keep_cols_bl = [c for c in bl.columns if c in META_COLS or c in common]
    lpcd = lpcd[keep_cols_lpcd]
    bl = bl[keep_cols_bl]

    # ---- Inner CV to select winning algo per dataset --------------------
    print("\n[inner-cv] selecting winning algorithm on LPCT-Dx train...")
    X_lpcd_tr, y_lpcd_tr, _ = split_xy(lpcd, ("train", "val"))
    inner_lpcd = inner_cv_select(X_lpcd_tr, y_lpcd_tr)
    print(f"  inner-cv summary (LPCT): {json.dumps({k: v for k, v in inner_lpcd.items() if k in ('svm', 'rf', 'gb')}, indent=2)}")
    print(f"  winner (LPCT) = {inner_lpcd['winner']} (mean macro-F1 = {inner_lpcd['winner_score']:.3f})")

    print("\n[inner-cv] selecting winning algorithm on BigLunge train...")
    X_bl_tr, y_bl_tr, _ = split_xy(bl, ("train", "val"))
    inner_bl = inner_cv_select(X_bl_tr, y_bl_tr)
    print(f"  inner-cv summary (BL): {json.dumps({k: v for k, v in inner_bl.items() if k in ('svm', 'rf', 'gb')}, indent=2)}")
    print(f"  winner (BL) = {inner_bl['winner']} (mean macro-F1 = {inner_bl['winner_score']:.3f})")

    # ---- Final fits + test eval -----------------------------------------
    # test_dfs entries are 4-tuples: (phase, label, df, split_filter).
    # Each fit is evaluated on multiple held-out subsets:
    #   * LPCT-trained: LPCT-val, LPCT-test, BL-val (transfer), BL-test (transfer).
    #   * BL-trained:   BL-val, BL-test, LPCT-val (transfer), LPCT-test (transfer).
    # The "transfer" rows are the cross-dataset story; in-sample rows give
    # the val-vs-test stability check the user asked for.
    reports: List[Dict[str, Any]] = []
    for algo in ("svm", "rf", "gb"):
        # LPCT-trained.
        model_type_lpcd = f"radiomics_{algo}"
        out_dir_lpcd = OUTPUT_ROOT / model_type_lpcd
        report = train_and_eval_algo(
            algo, inner_lpcd,
            train_df=lpcd,
            test_dfs=[
                ("dapt_val",  "Lung-PET-CT-Dx (val)",                  lpcd, "val"),
                ("dapt_test", "Lung-PET-CT-Dx (test)",                 lpcd, "test"),
                ("val",       "BigLunge (val, cross-dataset)",         bl,   "val"),
                ("test",      "BigLunge (test, cross-dataset)",        bl,   "test"),
            ],
            model_type=model_type_lpcd,
            feat_names=common,
            output_subdir=out_dir_lpcd,
        )
        report["trained_on"] = "lpcd"
        report["is_winner_for_dataset"] = (algo == inner_lpcd["winner"])
        reports.append(report)

        # BL-trained.
        model_type_bl = f"radiomics_{algo}_bl"
        out_dir_bl = OUTPUT_ROOT / model_type_bl
        report = train_and_eval_algo(
            algo, inner_bl,
            train_df=bl,
            test_dfs=[
                ("val",       "BigLunge (val, in-sample)",                bl,   "val"),
                ("test",      "BigLunge (test, in-sample)",               bl,   "test"),
                ("dapt_val",  "Lung-PET-CT-Dx (val, BL-trained transfer)", lpcd, "val"),
                ("dapt_test", "Lung-PET-CT-Dx (test, BL-trained transfer)",lpcd, "test"),
            ],
            model_type=model_type_bl,
            feat_names=common,
            output_subdir=out_dir_bl,
        )
        report["trained_on"] = "biglunge"
        report["is_winner_for_dataset"] = (algo == inner_bl["winner"])
        reports.append(report)

    summary = {
        "n_features_lpcd_pre_intersection": len(feat_lpcd),
        "n_features_bl_pre_intersection": len(feat_bl),
        "n_features_intersection": len(common),
        "inner_cv_lpcd": inner_lpcd,
        "inner_cv_biglunge": inner_bl,
        "model_reports": reports,
    }
    audit_path = RADIOMICS_DIR / "train_eval_summary.json"
    with open(audit_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[radiomics] wrote {audit_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    args = p.parse_args()
    run()


if __name__ == "__main__":
    main()
