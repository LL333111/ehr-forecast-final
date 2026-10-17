"""Training-only consensus feature selection."""

from __future__ import annotations

import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .io import write


def validate_fit(x, y):
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or (not len(x))
        or np.isinf(x).any()
        or (not np.isin(y, [0, 1]).all())
    ):
        raise ValueError("Invalid training arrays")


def scale_fit(x):
    """Robust asinh scaling with a separate train-supported missing channel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(x, axis=0)
        low, high = np.nanquantile(x, [0.25, 0.75], axis=0)
        std = np.nanstd(x, axis=0)
    median = np.nan_to_num(median, nan=0.0)
    scale = np.where(high - low > 1e-06, high - low, np.where(std > 1e-06, std, 1.0))
    return dict(
        median=median.astype(np.float32),
        scale=scale.astype(np.float32),
        missing=np.flatnonzero(np.isnan(x).any(axis=0)),
    )


def scale_transform(x, state):
    z = np.arcsinh((np.where(np.isnan(x), state["median"], x) - state["median"]) / state["scale"])
    z = np.clip(z, -8, 8).astype(np.float32)
    return np.concatenate([z, np.isnan(x[:, state["missing"]]).astype(np.float32)], axis=1)


def binned_mutual_information(x, y):
    """Train-quantile five-bin MI plus a separate missing category, per target."""
    validate_fit(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        edges = np.nanquantile(x, [0.2, 0.4, 0.6, 0.8], axis=0)
    bins = np.ones(x.shape, dtype=np.uint8)
    for edge in edges:
        bins += (x > edge).astype(np.uint8)
    bins[np.isnan(x)] = 0
    n = len(x)
    p = y.mean(axis=0).astype(np.float64)
    result = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    yf = y.astype(np.float32)
    for b in range(6):
        membership = (bins == b).astype(np.float32)
        pb = membership.mean(axis=0, dtype=np.float64)[:, None]
        positive = membership.T @ yf / n
        negative = np.maximum(pb - positive, 0.0)
        for joint, targetprob in [(positive, p), (negative, 1 - p)]:
            denominator = pb * targetprob[None, :]
            ratio = np.divide(
                joint, denominator, out=np.ones_like(result), where=(joint > 0) & (denominator > 0)
            )
            result += np.where(joint > 0, joint * np.log(np.maximum(ratio, 1e-15)), 0.0)
    entropy = -(p * np.log(np.maximum(p, 1e-15)) + (1 - p) * np.log(np.maximum(1 - p, 1e-15)))
    result = np.divide(
        result, entropy[None, :], out=np.zeros_like(result), where=entropy[None, :] > 1e-10
    )
    return np.maximum(result, 0.0).astype(np.float32)


def combine_rank(signal):
    maxima = signal.max(axis=0, keepdims=True)
    normalized = np.divide(signal, maxima, out=np.zeros_like(signal), where=maxima > 0)
    return 0.5 * normalized.mean(axis=1) + 0.5 * normalized.max(axis=1)


def choose(score, per_target, k, protected_per_target=4, eligible=None):
    eligible = (
        np.ones(len(score), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    )
    protected = set()
    for j in range(per_target.shape[1]):
        order = np.argsort(-per_target[:, j], kind="stable")
        protected.update(
            (int(i) for i in order[:protected_per_target] if per_target[i, j] > 0 and eligible[i])
        )
    ordered = sorted(protected, key=lambda i: (-float(score[i]), i))
    ordered += [
        int(i) for i in np.argsort(-score, kind="stable") if i not in protected and eligible[i]
    ]
    return np.asarray(ordered[: min(k, len(ordered))], dtype=np.int64)


def select_features(x, y, quality_columns, folder, *, quick=False, threads=2):
    validate_fit(x, y)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cols = np.asarray(quality_columns, dtype=np.int64)
    if not len(cols) or len(set(cols)) != len(cols) or cols.min() < 0 or (cols.max() >= x.shape[1]):
        raise ValueError("Invalid quality feature indices")
    xq = x[:, cols]
    mi = binned_mutual_information(xq, y)
    from lightgbm import LGBMClassifier

    def tree_one(j):
        if len(np.unique(y[:, j])) < 2:
            return (np.zeros(xq.shape[1]), None)
        model = LGBMClassifier(
            n_estimators=5 if quick else 64,
            num_leaves=7,
            learning_rate=0.05,
            min_child_samples=2 if quick else 20,
            max_bin=63,
            reg_lambda=20,
            colsample_bytree=0.8,
            verbosity=-1,
            n_jobs=threads,
            random_state=2718 + j,
        )
        model.fit(xq, y[:, j])
        return (model.booster_.feature_importance(importance_type="gain"), model)

    with ThreadPoolExecutor(max_workers=1 if quick else 4) as pool:
        tree_results = list(pool.map(tree_one, range(y.shape[1])))
    gain = np.stack([v[0] for v in tree_results], axis=1).astype(np.float32)
    candidate = choose(
        combine_rank(mi) + combine_rank(gain),
        mi + gain / (gain.max(axis=0, keepdims=True) + 1e-12),
        2048,
    )
    scaler = scale_fit(xq[:, candidate])
    z = scale_transform(xq[:, candidate], scaler)
    coef = np.zeros_like(mi)
    linear_models = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            linear_models.append(None)
            continue
        m = LogisticRegression(
            l1_ratio=1.0, solver="liblinear", C=0.1, max_iter=1000, random_state=2718 + j
        )
        m.fit(z, y[:, j])
        linear_models.append(m)
        values = np.abs(m.coef_[0, : len(candidate)])
        values[scaler["missing"]] += np.abs(m.coef_[0, len(candidate) :])
        coef[candidate, j] = values
    votes = np.zeros(len(cols), dtype=np.int64)
    rng = np.random.default_rng(20261016)
    for _ in range(3):
        subset = rng.choice(len(xq), max(2, int(len(xq) * 0.8)), replace=False)
        sampled = binned_mutual_information(xq[subset], y[subset])
        voted = choose(combine_rank(sampled), sampled, 1024)
        votes[voted] += 1
    score = (combine_rank(mi) + combine_rank(gain) + combine_rank(coef)) / 3
    per = (
        mi / (mi.max(axis=0, keepdims=True) + 1e-12)
        + gain / (gain.max(axis=0, keepdims=True) + 1e-12)
        + coef / (coef.max(axis=0, keepdims=True) + 1e-12)
    )
    compact = choose(score, per, 512)
    stable = choose(score, per, 256, protected_per_target=2, eligible=votes >= 2)
    if not len(stable):
        raise ValueError("No stable features")
    np.savez_compressed(
        folder / "selection.npz",
        quality=cols,
        compact=cols[compact],
        stable=cols[stable],
        mutual_information=mi,
        gain=gain,
        linear_coefficient=coef,
        stability_votes=votes,
        consensus=score,
    )
    joblib.dump(
        dict(
            trees=[v[1] for v in tree_results],
            linear=linear_models,
            linear_candidate_columns=cols[candidate],
            linear_scaler=scaler,
        ),
        folder / "selector_models.joblib",
        compress=3,
    )
    report = dict(
        training_rows=len(x),
        targets=y.shape[1],
        all_width=x.shape[1],
        quality_width=len(cols),
        compact_width=len(compact),
        stable_width=len(stable),
        methods=[
            "normalized binned mutual information",
            "regularized shallow tree gain",
            "L1 logistic coefficients",
            "three 80% train-only MI subsample votes",
        ],
        compact_policy="512 consensus ranks, up to four protected features per supported target",
        stable_policy="up to 256 consensus ranks supported by at least two train-only subsamples",
        scoring_labels_used=False,
        outer_permutation_used_for_selection=False,
        train_x_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        train_y_sha256=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
    )
    write(folder / "selection.json", report)
    return report


def binary_model(seed=0, quick=False, threads=2):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=5 if quick else 150,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=2 if quick else 20,
        max_bin=127,
        reg_lambda=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbosity=-1,
        n_jobs=threads,
        random_state=seed,
    )
