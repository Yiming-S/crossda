"""
pipeline_utils.py — Shared utilities for MAP, MMP, and BDP pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from da4bci.metrics.distance import (
    compute_mmd,
    compute_wasserstein,
    compute_energy,
    compute_mahalanobis,
)
from da4bci.geometry.spd import compute_geodesic

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from pyriemann.classification import MDM


###############################################################################
# Distance
###############################################################################

def session_distance(
    Xs: NDArray, Xt: NDArray, dist_type: str, dist_param: Optional[dict] = None,
) -> float:
    dist_param = dist_param or {}
    dispatch = {
        "mmd": lambda: compute_mmd(Xs, Xt, sigma=dist_param.get("sigma", 1.0)),
        "wasserstein": lambda: compute_wasserstein(Xs, Xt),
        "energy": lambda: compute_energy(Xs, Xt),
        "geodesic": lambda: compute_geodesic(Xs, Xt),
        "mahalanobis": lambda: compute_mahalanobis(Xs, Xt),
    }
    if dist_type not in dispatch:
        raise ValueError(f"Unknown distance type: {dist_type!r}")
    return float(dispatch[dist_type]())


def distance_ci(
    X: NDArray, Y: NDArray, dist_type: str, dist_param: Optional[dict] = None,
    B: int = 200, alpha: float = 0.05, est_method: str = "direct",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Bootstrap confidence interval for session_distance."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = X.shape[0], Y.shape[0]
    if n < 1 or m < 1:
        return dict(est=np.inf, lwr=np.inf, upr=np.inf)

    if rng is None:
        rng = np.random.default_rng()

    d_hat = np.nan
    if est_method == "direct":
        try:
            d_hat = session_distance(X, Y, dist_type, dist_param)
        except Exception:
            d_hat = np.inf

    vals = np.full(B, np.nan)
    for b in range(B):
        try:
            vals[b] = session_distance(
                X[rng.integers(n, size=n)], Y[rng.integers(m, size=m)],
                dist_type, dist_param,
            )
        except Exception:
            pass

    finite = vals[np.isfinite(vals)]
    if len(finite) < 2:
        est = d_hat if np.isfinite(d_hat) else np.inf
        return dict(est=est, lwr=np.inf, upr=np.inf)

    lwr = float(np.quantile(finite, alpha / 2))
    upr = float(np.quantile(finite, 1 - alpha / 2))
    est = float(np.mean(finite)) if est_method == "boot_mean" else d_hat
    return dict(est=est, lwr=lwr, upr=upr)


###############################################################################
# Feature scaling
###############################################################################

def scale_to_target(
    X_list: List[NDArray], X_target: NDArray,
) -> Tuple[List[NDArray], NDArray]:
    """Standardise sources + target to the target's column-wise mean and sd."""
    X_target = np.asarray(X_target, dtype=float)
    center = X_target.mean(axis=0)
    sds = X_target.std(axis=0, ddof=1)
    sds[~np.isfinite(sds) | (sds <= 0)] = 1.0
    scaled_sources = [(np.asarray(X, dtype=float) - center) / sds for X in X_list]
    scaled_target = (X_target - center) / sds
    return scaled_sources, scaled_target


###############################################################################
# Feature extraction
###############################################################################

def extract_features_train(
    X: NDArray, y: NDArray, feat_name: str, feat_params: Optional[dict] = None,
) -> Tuple[NDArray, Any]:
    """
    Fit a feature extractor on training data.

    Parameters
    ----------
    X : 3D array (n_trials, n_channels, n_times)
    y : labels
    feat_name : "logvar" | "CSP" | "TS"

    Returns (features, fitted_object)
    """
    feat_params = feat_params or {}

    if feat_name == "logvar":
        return _logvar(X), {"type": "logvar"}

    if feat_name == "CSP":
        n_components = feat_params.get("n_components", feat_params.get("ncomps", 8))
        csp = CSP(n_components=n_components, log=True)
        return csp.fit_transform(X, y), {"type": "CSP", "csp": csp}

    if feat_name == "TS":
        cov_est = Covariances(estimator="lwf")
        ts = TangentSpace()
        covs = cov_est.fit_transform(X)
        return ts.fit_transform(covs, y), {"type": "TS", "cov_est": cov_est, "ts": ts}

    raise ValueError(f"Unknown feature: {feat_name!r}. Supported: logvar, CSP, TS")


def extract_features_test(
    X: NDArray, fitted_obj: Any, feat_name: str,
) -> NDArray:
    """Apply a fitted feature extractor to new data."""
    if feat_name == "logvar":
        return _logvar(X)
    if feat_name == "CSP":
        return fitted_obj["csp"].transform(X)
    if feat_name == "TS":
        return fitted_obj["ts"].transform(fitted_obj["cov_est"].transform(X))
    raise ValueError(f"Unknown feature: {feat_name!r}")


def _logvar(X: NDArray) -> NDArray:
    """Log-variance features: (n_trials, n_channels)."""
    return np.log(np.var(X, axis=2) + 1e-10)


###############################################################################
# Classifier dispatch
###############################################################################

def get_classifier(clf_name: str, clf_params: Optional[dict] = None) -> Any:
    """Create an sklearn-compatible classifier instance."""
    p = clf_params or {}

    if clf_name == "lda":
        return LinearDiscriminantAnalysis(shrinkage=p.get("shrinkage", "auto"), solver="lsqr")

    if clf_name == "svm_linear":
        return SVC(kernel="linear", C=p.get("cost", p.get("C", 0.1)))

    if clf_name == "svm_radial":
        gamma = p.get("gamma", 0.1)
        return SVC(kernel="rbf", C=p.get("cost", p.get("C", 0.1)),
                   gamma="scale" if gamma == -1 else gamma)

    if clf_name == "mdm":
        return MDM()

    if clf_name == "el":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            l1_ratio=p.get("alpha", 0.5), max_iter=100000, random_state=42,
        )

    if clf_name == "lr":
        return LogisticRegression(penalty="l2", max_iter=20000, solver="lbfgs")

    if clf_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            num_leaves=p.get("num_leaves", 31), learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", -1), n_estimators=p.get("nrounds", 200),
            reg_alpha=p.get("lambda_l1", 0.1), reg_lambda=p.get("lambda_l2", 0.1),
            subsample=p.get("bagging_fraction", 0.8), colsample_bytree=p.get("feature_fraction", 0.8),
            verbose=-1, random_state=p.get("seed", 123),
        )

    if clf_name == "elm":
        from sklearn.kernel_approximation import RBFSampler
        from sklearn.linear_model import RidgeClassifier
        return make_pipeline(RBFSampler(n_components=p.get("nhid", 500), random_state=1), RidgeClassifier())

    if clf_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            depth=p.get("depth", 6), learning_rate=p.get("lr", 0.05),
            iterations=p.get("iters", 500), early_stopping_rounds=p.get("es_round", 30), verbose=0,
        )

    raise ValueError(f"Unknown classifier: {clf_name!r}")
