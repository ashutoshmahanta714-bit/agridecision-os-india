"""Cross-fitted doubly robust estimation for observational interventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CausalEstimate:
    average_treatment_effect: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    propensity_min: float
    propensity_max: float
    sample_size: int


def _propensity_model() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )


def _outcome_model() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06)),
        ]
    )


def doubly_robust_ate(
    frame: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    folds: int = 5,
    propensity_clip: float = 0.03,
    random_state: int = 42,
) -> CausalEstimate:
    """Estimate an ATE using cross-fitted augmented inverse propensity weighting.

    This estimates association under explicit causal assumptions; it does not
    prove causality. Covariates must be measured before treatment.
    """

    required = {treatment, outcome, *covariates}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Causal frame is missing: {missing}")
    working = frame.dropna(subset=[treatment, outcome]).copy()
    x = working[covariates]
    t = working[treatment].astype(int).to_numpy()
    y = working[outcome].astype(float).to_numpy()
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1")
    if min(np.bincount(t)) < folds:
        raise ValueError("Each treatment group needs at least one row per fold")

    propensity = np.empty(len(working), dtype=float)
    outcome_if_treated = np.empty(len(working), dtype=float)
    outcome_if_control = np.empty(len(working), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    for train_index, test_index in splitter.split(x, t):
        propensity_model = _propensity_model().fit(x.iloc[train_index], t[train_index])
        propensity[test_index] = propensity_model.predict_proba(x.iloc[test_index])[:, 1]

        treated_train = train_index[t[train_index] == 1]
        control_train = train_index[t[train_index] == 0]
        treated_model = _outcome_model().fit(x.iloc[treated_train], y[treated_train])
        control_model = _outcome_model().fit(x.iloc[control_train], y[control_train])
        outcome_if_treated[test_index] = treated_model.predict(x.iloc[test_index])
        outcome_if_control[test_index] = control_model.predict(x.iloc[test_index])

    propensity = np.clip(propensity, propensity_clip, 1 - propensity_clip)
    influence = (
        outcome_if_treated
        - outcome_if_control
        + t * (y - outcome_if_treated) / propensity
        - (1 - t) * (y - outcome_if_control) / (1 - propensity)
    )
    effect = float(np.mean(influence))
    standard_error = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    return CausalEstimate(
        average_treatment_effect=effect,
        standard_error=standard_error,
        confidence_interval_95=(effect - 1.96 * standard_error, effect + 1.96 * standard_error),
        propensity_min=float(propensity.min()),
        propensity_max=float(propensity.max()),
        sample_size=len(working),
    )
