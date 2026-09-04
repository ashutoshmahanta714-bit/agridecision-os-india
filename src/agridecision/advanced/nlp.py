"""Agricultural bulletin text classification and keyword explanation."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def make_bulletin_risk_model(*, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=15_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def top_explanatory_terms(model: Pipeline, *, count: int = 15) -> dict[str, list[str]]:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    terms = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    return {
        "higher_risk": terms[np.argsort(coefficients)[-count:][::-1]].tolist(),
        "lower_risk": terms[np.argsort(coefficients)[:count]].tolist(),
    }
