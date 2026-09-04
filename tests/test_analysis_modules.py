import numpy as np
import pandas as pd
from PIL import Image

from agridecision.advanced.causal import doubly_robust_ate
from agridecision.advanced.clustering import build_market_profiles, segment_markets
from agridecision.advanced.graph import correlation_edges, market_centrality
from agridecision.advanced.nlp import make_bulletin_risk_model, top_explanatory_terms
from agridecision.advanced.statistics import compare_market_prices
from agridecision.advanced.vision import extract_image_features, make_image_quality_model
from agridecision.data.demo import generate_demo_mandi_data
from agridecision.features.geospatial import haversine_km


def test_statistics_clustering_graph_and_geospatial():
    result = compare_market_prices(np.arange(10), np.arange(10) + 2)
    assert result.mean_difference == -2
    demo = generate_demo_mandi_data(days=100)
    profiles = build_market_profiles(demo)
    segmented, _ = segment_markets(profiles, clusters=2)
    assert segmented["cluster"].nunique() == 2
    edges = correlation_edges(demo, minimum_absolute_correlation=0.0)
    assert not edges.empty
    assert not market_centrality(edges).empty
    assert haversine_km(0, 0, 0, 0).item() == 0


def test_nlp_and_image_feature_pipeline(tmp_path):
    text = [
        "heavy rain disrupted supply",
        "normal arrivals stable",
        "flood damaged crop",
        "good harvest normal",
    ]
    labels = np.array([1, 0, 1, 0])
    nlp = make_bulletin_risk_model().fit(text, labels)
    terms = top_explanatory_terms(nlp, count=2)
    assert len(terms["higher_risk"]) == 2

    image_path = tmp_path / "onion.png"
    Image.new("RGB", (32, 32), color=(180, 80, 50)).save(image_path)
    features = extract_image_features(image_path)
    x = pd.DataFrame([features, {key: value * 0.9 for key, value in features.items()}])
    model = make_image_quality_model().fit(x, [1, 0])
    assert model.predict(x).shape == (2,)


def test_cross_fitted_causal_estimator_recovers_signal():
    rng = np.random.default_rng(3)
    size = 320
    x1 = rng.normal(size=size)
    x2 = rng.normal(size=size)
    propensity = 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2)))
    treatment = rng.binomial(1, propensity)
    outcome = 2.0 * treatment + 0.8 * x1 - 0.2 * x2 + rng.normal(0, 0.5, size)
    frame = pd.DataFrame({"treatment": treatment, "outcome": outcome, "x1": x1, "x2": x2})
    estimate = doubly_robust_ate(
        frame, treatment="treatment", outcome="outcome", covariates=["x1", "x2"], folds=4
    )
    assert 1.2 < estimate.average_treatment_effect < 2.8
    assert estimate.propensity_min >= 0.03
