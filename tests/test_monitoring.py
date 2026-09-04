import numpy as np
import pandas as pd

from agridecision.monitoring.drift import numeric_drift_report, population_stability_index


def test_drift_detects_distribution_change():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current = pd.Series(rng.normal(2, 1, 2000))
    assert population_stability_index(reference, current) > 0.25
    report = numeric_drift_report(
        pd.DataFrame({"price": reference}), pd.DataFrame({"price": current}), ["price"]
    )
    assert report.loc[0, "status"] == "alert"
