"""Image-quality feature extraction for crop/produce photographs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_image_features(path: str | Path) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((224, 224)), dtype=float) / 255.0
    gray = rgb.mean(axis=2)
    gradient_y, gradient_x = np.gradient(gray)
    features: dict[str, float] = {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "sharpness": float(np.mean(gradient_x**2 + gradient_y**2)),
        "red_mean": float(rgb[:, :, 0].mean()),
        "green_mean": float(rgb[:, :, 1].mean()),
        "blue_mean": float(rgb[:, :, 2].mean()),
        "red_std": float(rgb[:, :, 0].std()),
        "green_std": float(rgb[:, :, 1].std()),
        "blue_std": float(rgb[:, :, 2].std()),
    }
    for channel_index, channel_name in enumerate(("red", "green", "blue")):
        histogram, _ = np.histogram(rgb[:, :, channel_index], bins=8, range=(0, 1), density=True)
        for bin_index, value in enumerate(histogram):
            features[f"{channel_name}_hist_{bin_index}"] = float(value)
    return features


def image_feature_table(paths: list[str | Path]) -> pd.DataFrame:
    rows = [{"image_path": str(path), **extract_image_features(path)} for path in paths]
    return pd.DataFrame(rows)


def make_image_quality_model(*, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
