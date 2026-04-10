# Pour vérifier que le preprocessing gère bien numériques/catégorielles et les valeurs manquantes

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing import build_preprocessor, filter_existing_features


def test_filter_existing_features_keeps_only_present_columns():
    features = ["age", "revenu", "missing_col"]
    columns = ["age", "revenu", "target"]

    result = filter_existing_features(features, columns)

    assert result == ["age", "revenu"]


def test_build_preprocessor_fit_transform_runs_with_missing_values():
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 45],
            "revenu": [2000, None, 3200, 4100],
            "departement": ["IT", "RH", None, "Finance"],
        }
    )

    preprocessor = build_preprocessor(
        num_features=["age", "revenu"],
        cat_features=["departement"],
    )

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 4
    assert transformed.shape[1] >= 3
