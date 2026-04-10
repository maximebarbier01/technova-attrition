# Pour vérifier que le split marche et conserve bien les classes

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.split_data import make_train_test_split, split_features_target


def test_split_features_target_returns_expected_columns():
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "revenu": [2000, 2500, 3000],
            "target": [0, 1, 0],
        }
    )

    X, y = split_features_target(
        df=df,
        feature_columns=["age", "revenu"],
        target_column="target",
    )

    assert list(X.columns) == ["age", "revenu"]
    assert y.name == "target"
    assert len(X) == len(y) == 3


def test_make_train_test_split_keeps_all_rows():
    df = pd.DataFrame(
        {
            "age": list(range(20)),
            "revenu": [2000 + i * 100 for i in range(20)],
            "target": [0] * 10 + [1] * 10,
        }
    )

    X_train, X_test, y_train, y_test = make_train_test_split(
        df=df,
        feature_columns=["age", "revenu"],
        target_column="target",
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) + len(y_test) == 20
