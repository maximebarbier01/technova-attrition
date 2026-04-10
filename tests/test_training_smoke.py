# Pour vérifier qu’un entraînement minimal va jusqu’au bout sans planter

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_preprocessor
from src.modeling.train import train_model

def test_train_model_smoke():
    X_train = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50],
            "revenu": [2000, 2500, 3000, 3500, 4000, 4500],
            "departement": ["IT", "RH", "IT", "Finance", "RH", "IT"],
        }
    )
    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    preprocessor = build_preprocessor(
        num_features=["age", "revenu"],
        cat_features=["departement"],
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    trained_model = train_model(model, X_train, y_train)

    preds = trained_model.predict(X_train)

    assert len(preds) == len(X_train)
