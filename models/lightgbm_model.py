from __future__ import annotations

from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier


def _get_sampling_steps(
    sampling_method: str | None,
    random_state: int = 42,
) -> list[tuple[str, object]]:
    """
    Retourne les ?tapes de sampling ? ins?rer dans le pipeline.
    """
    if sampling_method is None:
        return []

    if sampling_method == "smote":
        return [
            (
                "smote",
                SMOTE(
                    sampling_strategy=0.3,
                    k_neighbors=5,
                    random_state=random_state,
                ),
            )
        ]

    if sampling_method == "borderline":
        return [
            (
                "smote",
                BorderlineSMOTE(
                    sampling_strategy=0.3,
                    k_neighbors=5,
                    random_state=random_state,
                ),
            )
        ]

    if sampling_method == "smote_under":
        return [
            (
                "smote",
                SMOTE(
                    sampling_strategy=0.3,
                    k_neighbors=5,
                    random_state=random_state,
                ),
            ),
            (
                "under",
                RandomUnderSampler(
                    sampling_strategy=0.7,
                    random_state=random_state,
                ),
            ),
        ]

    raise ValueError(
        f"sampling_method non support?e : {sampling_method}. "
        "Valeurs possibles : None, 'smote', 'borderline', 'smote_under'."
    )


def build_lightgbm_pipeline(
    preprocessor,
    random_state: int = 42,
    sampling_method: str | None = None,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    class_weight: str | dict | None = 'balanced',
    n_jobs: int = -1,
) -> ImbPipeline:
    """
    Construit un pipeline LightGBM avec ou sans sampling.
    """
    model = LGBMClassifier(
        objective='binary',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
    )

    steps = [("prep", preprocessor)]
    steps.extend(
        _get_sampling_steps(
            sampling_method=sampling_method,
            random_state=random_state,
        )
    )
    steps.append(("model", model))

    return ImbPipeline(steps=steps)


def get_lightgbm_param_distributions() -> dict:
    """
    Param?tres ? tester via RandomizedSearchCV.
    """
    return {
        'model__n_estimators': [100, 200, 300, 500, 700],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'model__num_leaves': [15, 31, 63, 127],
        'model__max_depth': [-1, 3, 5, 7, 10],
        'model__min_child_samples': [10, 20, 30, 50],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__reg_alpha': [0.0, 0.01, 0.1, 0.5, 1.0],
        'model__reg_lambda': [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
        'model__class_weight': [None, 'balanced'],
    }
