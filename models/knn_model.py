from __future__ import annotations

from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier


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


def build_knn_pipeline(
    preprocessor,
    random_state: int = 42,
    sampling_method: str | None = None,
    n_neighbors: int = 15,
    weights: str = 'distance',
    p: int = 2,
    leaf_size: int = 30,
    n_jobs: int = -1,
) -> ImbPipeline:
    """
    Construit un pipeline KNeighborsClassifier avec ou sans sampling.
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        leaf_size=leaf_size,
        metric='minkowski',
        n_jobs=n_jobs,
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


def get_knn_param_distributions() -> dict:
    """
    Param?tres ? tester via RandomizedSearchCV.
    """
    return {
        'model__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2],
        'model__leaf_size': [20, 30, 40, 60],
    }
