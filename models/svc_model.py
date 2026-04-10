from __future__ import annotations

from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC


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


def build_svc_pipeline(
    preprocessor,
    random_state: int = 42,
    sampling_method: str | None = None,
    C: float = 1.0,
    kernel: str = 'rbf',
    gamma: str | float = 'scale',
    degree: int = 3,
    coef0: float = 0.0,
    class_weight: str | dict | None = 'balanced',
    probability: bool = True,
) -> ImbPipeline:
    """
    Construit un pipeline SVC avec ou sans sampling.
    """
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state,
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


def get_svc_param_distributions() -> dict:
    """
    Param?tres ? tester via RandomizedSearchCV.
    """
    return {
        'model__C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20],
        'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'model__degree': [2, 3, 4],
        'model__coef0': [0.0, 0.5, 1.0],
        'model__class_weight': [None, 'balanced'],
    }
