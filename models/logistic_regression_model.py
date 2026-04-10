from __future__ import annotations

from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression


def _get_sampling_steps(
    sampling_method: str | None,
    random_state: int = 42,
) -> list[tuple[str, object]]:
    """
    Retourne les ?tapes de sampling ? ins?rer dans le pipeline.

    Parameters
    ----------
    sampling_method : str | None
        M?thode de sampling ? utiliser.
        Valeurs support?es :
        - None
        - "smote"
        - "borderline"
        - "smote_under"
    random_state : int
        Seed globale.

    Returns
    -------
    list[tuple[str, object]]
        Liste d'?tapes pour le pipeline imblearn.
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


def build_logistic_regression_pipeline(
    preprocessor,
    random_state: int = 42,
    sampling_method: str | None = None,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
    solver: str | None = None,
    l1_ratio: float | None = None,
) -> ImbPipeline:
    """
    Construit un pipeline de r?gression logistique avec ou sans sampling.

    Parameters
    ----------
    preprocessor :
        ColumnTransformer ou pipeline de preprocessing.
    random_state : int
        Seed.
    sampling_method : str | None
        M?thode de sampling ? utiliser :
        - None
        - "smote"
        - "borderline"
        - "smote_under"
    penalty : str
        'l1' pour Lasso, 'l2' pour Ridge, 'elasticnet' pour Elastic Net.
    C : float
        Inverse de la r?gularisation.
    class_weight : str | dict | None
        Pond?ration des classes.
    max_iter : int
        Nombre maximum d'it?rations.
    solver : str | None
        Solveur sklearn. Si None, choisi automatiquement.
    l1_ratio : float | None
        Part de r?gularisation L1 si penalty='elasticnet'.

    Returns
    -------
    ImbPipeline
        Pipeline complet entra?nable.
    """
    if solver is None:
        if penalty == "l1":
            solver = "liblinear"
        elif penalty == "l2":
            solver = "liblinear"
        elif penalty == "elasticnet":
            solver = "saga"
        else:
            raise ValueError(f"Penalty non support?e : {penalty}")

    if penalty == "elasticnet" and l1_ratio is None:
        l1_ratio = 0.5

    model = LogisticRegression(
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        C=C,
        l1_ratio=l1_ratio,
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


def build_lasso_logistic_regression_pipeline(
    preprocessor,
    random_state: int = 42,
    sampling_method: str | None = None,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
) -> ImbPipeline:
    """
    Construit une r?gression logistique L1 (Lasso) avec ou sans sampling.
    """
    return build_logistic_regression_pipeline(
        preprocessor=preprocessor,
        random_state=random_state,
        sampling_method=sampling_method,
        penalty="l1",
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver="liblinear",
    )


def build_elastic_net_logistic_regression_pipeline(
    preprocessor,
    random_state: int = 42,
    sampling_method: str | None = None,
    C: float = 1.0,
    l1_ratio: float = 0.5,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 3000,
) -> ImbPipeline:
    """
    Construit une r?gression logistique Elastic Net avec ou sans sampling.
    """
    return build_logistic_regression_pipeline(
        preprocessor=preprocessor,
        random_state=random_state,
        sampling_method=sampling_method,
        penalty="elasticnet",
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver="saga",
        l1_ratio=l1_ratio,
    )


def get_logistic_regression_param_grid() -> dict:
    """
    Param grid pour GridSearchCV sur une logistic classique / L2.
    """
    return {
        "model__penalty": ["l2"],
        "model__C": [0.001, 0.01, 0.1, 1, 5, 10, 20],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [2000],
    }


def get_lasso_logistic_regression_param_grid() -> dict:
    """
    Param grid pour GridSearchCV sur une logistic L1 (Lasso).
    """
    return {
        "model__penalty": ["l1"],
        "model__solver": ["liblinear"],
        "model__C": [0.001, 0.01, 0.1, 1, 5, 10, 20],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [2000],
    }


def get_logistic_regression_param_distributions() -> dict:
    """
    Param?tres ? tester via RandomizedSearchCV pour une logistic classique / L2.
    """
    return {
        "model__penalty": ["l2"],
        "model__C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [1000, 2000, 3000],
    }


def get_lasso_logistic_regression_param_distributions() -> dict:
    """
    Param?tres ? tester via RandomizedSearchCV pour une logistic L1 (Lasso).
    """
    return {
        "model__penalty": ["l1"],
        "model__solver": ["liblinear"],
        "model__C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [1000, 2000, 3000],
    }


def get_elastic_net_logistic_regression_param_distributions() -> dict:
    """
    Param?tres ? tester via RandomizedSearchCV pour une logistic Elastic Net.
    """
    return {
        "model__penalty": ["elasticnet"],
        "model__solver": ["saga"],
        "model__C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20],
        "model__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [2000, 3000, 5000],
    }
