from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression


def build_logistic_regression_pipeline(
    preprocessor,
    random_state: int = 42,
    use_smote: bool = True,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
    solver: str | None = None,
) -> ImbPipeline:
    """
    Construit un pipeline de régression logistique avec ou sans SMOTE.

    Parameters
    ----------
    preprocessor :
        ColumnTransformer ou pipeline de preprocessing.
    random_state : int
        Seed.
    use_smote : bool
        Active ou non le SMOTE dans le pipeline.
    penalty : str
        'l1' pour Lasso, 'l2' pour Ridge.
    C : float
        Inverse de la régularisation.
    class_weight : str | dict | None
        Pondération des classes.
    max_iter : int
        Nombre maximum d'itérations.
    solver : str | None
        Solveur sklearn. Si None, choisi automatiquement.

    Returns
    -------
    ImbPipeline
        Pipeline complet entraînable.
    """
    if solver is None:
        if penalty == "l1":
            solver = "liblinear"
        elif penalty == "l2":
            solver = "liblinear"
        else:
            raise ValueError(f"Penalty non supportée : {penalty}")

    model = LogisticRegression(
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        C=C,
    )

    steps = [("prep", preprocessor)]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))

    steps.append(("model", model))

    return ImbPipeline(steps=steps)


def build_lasso_logistic_regression_pipeline(
    preprocessor,
    random_state: int = 42,
    use_smote: bool = True,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
) -> ImbPipeline:
    """
    Construit une régression logistique L1 (Lasso).
    """
    return build_logistic_regression_pipeline(
        preprocessor=preprocessor,
        random_state=random_state,
        use_smote=use_smote,
        penalty="l1",
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver="liblinear",
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
    Paramètres à tester via RandomizedSearchCV pour une logistic classique / L2.
    """
    return {
        "model__penalty": ["l2"],
        "model__C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [1000, 2000, 3000],
    }


def get_lasso_logistic_regression_param_distributions() -> dict:
    """
    Paramètres à tester via RandomizedSearchCV pour une logistic L1 (Lasso).
    """
    return {
        "model__penalty": ["l1"],
        "model__solver": ["liblinear"],
        "model__C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [1000, 2000, 3000],
    }