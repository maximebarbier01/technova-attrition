from __future__ import annotations

from catboost import CatBoostClassifier


def build_catboost_model(
    random_state: int = 42,
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 3.0,
    loss_function: str = "Logloss",
    eval_metric: str = "AUC",
    verbose: int = 0,
    class_weights: list[float] | None = None,
) -> CatBoostClassifier:
    """
    Construit un modèle CatBoostClassifier.

    Parameters
    ----------
    random_state : int
        Seed.
    iterations : int
        Nombre d'itérations / arbres.
    depth : int
        Profondeur des arbres.
    learning_rate : float
        Taux d'apprentissage.
    l2_leaf_reg : float
        Régularisation L2.
    loss_function : str
        Fonction de perte.
    eval_metric : str
        Métrique interne CatBoost.
    verbose : int
        Verbosité CatBoost.
    class_weights : list[float] | None
        Pondération des classes, ex: [1.0, 3.0].

    Returns
    -------
    CatBoostClassifier
        Modèle entraînable.
    """
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function=loss_function,
        eval_metric=eval_metric,
        random_seed=random_state,
        verbose=verbose,
        class_weights=class_weights,
    )


def get_catboost_param_grid() -> dict:
    """
    Param grid pour GridSearchCV.
    """
    return {
        "iterations": [300, 500, 700],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
        "class_weights": [None, [1.0, 2.0], [1.0, 3.0], [1.0, 5.0]],
    }


def get_catboost_param_distributions() -> dict:
    """
    Paramètres à tester via RandomizedSearchCV.
    """
    return {
        "iterations": [200, 300, 500, 700, 1000],
        "depth": [4, 5, 6, 7, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1.0, 2.0, 3.0, 5.0, 7.0, 9.0],
        "class_weights": [None, [1.0, 2.0], [1.0, 3.0], [1.0, 5.0], [1.0, 7.0]],
    }