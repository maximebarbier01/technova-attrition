from __future__ import annotations

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def train_model(model, X_train, y_train, **fit_kwargs):
    """
    Entraîne un modèle ou pipeline.

    Parameters
    ----------
    model :
        Modèle ou pipeline sklearn-compatible.
    X_train :
        Features d'entraînement.
    y_train :
        Target d'entraînement.
    **fit_kwargs :
        Arguments supplémentaires passés à model.fit().

    Returns
    -------
    model
        Modèle entraîné.
    """
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_model_with_gridsearch(
    model,
    X_train,
    y_train,
    param_grid: dict,
    scoring: str = "average_precision",
    cv: int = 5,
    n_jobs: int = -1,
    refit: bool = True,
    verbose: int = 1,
    fit_kwargs: dict | None = None,
):
    """
    Entraîne un modèle avec GridSearchCV.

    Parameters
    ----------
    model :
        Modèle ou pipeline sklearn-compatible.
    X_train :
        Features d'entraînement.
    y_train :
        Target d'entraînement.
    param_grid : dict
        Dictionnaire d'hyperparamètres pour GridSearchCV.
    scoring : str, default="average_precision"
        Métrique d'évaluation CV.
    cv : int, default=5
        Nombre de folds.
    n_jobs : int, default=-1
        Nombre de coeurs CPU.
    refit : bool, default=True
        Réentraîne le meilleur modèle sur tout X_train.
    verbose : int, default=0
        Niveau de verbosité.
    fit_kwargs : dict | None, default=None
        Arguments supplémentaires passés à .fit().

    Returns
    -------
    GridSearchCV
        Objet GridSearchCV entraîné.
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
    )

    grid.fit(X_train, y_train, **fit_kwargs)
    return grid


def train_model_with_randomized_search(
    model,
    X_train,
    y_train,
    param_distributions: dict,
    n_iter: int = 20,
    scoring: str = "average_precision",
    cv: int = 5,
    n_jobs: int = -1,
    refit: bool = True,
    verbose: int = 0,
    random_state: int = 42,
    fit_kwargs: dict | None = None,
):
    """
    Entraîne un modèle avec RandomizedSearchCV.

    Parameters
    ----------
    model :
        Modèle ou pipeline sklearn-compatible.
    X_train :
        Features d'entraînement.
    y_train :
        Target d'entraînement.
    param_distributions : dict
        Dictionnaire d'hyperparamètres pour RandomizedSearchCV.
    n_iter : int, default=20
        Nombre de combinaisons testées.
    scoring : str, default="average_precision"
        Métrique d'évaluation CV.
    cv : int, default=5
        Nombre de folds.
    n_jobs : int, default=-1
        Nombre de coeurs CPU.
    refit : bool, default=True
        Réentraîne le meilleur modèle sur tout X_train.
    verbose : int, default=0
        Niveau de verbosité.
    random_state : int, default=42
        Seed.
    fit_kwargs : dict | None, default=None
        Arguments supplémentaires passés à .fit().

    Returns
    -------
    RandomizedSearchCV
        Objet RandomizedSearchCV entraîné.
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
        random_state=random_state,
    )

    random_search.fit(X_train, y_train, **fit_kwargs)
    return random_search