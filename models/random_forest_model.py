from __future__ import annotations

from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier


def _get_sampling_steps(
    sampling_method: str | None,
    random_state: int = 42,
) -> list[tuple[str, object]]:
    """
    Retourne les étapes de sampling à insérer dans le pipeline.

    Parameters
    ----------
    sampling_method : str | None
        Méthode de sampling à utiliser.
        Valeurs supportées :
        - None
        - "smote"
        - "borderline"
        - "smote_under"
    random_state : int
        Seed globale.

    Returns
    -------
    list[tuple[str, object]]
        Liste d'étapes pour le pipeline imblearn.
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
        f"sampling_method non supportée : {sampling_method}. "
        "Valeurs possibles : None, 'smote', 'borderline', 'smote_under'."
    )


def build_random_forest_pipeline(
    preprocessor,
    random_state: int = 42,
    sampling_method: str | None = None,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | int | float | None = "sqrt",
    class_weight: str | dict | None = "balanced",
    n_jobs: int = -1,
) -> ImbPipeline:
    """
    Construit un pipeline Random Forest avec ou sans sampling.

    Parameters
    ----------
    preprocessor :
        Preprocessor sklearn (ColumnTransformer).
    random_state : int
        Seed.
    sampling_method : str | None
        Méthode de sampling à utiliser :
        - None
        - "smote"
        - "borderline"
        - "smote_under"
    n_estimators : int
        Nombre d'arbres.
    max_depth : int | None
        Profondeur maximale des arbres.
    min_samples_split : int
        Nombre minimal d'observations pour splitter un noeud.
    min_samples_leaf : int
        Nombre minimal d'observations dans une feuille.
    max_features : str | int | float | None
        Nombre de variables testées à chaque split.
    class_weight : str | dict | None
        Pondération des classes.
    n_jobs : int
        Nombre de coeurs CPU.

    Returns
    -------
    ImbPipeline
        Pipeline entraînable.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
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


def get_random_forest_param_grid() -> dict:
    """
    Param grid pour GridSearchCV.
    """
    return {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }


def get_random_forest_param_distributions() -> dict:
    """
    Paramètres à tester via RandomizedSearchCV.
    """
    return {
        "model__n_estimators": [100, 200, 300, 500, 700],
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__min_samples_split": [2, 5, 10, 15],
        "model__min_samples_leaf": [1, 2, 4, 6],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }