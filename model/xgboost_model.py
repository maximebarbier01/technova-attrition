from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


def build_xgboost_pipeline(
    preprocessor,
    random_state: int = 42,
    use_smote: bool = True,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    scale_pos_weight: float = 1.0,
    n_jobs: int = -1,
) -> ImbPipeline:
    """
    Construit un pipeline XGBoost avec ou sans SMOTE.

    Parameters
    ----------
    preprocessor :
        Preprocessor sklearn (ColumnTransformer).
    random_state : int
        Seed.
    use_smote : bool
        Active ou non le SMOTE.
    n_estimators : int
        Nombre d'arbres.
    max_depth : int
        Profondeur max des arbres.
    learning_rate : float
        Taux d'apprentissage.
    subsample : float
        Sous-échantillonnage des lignes.
    colsample_bytree : float
        Sous-échantillonnage des colonnes.
    min_child_weight : int
        Poids minimal d'un noeud enfant.
    gamma : float
        Gain minimal pour effectuer un split.
    reg_alpha : float
        Régularisation L1.
    reg_lambda : float
        Régularisation L2.
    scale_pos_weight : float
        Poids de la classe positive.
    n_jobs : int
        Nombre de coeurs CPU.

    Returns
    -------
    ImbPipeline
        Pipeline entraînable.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    steps = [("prep", preprocessor)]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))

    steps.append(("model", model))

    return ImbPipeline(steps=steps)


def get_xgboost_param_grid() -> dict:
    """
    Param grid pour GridSearchCV.
    """
    return {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0.0, 0.1, 0.3],
        "model__reg_alpha": [0.0, 0.1],
        "model__reg_lambda": [1.0, 3.0],
        "model__scale_pos_weight": [1.0, 2.0, 3.0],
    }


def get_xgboost_param_distributions() -> dict:
    """
    Paramètres à tester via RandomizedSearchCV.
    """
    return {
        "model__n_estimators": [100, 200, 300, 500, 700],
        "model__max_depth": [3, 4, 5, 6, 7, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 2, 3, 5, 7],
        "model__gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
        "model__reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 3.0, 5.0],
        "model__scale_pos_weight": [1.0, 1.5, 2.0, 3.0, 5.0],
    }