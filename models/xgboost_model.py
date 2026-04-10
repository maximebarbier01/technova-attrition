from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.modeling.compare import find_best_threshold_from_proba


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


def build_xgboost_pipeline(
    preprocessor,
    random_state: int = 42,
    sampling_method: str | None = None,
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
    Construit un pipeline XGBoost avec ou sans sampling.
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
    steps.extend(
        _get_sampling_steps(
            sampling_method=sampling_method,
            random_state=random_state,
        )
    )
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


def optimize_xgboost_with_optuna(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    preprocessor,
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
    timeout: int | None = None,
    optimize_metric: str = "average_precision",
    f1_threshold: float = 0.5,
    n_jobs: int = -1,
) -> tuple[ImbPipeline, optuna.study.Study]:
    allowed_metrics = {"average_precision", "f1"}
    if optimize_metric not in allowed_metrics:
        raise ValueError(
            f"optimize_metric doit être dans {allowed_metrics}, reçu: {optimize_metric}"
        )

    X_data = X
    y_data = np.asarray(y)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def objective(trial: optuna.trial.Trial) -> float:
        sampling_method = trial.suggest_categorical(
            "sampling_method",
            [None, "smote", "borderline", "smote_under"],
        )

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": 1.0,
        }

        # scale_pos_weight n'a du sens que sans sampling explicite
        if sampling_method is None:
            params["scale_pos_weight"] = trial.suggest_float(
                "scale_pos_weight", 1.0, 10.0
            )

        scores = []

        for train_idx, valid_idx in skf.split(X_data, y_data):
            if isinstance(X_data, pd.DataFrame):
                X_train = X_data.iloc[train_idx]
                X_valid = X_data.iloc[valid_idx]
            else:
                X_train = X_data[train_idx]
                X_valid = X_data[valid_idx]

            y_train = y_data[train_idx]
            y_valid = y_data[valid_idx]

            pipeline = build_xgboost_pipeline(
                preprocessor=clone(preprocessor),
                random_state=random_state,
                sampling_method=sampling_method,
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                min_child_weight=params["min_child_weight"],
                gamma=params["gamma"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                scale_pos_weight=params["scale_pos_weight"],
                n_jobs=n_jobs,
            )

            pipeline.fit(X_train, y_train)
            y_proba = pipeline.predict_proba(X_valid)[:, 1]

            if optimize_metric == "average_precision":
                score = average_precision_score(y_valid, y_proba)
            else:
                best_threshold_info = find_best_threshold_from_proba(
                    y_true=y_valid,
                    y_proba=y_proba,
                    metric="f1",
                )
                best_threshold = best_threshold_info["best_threshold"]
                y_pred = (y_proba >= best_threshold).astype(int)
                score = f1_score(y_valid, y_pred, zero_division=0)

            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params.copy()
    sampling_method = best_params.pop("sampling_method")

    scale_pos_weight = 1.0
    if sampling_method is None:
        scale_pos_weight = best_params.pop("scale_pos_weight")

    best_pipeline = build_xgboost_pipeline(
        preprocessor=clone(preprocessor),
        random_state=random_state,
        sampling_method=sampling_method,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        min_child_weight=best_params["min_child_weight"],
        gamma=best_params["gamma"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
    )

    best_pipeline.fit(X_data, y_data)

    return best_pipeline, study