from __future__ import annotations

from typing import Sequence

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.modeling.compare import find_best_threshold_from_proba


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
    auto_class_weights: str | None = None,
    random_strength: float = 1.0,
    bagging_temperature: float = 0.0,
    border_count: int = 254,
) -> CatBoostClassifier:
    """
    Construit un modèle CatBoostClassifier.
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
        auto_class_weights=auto_class_weights,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        border_count=border_count,
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
        "class_weights": [[1.0, 2.0], [1.0, 3.0], [1.0, 5.0]],
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
        "class_weights": [[1.0, 2.0], [1.0, 3.0], [1.0, 5.0], [1.0, 7.0]],
    }


def optimize_catboost_with_optuna(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cat_features: Sequence[int | str] | None = None,
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
    timeout: int | None = None,
    optimize_metric: str = "average_precision",
    f1_threshold: float = 0.5,
) -> tuple[CatBoostClassifier, optuna.study.Study]:
    """
    Optimise un CatBoostClassifier avec Optuna via validation croisée.
    """
    allowed_metrics = {"average_precision", "f1"}
    if optimize_metric not in allowed_metrics:
        raise ValueError(
            f"optimize_metric doit être dans {allowed_metrics}, reçu: {optimize_metric}"
        )

    X_data = X
    y_data = np.asarray(y)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1200),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 0,
            "random_seed": random_state,
        }

        class_weight_mode = trial.suggest_categorical(
            "class_weight_mode",
            ["none", "balanced", "manual_2", "manual_3", "manual_5"],
        )

        if class_weight_mode == "balanced":
            params["auto_class_weights"] = "Balanced"
        elif class_weight_mode == "manual_2":
            params["class_weights"] = [1.0, 2.0]
        elif class_weight_mode == "manual_3":
            params["class_weights"] = [1.0, 3.0]
        elif class_weight_mode == "manual_5":
            params["class_weights"] = [1.0, 5.0]

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

            model = CatBoostClassifier(**params)

            fit_kwargs = {}
            if cat_features is not None:
                fit_kwargs["cat_features"] = cat_features

            model.fit(X_train, y_train, **fit_kwargs)

            y_proba = model.predict_proba(X_valid)[:, 1]

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
    class_weight_mode = best_params.pop("class_weight_mode", "none")

    final_params = {
        "iterations": best_params["iterations"],
        "depth": best_params["depth"],
        "learning_rate": best_params["learning_rate"],
        "l2_leaf_reg": best_params["l2_leaf_reg"],
        "random_strength": best_params["random_strength"],
        "bagging_temperature": best_params["bagging_temperature"],
        "border_count": best_params["border_count"],
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "random_seed": random_state,
    }

    if class_weight_mode == "balanced":
        final_params["auto_class_weights"] = "Balanced"
    elif class_weight_mode == "manual_2":
        final_params["class_weights"] = [1.0, 2.0]
    elif class_weight_mode == "manual_3":
        final_params["class_weights"] = [1.0, 3.0]
    elif class_weight_mode == "manual_5":
        final_params["class_weights"] = [1.0, 5.0]

    best_model = CatBoostClassifier(**final_params)

    fit_kwargs = {}
    if cat_features is not None:
        fit_kwargs["cat_features"] = cat_features

    best_model.fit(X_data, y_data, **fit_kwargs)

    return best_model, study