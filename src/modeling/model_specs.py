from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.dummy_classifier_model import get_dummy_model

from models.logistic_regression_model import (
    build_logistic_regression_pipeline,
    build_lasso_logistic_regression_pipeline,
    build_elastic_net_logistic_regression_pipeline,
    get_logistic_regression_param_grid,
    get_logistic_regression_param_distributions,
    get_lasso_logistic_regression_param_grid,
    get_elastic_net_logistic_regression_param_distributions,
)

from models.random_forest_model import (
    build_random_forest_pipeline,
    get_random_forest_param_distributions,
)

from models.xgboost_model import (
    build_xgboost_pipeline,
    get_xgboost_param_distributions,
    optimize_xgboost_with_optuna,
)

from models.lightgbm_model import (
    build_lightgbm_pipeline,
    get_lightgbm_param_distributions,
)

from models.knn_model import (
    build_knn_pipeline,
    get_knn_param_distributions,
)

from models.svc_model import (
    build_svc_pipeline,
    get_svc_param_distributions,
)

from models.catboost_model import (
    build_catboost_model,
    get_catboost_param_distributions,
    optimize_catboost_with_optuna,
)

from src.modeling.train import (
    train_model_with_gridsearch,
    train_model_with_randomized_search,
)


def _infer_sampling_method_from_pipeline(model) -> str | None:
    """
    Infère la méthode de sampling à partir des étapes du pipeline.
    """
    if not hasattr(model, "named_steps"):
        return None

    step_names = set(model.named_steps.keys())

    if "under" in step_names:
        return "smote_under"

    if "smote" in step_names:
        sampler = model.named_steps["smote"].__class__.__name__.lower()
        if "borderline" in sampler:
            return "borderline"
        return "smote"

    return None


# =========================================
# BASELINE MODELS
# =========================================
def get_baseline_model_specs(preprocessor, cat_features, seed=42):
    return {
        "dummy": {
            "model": get_dummy_model(),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "baseline",
            "sampling_method": None,
        },

        # Logistic Regression
        "log_reg_baseline": {
            "model": build_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method=None
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear",
            "sampling_method": None,
        },
        "log_reg_smote": {
            "model": build_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="smote"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear",
            "sampling_method": "smote",
        },
        "log_reg_borderline": {
            "model": build_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear",
            "sampling_method": "borderline",
        },
        "log_reg_smote_under": {
            "model": build_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="smote_under"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear",
            "sampling_method": "smote_under",
        },

        # Lasso Logistic Regression
        "lasso_log_reg_baseline": {
            "model": build_lasso_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method=None
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_l1",
            "sampling_method": None,
        },
        "lasso_log_reg_smote": {
            "model": build_lasso_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="smote"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_l1",
            "sampling_method": "smote",
        },
        "lasso_log_reg_borderline": {
            "model": build_lasso_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_l1",
            "sampling_method": "borderline",
        },
        "lasso_log_reg_smote_under": {
            "model": build_lasso_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="smote_under"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_l1",
            "sampling_method": "smote_under",
        },

        # Elastic Net Logistic Regression
        "elastic_net_log_reg_baseline": {
            "model": build_elastic_net_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method=None
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_elastic_net",
            "sampling_method": None,
        },
        "elastic_net_log_reg_smote": {
            "model": build_elastic_net_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="smote"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_elastic_net",
            "sampling_method": "smote",
        },
        "elastic_net_log_reg_borderline": {
            "model": build_elastic_net_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_elastic_net",
            "sampling_method": "borderline",
        },
        "elastic_net_log_reg_smote_under": {
            "model": build_elastic_net_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="smote_under"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_elastic_net",
            "sampling_method": "smote_under",
        },

        # Random Forest
        "random_forest_baseline": {
            "model": build_random_forest_pipeline(
                preprocessor, random_state=seed, sampling_method=None
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "tree_ensemble",
            "sampling_method": None,
        },
        "random_forest_smote": {
            "model": build_random_forest_pipeline(
                preprocessor, random_state=seed, sampling_method="smote"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "tree_ensemble",
            "sampling_method": "smote",
        },
        "random_forest_borderline": {
            "model": build_random_forest_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "tree_ensemble",
            "sampling_method": "borderline",
        },
        "random_forest_smote_under": {
            "model": build_random_forest_pipeline(
                preprocessor, random_state=seed, sampling_method="smote_under"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "tree_ensemble",
            "sampling_method": "smote_under",
        },

        # XGBoost
        "xgboost_baseline": {
            "model": build_xgboost_pipeline(
                preprocessor, random_state=seed, sampling_method=None
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting",
            "sampling_method": None,
        },
        "xgboost_smote": {
            "model": build_xgboost_pipeline(
                preprocessor, random_state=seed, sampling_method="smote"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting",
            "sampling_method": "smote",
        },
        "xgboost_borderline": {
            "model": build_xgboost_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting",
            "sampling_method": "borderline",
        },
        "xgboost_smote_under": {
            "model": build_xgboost_pipeline(
                preprocessor, random_state=seed, sampling_method="smote_under"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting",
            "sampling_method": "smote_under",
        },

        # LightGBM
        "lightgbm_baseline": {
            "model": build_lightgbm_pipeline(
                preprocessor, random_state=seed, sampling_method=None
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting_lgbm",
            "sampling_method": None,
        },
        "lightgbm_smote": {
            "model": build_lightgbm_pipeline(
                preprocessor, random_state=seed, sampling_method="smote"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting_lgbm",
            "sampling_method": "smote",
        },
        "lightgbm_borderline": {
            "model": build_lightgbm_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting_lgbm",
            "sampling_method": "borderline",
        },
        "lightgbm_smote_under": {
            "model": build_lightgbm_pipeline(
                preprocessor, random_state=seed, sampling_method="smote_under"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting_lgbm",
            "sampling_method": "smote_under",
        },

        # KNN
        "knn_baseline": {
            "model": build_knn_pipeline(
                preprocessor, random_state=seed, sampling_method=None
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "instance_based",
            "sampling_method": None,
        },
        "knn_smote": {
            "model": build_knn_pipeline(
                preprocessor, random_state=seed, sampling_method="smote"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "instance_based",
            "sampling_method": "smote",
        },
        "knn_borderline": {
            "model": build_knn_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "instance_based",
            "sampling_method": "borderline",
        },
        "knn_smote_under": {
            "model": build_knn_pipeline(
                preprocessor, random_state=seed, sampling_method="smote_under"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "instance_based",
            "sampling_method": "smote_under",
        },

        # SVC
        "svc_baseline": {
            "model": build_svc_pipeline(
                preprocessor, random_state=seed, sampling_method=None
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "kernel_svm",
            "sampling_method": None,
        },
        "svc_smote": {
            "model": build_svc_pipeline(
                preprocessor, random_state=seed, sampling_method="smote"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "kernel_svm",
            "sampling_method": "smote",
        },
        "svc_borderline": {
            "model": build_svc_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "kernel_svm",
            "sampling_method": "borderline",
        },
        "svc_smote_under": {
            "model": build_svc_pipeline(
                preprocessor, random_state=seed, sampling_method="smote_under"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "kernel_svm",
            "sampling_method": "smote_under",
        },

        # CatBoost
        "catboost": {
            "model": build_catboost_model(
                random_state=seed,
                auto_class_weights="Balanced",
            ),
            "fit_kwargs": {"cat_features": cat_features},
            "already_trained": False,
            "family": "boosting_cat",
            "sampling_method": None,
        },
    }


# =========================================
# TUNED MODELS - Grid / Random Search
# =========================================
def get_tuned_model_specs(
    preprocessor,
    cat_features,
    X_train,
    y_train,
    seed=42,
    scoring="average_precision",
):
    # ===== Logistic =====
    grid_log = train_model_with_gridsearch(
        model=build_logistic_regression_pipeline(
            preprocessor, random_state=seed, sampling_method="borderline"
        ),
        X_train=X_train,
        y_train=y_train,
        param_grid=get_logistic_regression_param_grid(),
        scoring=scoring,
        n_jobs=-1,
    )

    random_log = train_model_with_randomized_search(
        model=build_logistic_regression_pipeline(
            preprocessor, random_state=seed, sampling_method="borderline"
        ),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_logistic_regression_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    # ===== Lasso =====
    grid_lasso = train_model_with_gridsearch(
        model=build_lasso_logistic_regression_pipeline(
            preprocessor, random_state=seed, sampling_method="borderline"
        ),
        X_train=X_train,
        y_train=y_train,
        param_grid=get_lasso_logistic_regression_param_grid(),
        scoring=scoring,
        n_jobs=-1,
    )

    # ===== Elastic Net =====
    random_elastic_net = train_model_with_randomized_search(
        model=build_elastic_net_logistic_regression_pipeline(
            preprocessor, random_state=seed, sampling_method="borderline"
        ),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_elastic_net_logistic_regression_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    # ===== Random Forest =====
    random_rf = train_model_with_randomized_search(
        model=build_random_forest_pipeline(
            preprocessor, random_state=seed, sampling_method="borderline"
        ),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_random_forest_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    # ===== XGBoost =====
    random_xgb = train_model_with_randomized_search(
        model=build_xgboost_pipeline(
            preprocessor, random_state=seed, sampling_method="borderline"
        ),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_xgboost_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    # ===== LightGBM =====
    random_lgbm = train_model_with_randomized_search(
        model=build_lightgbm_pipeline(
            preprocessor, random_state=seed, sampling_method="borderline"
        ),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_lightgbm_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    # ===== KNN =====
    random_knn = train_model_with_randomized_search(
        model=build_knn_pipeline(
            preprocessor, random_state=seed, sampling_method="borderline"
        ),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_knn_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    # ===== SVC =====
    random_svc = train_model_with_randomized_search(
        model=build_svc_pipeline(
            preprocessor, random_state=seed, sampling_method="borderline"
        ),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_svc_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    # ===== CatBoost =====
    random_cat = train_model_with_randomized_search(
        model=build_catboost_model(random_state=seed),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_catboost_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
        fit_kwargs={"cat_features": cat_features},
    )

    return {
        "best_log_reg_grid": {
            "model": grid_log.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "linear",
            "sampling_method": _infer_sampling_method_from_pipeline(grid_log.best_estimator_),
        },
        "best_log_reg_random": {
            "model": random_log.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "linear",
            "sampling_method": _infer_sampling_method_from_pipeline(random_log.best_estimator_),
        },
        "best_lasso_log_reg": {
            "model": grid_lasso.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "linear_l1",
            "sampling_method": _infer_sampling_method_from_pipeline(grid_lasso.best_estimator_),
        },
        "best_elastic_net_log_reg": {
            "model": random_elastic_net.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "linear_elastic_net",
            "sampling_method": _infer_sampling_method_from_pipeline(random_elastic_net.best_estimator_),
        },
        "best_random_forest": {
            "model": random_rf.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "tree_ensemble",
            "sampling_method": _infer_sampling_method_from_pipeline(random_rf.best_estimator_),
        },
        "best_xgboost": {
            "model": random_xgb.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "boosting",
            "sampling_method": _infer_sampling_method_from_pipeline(random_xgb.best_estimator_),
        },
        "best_lightgbm": {
            "model": random_lgbm.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "boosting_lgbm",
            "sampling_method": _infer_sampling_method_from_pipeline(random_lgbm.best_estimator_),
        },
        "best_knn": {
            "model": random_knn.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "instance_based",
            "sampling_method": _infer_sampling_method_from_pipeline(random_knn.best_estimator_),
        },
        "best_svc": {
            "model": random_svc.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "kernel_svm",
            "sampling_method": _infer_sampling_method_from_pipeline(random_svc.best_estimator_),
        },
        "best_catboost": {
            "model": random_cat.best_estimator_,
            "fit_kwargs": {"cat_features": cat_features},
            "already_trained": True,
            "family": "boosting_cat",
            "sampling_method": None,
        },
    }


# =========================================
# BEST MODELS - Optuna
# =========================================
def get_optuna_model_specs(
    preprocessor,
    cat_features,
    X_train,
    y_train,
    seed=42,
    scoring="average_precision",
    n_trials=50,
    cv=5,
):
    # ===== XGBoost =====
    best_xgb_model, xgb_study = optimize_xgboost_with_optuna(
        X=X_train,
        y=y_train,
        preprocessor=preprocessor,
        n_trials=n_trials,
        cv=cv,
        random_state=seed,
        optimize_metric=scoring,
        n_jobs=-1,
    )

    # ===== CatBoost =====
    best_cat_model, cat_study = optimize_catboost_with_optuna(
        X=X_train,
        y=y_train,
        cat_features=cat_features,
        n_trials=n_trials,
        cv=cv,
        random_state=seed,
        optimize_metric=scoring,
    )

    return {
        "best_xgboost_optuna": {
            "model": best_xgb_model,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "boosting",
            "sampling_method": _infer_sampling_method_from_pipeline(best_xgb_model),
            "study": xgb_study,
        },
        "best_catboost_optuna": {
            "model": best_cat_model,
            "fit_kwargs": {"cat_features": cat_features},
            "already_trained": True,
            "family": "boosting_cat",
            "sampling_method": None,
            "study": cat_study,
        },
    }

# =========================================
# NEW MODELS SPECS
# =========================================
def get_new_models_model_specs(preprocessor, seed=42):
    """
    Retourne un petit pack de specs pour tester uniquement les nouveaux mod?les.
    Ce pack reste l?ger pour pouvoir benchmarker rapidement Elastic Net,
    LightGBM, KNN et SVC sans relancer tout le zoo de mod?les.
    """
    return {
        "elastic_net_candidate": {
            "model": build_elastic_net_logistic_regression_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_elastic_net",
            "sampling_method": "borderline",
        },
        "lightgbm_candidate": {
            "model": build_lightgbm_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting_lgbm",
            "sampling_method": "borderline",
        },
        "knn_candidate": {
            "model": build_knn_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "instance_based",
            "sampling_method": "borderline",
        },
        "svc_candidate": {
            "model": build_svc_pipeline(
                preprocessor, random_state=seed, sampling_method="borderline"
            ),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "kernel_svm",
            "sampling_method": "borderline",
        },
    }
