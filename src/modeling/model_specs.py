from __future__ import annotations

# =========================================
# Imports modèles
# =========================================
from models.dummy_classifier_model import get_dummy_model

from models.logistic_regression_model import (
    build_logistic_regression_pipeline,
    build_lasso_logistic_regression_pipeline,
    get_logistic_regression_param_grid,
    get_logistic_regression_param_distributions,
    get_lasso_logistic_regression_param_grid,
)

from models.random_forest_model import (
    build_random_forest_pipeline,
    get_random_forest_param_distributions,
)

from models.xgboost_model import (
    build_xgboost_pipeline,
    get_xgboost_param_distributions,
)

from models.catboost_model import (
    build_catboost_model,
    get_catboost_param_distributions,
)

from src.modeling.train import (
    train_model_with_gridsearch,
    train_model_with_randomized_search,
)

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
            "use_smote": False,
        },

        "log_reg": {
            "model": build_logistic_regression_pipeline(preprocessor, random_state=seed, use_smote=True),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear",
            "use_smote": True,
        },

        "lasso_log_reg": {
            "model": build_lasso_logistic_regression_pipeline(preprocessor, random_state=seed, use_smote=True),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "linear_l1",
            "use_smote": True,
        },

        "random_forest": {
            "model": build_random_forest_pipeline(preprocessor, random_state=seed, use_smote=True),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "tree_ensemble",
            "use_smote": True,
        },

        "xgboost": {
            "model": build_xgboost_pipeline(preprocessor, random_state=seed, use_smote=True),
            "fit_kwargs": {},
            "already_trained": False,
            "family": "boosting",
            "use_smote": True,
        },

        "catboost": {
            "model": build_catboost_model(random_state=seed),
            "fit_kwargs": {"cat_features": cat_features},
            "already_trained": False,
            "family": "boosting_cat",
            "use_smote": False,  # ⚠️ jamais avec CatBoost
        },
    }


# =========================================
# TUNED MODELS
# =========================================

def get_tuned_model_specs(preprocessor, cat_features, X_train, y_train, seed=42, scoring = "average_precision"):
    scoring = scoring

    #* ===== Logistic =====
    grid_log = train_model_with_gridsearch(
        model=build_logistic_regression_pipeline(preprocessor, random_state=seed, use_smote=True),
        X_train=X_train,
        y_train=y_train,
        param_grid=get_logistic_regression_param_grid(),
        scoring=scoring,
        n_jobs=-1,
    )

    random_log = train_model_with_randomized_search(
        model=build_logistic_regression_pipeline(preprocessor, random_state=seed, use_smote=True),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_logistic_regression_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    #* ===== Lasso =====
    grid_lasso = train_model_with_gridsearch(
        model=build_lasso_logistic_regression_pipeline(preprocessor, random_state=seed, use_smote=True),
        X_train=X_train,
        y_train=y_train,
        param_grid=get_lasso_logistic_regression_param_grid(),
        scoring=scoring,
        n_jobs=-1,
    )

    #* ===== Random Forest =====
    random_rf = train_model_with_randomized_search(
        model=build_random_forest_pipeline(preprocessor, random_state=seed, use_smote=True),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_random_forest_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    #* ===== XGBoost =====
    random_xgb = train_model_with_randomized_search(
        model=build_xgboost_pipeline(preprocessor, random_state=seed, use_smote=True),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_xgboost_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    #* ===== CatBoost =====
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
            "use_smote": True,
        },

        "best_log_reg_random": {
            "model": random_log.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "linear",
            "use_smote": True,
        },

        "best_lasso_log_reg": {
            "model": grid_lasso.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "linear_l1",
            "use_smote": True,
        },

        "best_random_forest": {
            "model": random_rf.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "tree_ensemble",
            "use_smote": True,
        },

        "best_xgboost": {
            "model": random_xgb.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "boosting",
            "use_smote": True,
        },

        "best_catboost": {
            "model": random_cat.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "boosting_cat",
            "use_smote": False,
        },
    }



# =========================================
# BEST MODEL
# =========================================

def get_best_model_specs(preprocessor, cat_features, X_train, y_train, seed=42, scoring = "average_precision"):
    scoring = scoring

    #* ===== XGBoost =====
    random_xgb = train_model_with_randomized_search(
        model=build_xgboost_pipeline(preprocessor, random_state=seed, use_smote=True),
        X_train=X_train,
        y_train=y_train,
        param_distributions=get_xgboost_param_distributions(),
        n_iter=20,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )

    #* ===== CatBoost =====
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
        "best_xgboost": {
            "model": random_xgb.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "boosting",
            "use_smote": True,
        },

        "best_catboost": {
            "model": random_cat.best_estimator_,
            "fit_kwargs": {},
            "already_trained": True,
            "family": "boosting_cat",
            "use_smote": False,
        },
    }