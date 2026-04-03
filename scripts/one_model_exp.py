from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.xgboost_model import build_xgboost_pipeline
from src.data.split_data import make_train_test_split
from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.features.features_selection import TARGET, DROP_COLUMNS, get_feature_set
from src.features.feature_engineering import make_feature_engineering
from src.modeling.model_specs import get_baseline_model_specs, get_tuned_model_specs
from src.modeling.train import train_model
from src.modeling.compare import (
    compare_models,
    compare_models_with_pr_optimal_threshold,
    compare_models_with_target_recall,
    find_best_threshold_from_pr_curve,
    evaluate_binary_classifier
)

TO_TEST = [
    "fe_compact",
    "fe_compact_minus_fn",
    "fe_compact_minus_flags",
    "fe_compact_minus_buckets",
    "fe_compact_plus_cat_driven",
    "fe_compact_plus_risk",
]

def run_ablation_study(
    df,
    feature_set_names,
    model_builder,
    seed=51,
):
    results = []

    for feature_set_name in feature_set_names:
        feature_config = get_feature_set(feature_set_name)
        num_features = feature_config["num"]
        cat_features = feature_config["cat"]
        feature_columns = num_features + cat_features

        X_train, X_test, y_train, y_test = make_train_test_split(
            df=df,
            feature_columns=feature_columns,
            target_column=TARGET,
            test_size=0.2,
            random_state=seed,
            stratify=True,
        )

        num_features = filter_existing_features(num_features, X_train.columns.tolist())
        cat_features = filter_existing_features(cat_features, X_train.columns.tolist())

        preprocessor = build_preprocessor(
            num_features=num_features,
            cat_features=cat_features,
        )

        model = model_builder(preprocessor, seed)

        model = train_model(model, X_train, y_train)

        best_info = find_best_threshold_from_pr_curve(
            model=model,
            X_test=X_test,
            y_test=y_test,
            metric="f1",
        )

        metrics = evaluate_binary_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=best_info["best_threshold"],
        )

        metrics["feature_set"] = feature_set_name
        metrics["best_threshold"] = best_info["best_threshold"]
        metrics["n_features"] = len(num_features) + len(cat_features)

        results.append(metrics)

    return pd.DataFrame(results).sort_values("f1_1", ascending=False)





def xgb_builder(preprocessor, seed):
    return build_xgboost_pipeline(
        preprocessor=preprocessor,
        random_state=seed,
        use_smote=True,
    )

ablation_results = run_ablation_study(
    df=df,
    feature_set_names=[
        "fe_compact",
        "fe_compact_plus_fn_focus",
        "fe_compact_minus_salary",
        "fe_compact_minus_eval",
    ],
    model_builder=xgb_builder,
    seed=51,
)

print(ablation_results.round(3))