from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.split_data import make_train_test_split
from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.features.features_selection import TARGET, DROP_COLUMNS, get_feature_set
from src.features.feature_engineering import make_feature_engineering
from src.models.model_specs import get_baseline_model_specs, get_tuned_model_specs
from src.models.train import train_model
from src.models.compare import (
    compare_models,
    compare_models_with_pr_optimal_threshold,
    compare_models_with_target_recall,
)


TO_TEST = [
    "raw_baseline",
    "fe_core",
    "fe_compact",
    "fe_fn_focus",
    "fe_full_robust",
]


def prepare_dataset(
    data_path: Path,
    feature_set_name: str,
    test_size: float = 0.2,
    seed: int = 51,
):
    df = pd.read_csv(data_path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    df = make_feature_engineering(df)

    feature_config = get_feature_set(feature_set_name)
    num_features = feature_config["num"]
    cat_features = feature_config["cat"]

    feature_columns = num_features + cat_features

    X_train, X_test, y_train, y_test = make_train_test_split(
        df=df,
        feature_columns=feature_columns,
        target_column=TARGET,
        test_size=test_size,
        random_state=seed,
        stratify=True,
    )

    num_features = filter_existing_features(num_features, X_train.columns.tolist())
    cat_features = filter_existing_features(cat_features, X_train.columns.tolist())

    preprocessor = build_preprocessor(
        num_features=num_features,
        cat_features=cat_features,
    )

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_features": num_features,
        "cat_features": cat_features,
        "preprocessor": preprocessor,
    }


def train_all_models(model_specs: dict, X_train, y_train) -> dict:
    trained_models = {}

    for model_name, spec in model_specs.items():
        print(f"  Training model: {model_name}")

        if spec["already_trained"]:
            trained_models[model_name] = spec["model"]
        else:
            trained_models[model_name] = train_model(
                model=spec["model"],
                X_train=X_train,
                y_train=y_train,
                **spec["fit_kwargs"],
            )

    return trained_models


def build_final_results_dataframe(
    results_05: pd.DataFrame,
    results_pr: pd.DataFrame,
    results_recall: pd.DataFrame,
    feature_set_name: str,
    target_recall: float,
) -> pd.DataFrame:
    df1 = results_05[
        ["model", "threshold", "precision_1", "recall_1", "f1_1", "prc_auc", "tn", "fp", "fn", "tp"]
    ].copy()

    df3 = results_pr[
        ["model", "best_threshold", "precision_1", "recall_1", "f1_1", "prc_auc", "tn", "fp", "fn", "tp"]
    ].copy()

    df4 = results_recall[
        ["model", "best_threshold", "precision_1", "recall_1", "f1_1", "prc_auc", "tn", "fp", "fn", "tp"]
    ].copy()

    df3 = df3.rename(columns={"best_threshold": "threshold"})
    df4 = df4.rename(columns={"best_threshold": "threshold"})

    df1["strategie_seuil"] = "seuil_0_5"
    df3["strategie_seuil"] = "seuil_opt_prc"
    df4["strategie_seuil"] = f"recall_cible_{target_recall}"

    df1["feature_set"] = feature_set_name
    df3["feature_set"] = feature_set_name
    df4["feature_set"] = feature_set_name

    df_all_results = pd.concat([df1, df3, df4], ignore_index=True)

    ordered_cols = [
        "model",
        "strategie_seuil",
        "feature_set",
        "threshold",
        "precision_1",
        "recall_1",
        "f1_1",
        "prc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    return df_all_results[ordered_cols].copy()


def export_global_results(
    all_results: pd.DataFrame,
    summary_best: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / "benchmark_feature_sets.xlsx"

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        all_results.to_excel(writer, sheet_name="all_results", index=False)
        summary_best.to_excel(writer, sheet_name="best_by_feature_set", index=False)

    print(f"Global benchmark exported: {file_path}")


def run_one_feature_set(
    data_path: Path,
    feature_set_name: str,
    seed: int,
    test_size: float,
    scoring_metric: str,
    target_recall: float,
) -> pd.DataFrame:
    print("=" * 80)
    print(f"RUN FEATURE SET: {feature_set_name}")
    print("=" * 80)

    prepared = prepare_dataset(
        data_path=data_path,
        feature_set_name=feature_set_name,
        test_size=test_size,
        seed=seed,
    )

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    cat_features = prepared["cat_features"]
    preprocessor = prepared["preprocessor"]

    print(" Building model specs...")
    baseline_specs = get_baseline_model_specs(
        preprocessor=preprocessor,
        cat_features=cat_features,
        seed=seed,
    )

    tuned_specs = get_tuned_model_specs(
        preprocessor=preprocessor,
        cat_features=cat_features,
        X_train=X_train,
        y_train=y_train,
        seed=seed,
        scoring=scoring_metric,
    )

    model_specs = {**baseline_specs, **tuned_specs}

    print(" Training models...")
    trained_models = train_all_models(
        model_specs=model_specs,
        X_train=X_train,
        y_train=y_train,
    )

    print(" Comparing models at threshold 0.5...")
    results_05 = compare_models(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
        threshold=0.5,
        sort_by="prc_auc",
    )

    print(" Comparing models with PR-optimized threshold...")
    results_pr = compare_models_with_pr_optimal_threshold(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
        metric="f1",
        sort_by="f1_1",
    )

    print(f" Comparing models with target recall = {target_recall}...")
    results_recall = compare_models_with_target_recall(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
        target_recall=target_recall,
        sort_by="precision_1",
    )

    final_df = build_final_results_dataframe(
        results_05=results_05,
        results_pr=results_pr,
        results_recall=results_recall,
        feature_set_name=feature_set_name,
        target_recall=target_recall,
    )

    return final_df


def build_best_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    Garde la meilleure ligne par feature_set selon f1_1,
    puis prc_auc en tie-break.
    """
    summary = (
        all_results
        .sort_values(["feature_set", "f1_1", "prc_auc"], ascending=[True, False, False])
        .groupby("feature_set", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return summary


def main():
    seed = 51
    test_size = 0.2
    scoring_metric = "average_precision"
    target_recall = 0.9

    data_path = PROJECT_ROOT / "data" / "interim" / "data_eda.csv"
    output_dir = PROJECT_ROOT / "data" / "processed"

    all_feature_set_results = []

    for feature_set_name in TO_TEST:
        result_df = run_one_feature_set(
            data_path=data_path,
            feature_set_name=feature_set_name,
            seed=seed,
            test_size=test_size,
            scoring_metric=scoring_metric,
            target_recall=target_recall,
        )
        all_feature_set_results.append(result_df)

    all_results = pd.concat(all_feature_set_results, ignore_index=True)

    summary_best = build_best_summary(all_results)

    print("\n=== TOP RESULTS BY FEATURE SET ===")
    print(summary_best.round(3))

    export_global_results(
        all_results=all_results,
        summary_best=summary_best,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()