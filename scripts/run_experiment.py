from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.data.split_data import make_train_test_split
from src.features.feature_engineering import make_feature_engineering
from src.features.features_selection import DROP_COLUMNS, TARGET, get_feature_set
from src.modeling.compare import (
    compare_models,
    compare_models_with_cv_pr_optimal_threshold,
    compare_models_with_cv_target_recall,
    cross_validate_model_specs,
    get_oof_predicted_proba_by_model_specs,
)
from src.modeling.model_specs import (
    get_baseline_model_specs,
    get_new_models_model_specs,
    get_optuna_model_specs,
    get_tuned_model_specs,
)
from src.modeling.train import train_model

TO_TEST = [
    "raw_baseline",
    "raw_baseline_reduc",
    "fe_core",
    "fe_compact",
    "fe_fn_focus",
    "fe_full_robust",
]

# Choix du bloc de specs ? lancer
# Valeurs possibles :
# - "baseline"
# - "tuned"
# - "optuna"
# - "baseline+tuned"
# - "baseline+optuna"
# - "new_models"
SPEC_MODE = "baseline+tuned"


def prepare_dataset(
    data_path: Path,
    feature_set_name: str,
    test_size: float = 0.2,
    seed: int = 42,
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
        X_reference=X_train,
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


def build_model_specs(
    spec_mode: str,
    preprocessor,
    cat_features,
    X_train,
    y_train,
    seed: int,
    scoring_metric: str,
) -> dict:
    valid_modes = {
        "baseline",
        "tuned",
        "optuna",
        "new_models",
        "baseline+tuned",
        "baseline+optuna",
    }
    if spec_mode not in valid_modes:
        raise ValueError(
            f"spec_mode invalide : {spec_mode}. Valeurs possibles : {sorted(valid_modes)}"
        )

    specs = {}

    if spec_mode in {"baseline", "baseline+tuned", "baseline+optuna"}:
        baseline_specs = get_baseline_model_specs(
            preprocessor=preprocessor,
            cat_features=cat_features,
            seed=seed,
        )
        specs.update(baseline_specs)

    if spec_mode == "new_models":
        new_models_specs = get_new_models_model_specs(
            preprocessor=preprocessor,
            seed=seed,
        )
        specs.update(new_models_specs)

    if spec_mode in {"tuned", "baseline+tuned"}:
        tuned_specs = get_tuned_model_specs(
            preprocessor=preprocessor,
            cat_features=cat_features,
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            scoring=scoring_metric,
        )
        specs.update(tuned_specs)

    if spec_mode in {"optuna", "baseline+optuna"}:
        optuna_specs = get_optuna_model_specs(
            preprocessor=preprocessor,
            cat_features=cat_features,
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            scoring=scoring_metric,
        )
        specs.update(optuna_specs)

    return specs


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


def _safe_add_metadata(results_df: pd.DataFrame, model_specs: dict) -> pd.DataFrame:
    df = results_df.copy()

    family_map = {
        model_name: spec.get("family")
        for model_name, spec in model_specs.items()
    }
    sampling_map = {
        model_name: spec.get("sampling_method")
        for model_name, spec in model_specs.items()
    }

    df["family"] = df["model"].map(family_map)
    df["sampling_method"] = df["model"].map(sampling_map)

    return df


def build_final_results_dataframe(
    results_05: pd.DataFrame,
    results_pr: pd.DataFrame,
    results_recall: pd.DataFrame,
    feature_set_name: str,
    target_recall: float,
    spec_mode: str,
    model_specs: dict,
) -> pd.DataFrame:
    base_columns = [
        "model",
        "threshold",
        "precision_1",
        "recall_1",
        "f1_1",
        "f2_1",
        "prc_auc",
        "train_precision_1",
        "train_recall_1",
        "train_f1_1",
        "train_f2_1",
        "train_prc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    df1 = results_05[[column for column in base_columns if column in results_05.columns]].copy()
    df3 = results_pr[
        [
            column
            for column in [
                "model",
                "best_threshold",
                "precision_1",
                "recall_1",
                "f1_1",
                "f2_1",
                "prc_auc",
                "train_precision_1",
                "train_recall_1",
                "train_f1_1",
                "train_f2_1",
                "train_prc_auc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
            if column in results_pr.columns
        ]
    ].copy()
    df4 = results_recall[
        [
            column
            for column in [
                "model",
                "best_threshold",
                "precision_1",
                "recall_1",
                "f1_1",
                "f2_1",
                "prc_auc",
                "train_precision_1",
                "train_recall_1",
                "train_f1_1",
                "train_f2_1",
                "train_prc_auc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
            if column in results_recall.columns
        ]
    ].copy()

    df3 = df3.rename(columns={"best_threshold": "threshold"})
    df4 = df4.rename(columns={"best_threshold": "threshold"})

    df1["strategie_seuil"] = "seuil_0_5"
    df3["strategie_seuil"] = "seuil_cv_opt_prc"
    df4["strategie_seuil"] = f"cv_recall_cible_{target_recall}"

    df1["feature_set"] = feature_set_name
    df3["feature_set"] = feature_set_name
    df4["feature_set"] = feature_set_name

    df1["spec_mode"] = spec_mode
    df3["spec_mode"] = spec_mode
    df4["spec_mode"] = spec_mode

    df1 = _safe_add_metadata(df1, model_specs)
    df3 = _safe_add_metadata(df3, model_specs)
    df4 = _safe_add_metadata(df4, model_specs)

    df_all_results = pd.concat([df1, df3, df4], ignore_index=True)

    ordered_cols = [
        "model",
        "family",
        "sampling_method",
        "spec_mode",
        "strategie_seuil",
        "feature_set",
        "threshold",
        "precision_1",
        "recall_1",
        "f1_1",
        "f2_1",
        "prc_auc",
        "train_precision_1",
        "train_recall_1",
        "train_f1_1",
        "train_f2_1",
        "train_prc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    existing_cols = [col for col in ordered_cols if col in df_all_results.columns]
    return df_all_results[existing_cols].copy()


def export_global_results(
    all_results: pd.DataFrame,
    summary_best: pd.DataFrame,
    output_dir: Path,
    spec_mode: str,
    baseline_cv_summary: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_spec_mode = spec_mode.replace("+", "_plus_")
    file_path = output_dir / f"{pd.Timestamp.now().strftime('%y%m%d')}_benchmark_feature_sets__{safe_spec_mode}.xlsx"

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        all_results.to_excel(writer, sheet_name="all_results", index=False)
        summary_best.to_excel(writer, sheet_name="best_by_feature_set", index=False)
        if baseline_cv_summary is not None and not baseline_cv_summary.empty:
            baseline_cv_summary.to_excel(
                writer,
                sheet_name="baseline_cv_summary",
                index=False,
            )

    print(f"Global benchmark exported: {file_path}")


def run_one_feature_set(
    data_path: Path,
    feature_set_name: str,
    seed: int,
    test_size: float,
    scoring_metric: str,
    target_recall: float,
    spec_mode: str,
    cv_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 80)
    print(f"RUN FEATURE SET: {feature_set_name}")
    print(f"SPEC MODE: {spec_mode}")
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
    model_specs = build_model_specs(
        spec_mode=spec_mode,
        preprocessor=preprocessor,
        cat_features=cat_features,
        X_train=X_train,
        y_train=y_train,
        seed=seed,
        scoring_metric=scoring_metric,
    )

    print(f" Number of models in this run: {len(model_specs)}")

    baseline_specs = {
        model_name: spec
        for model_name, spec in model_specs.items()
        if not spec.get("already_trained", False)
    }
    baseline_cv_summary = pd.DataFrame()

    if baseline_specs:
        print(f" Running {cv_folds}-fold cross-validation on baseline models...")
        _, baseline_cv_summary = cross_validate_model_specs(
            model_specs=baseline_specs,
            X=X_train,
            y=y_train,
            cv=cv_folds,
            threshold=0.5,
            random_state=seed,
        )
        if not baseline_cv_summary.empty:
            baseline_cv_summary = _safe_add_metadata(baseline_cv_summary, baseline_specs)
            baseline_cv_summary["feature_set"] = feature_set_name
            baseline_cv_summary["spec_mode"] = spec_mode

    print(f" Estimating thresholds with {cv_folds}-fold OOF probabilities...")
    oof_proba_by_model = get_oof_predicted_proba_by_model_specs(
        model_specs=model_specs,
        X=X_train,
        y=y_train,
        cv=cv_folds,
        random_state=seed,
    )

    print(" Training models on full train set...")
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
        X_train=X_train,
        y_train=y_train,
    )

    print(" Comparing models with CV PR-optimized threshold...")
    results_pr = compare_models_with_cv_pr_optimal_threshold(
        trained_models=trained_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metric="f2",
        sort_by="f2_1",
        oof_proba_by_model=oof_proba_by_model,
        cv=cv_folds,
        random_state=seed,
    )

    print(f" Comparing models with CV target recall = {target_recall}...")
    results_recall = compare_models_with_cv_target_recall(
        trained_models=trained_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        target_recall=target_recall,
        sort_by="precision_1",
        oof_proba_by_model=oof_proba_by_model,
        cv=cv_folds,
        random_state=seed,
    )

    final_df = build_final_results_dataframe(
        results_05=results_05,
        results_pr=results_pr,
        results_recall=results_recall,
        feature_set_name=feature_set_name,
        target_recall=target_recall,
        spec_mode=spec_mode,
        model_specs=model_specs,
    )

    return final_df, baseline_cv_summary


def build_best_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        all_results.sort_values(
            ["feature_set", "prc_auc", "f1_1"],
            ascending=[True, False, False],
        )
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
    cv_folds = 5

    data_path = PROJECT_ROOT / "data" / "interim" / "data_eda.csv"
    output_dir = PROJECT_ROOT / "data" / "processed"

    all_feature_set_results = []
    all_baseline_cv_summaries = []

    print("\n=== RUN CONFIG ===")
    print(f"SPEC_MODE      : {SPEC_MODE}")
    print(f"SCORING_METRIC : {scoring_metric}")
    print(f"TARGET_RECALL  : {target_recall}")
    print(f"CV_FOLDS       : {cv_folds}")
    print(f"FEATURE SETS   : {TO_TEST}")
    print()

    for feature_set_name in TO_TEST:
        result_df, baseline_cv_summary = run_one_feature_set(
            data_path=data_path,
            feature_set_name=feature_set_name,
            seed=seed,
            test_size=test_size,
            scoring_metric=scoring_metric,
            target_recall=target_recall,
            spec_mode=SPEC_MODE,
            cv_folds=cv_folds,
        )
        all_feature_set_results.append(result_df)
        if not baseline_cv_summary.empty:
            all_baseline_cv_summaries.append(baseline_cv_summary)

    all_results = pd.concat(all_feature_set_results, ignore_index=True)
    summary_best = build_best_summary(all_results)

    baseline_cv_summary = pd.DataFrame()
    if all_baseline_cv_summaries:
        baseline_cv_summary = pd.concat(all_baseline_cv_summaries, ignore_index=True)

    print("\n=== TOP RESULTS BY FEATURE SET ===")
    print(summary_best)

    if not baseline_cv_summary.empty:
        print("\n=== TOP BASELINE CV RESULTS ===")
        top_cv_source = baseline_cv_summary[baseline_cv_summary["model"] != "dummy"].copy()
        if top_cv_source.empty:
            top_cv_source = baseline_cv_summary.copy()
        top_cv = top_cv_source.sort_values(
            "valid_prc_auc_mean",
            ascending=False,
        ).head(10)
        print(top_cv)

    export_global_results(
        all_results=all_results,
        summary_best=summary_best,
        output_dir=output_dir,
        spec_mode=SPEC_MODE,
        baseline_cv_summary=baseline_cv_summary,
    )


if __name__ == "__main__":
    main()
