from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.logistic_regression_model import (
    build_elastic_net_logistic_regression_pipeline,
)
from models.random_forest_model import build_random_forest_pipeline
from models.xgboost_model import build_xgboost_pipeline
from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.data.split_data import make_train_test_split
from src.features.feature_engineering import make_feature_engineering
from src.features.features_selection import DROP_COLUMNS, TARGET, get_feature_set
from src.modeling.compare import (
    compare_models,
    compare_models_with_cv_pr_optimal_threshold,
    compare_models_with_cv_target_recall,
    cross_validate_model_specs,
    get_oof_predicted_proba,
)
from src.modeling.train import (
    train_model_with_gridsearch,
    train_model_with_randomized_search,
)

SEED = 51
TEST_SIZE = 0.2
SCORING_METRIC = "average_precision"
TARGET_RECALL = 0.70
CV_FOLDS = 5
USE_XGBOOST_GPU = True
XGBOOST_SEARCH_ITER = 45
RANDOM_FOREST_SEARCH_ITER = 40

FINAL_TUNING_CANDIDATES = [
    {
        "candidate_name": "elastic_net_fe_core_smote",
        "feature_set": "fe_core",
        "model_key": "elastic_net",
        "family": "linear_elastic_net",
        "sampling_method": "smote",
        "search_type": "gridsearch",
    },
    {
        "candidate_name": "elastic_net_fe_full_robust_borderline",
        "feature_set": "fe_full_robust",
        "model_key": "elastic_net",
        "family": "linear_elastic_net",
        "sampling_method": "borderline",
        "search_type": "gridsearch",
    },
    {
        "candidate_name": "xgboost_raw_baseline_smote",
        "feature_set": "raw_baseline",
        "model_key": "xgboost",
        "family": "boosting",
        "sampling_method": "smote",
        "search_type": "randomized_search",
    },
    {
        "candidate_name": "xgboost_fe_core_smote",
        "feature_set": "fe_core",
        "model_key": "xgboost",
        "family": "boosting",
        "sampling_method": "smote",
        "search_type": "randomized_search",
    },
    {
        "candidate_name": "random_forest_raw_baseline_borderline",
        "feature_set": "raw_baseline",
        "model_key": "random_forest",
        "family": "tree_ensemble",
        "sampling_method": "borderline",
        "search_type": "randomized_search",
    },
]


def prepare_dataset(
    data_path: Path,
    feature_set_name: str,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
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
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_features": num_features,
        "cat_features": cat_features,
        "preprocessor": preprocessor,
    }


def get_final_elastic_net_param_grid() -> dict:
    return {
        "model__penalty": ["elasticnet"],
        "model__solver": ["saga"],
        "model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        "model__l1_ratio": [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [5000],
    }


def get_final_random_forest_param_distributions() -> dict:
    return {
        "model__n_estimators": [300, 500, 800, 1200],
        "model__max_depth": [None, 6, 10, 14, 20],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", 0.5, 0.7],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }


def get_final_xgboost_param_distributions() -> dict:
    return {
        "model__n_estimators": [250, 350, 500, 700, 900],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
        "model__subsample": [0.65, 0.75, 0.85, 1.0],
        "model__colsample_bytree": [0.65, 0.75, 0.85, 1.0],
        "model__min_child_weight": [1, 2, 4, 6, 8],
        "model__gamma": [0.0, 0.1, 0.3, 0.7, 1.5],
        "model__reg_alpha": [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 4.0, 8.0],
        "model__scale_pos_weight": [1.0],
    }


def _to_builtin(value):
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(val) for val in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _prepare_result_block(df: pd.DataFrame, threshold_label: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    block = df.copy()
    if "best_threshold" in block.columns:
        block = block.rename(columns={"best_threshold": "threshold"})

    block["strategie_seuil"] = threshold_label
    return block


def _add_candidate_metadata(
    df: pd.DataFrame,
    candidate: dict,
    best_cv_score: float,
    best_params: dict,
    fit_time_sec: float,
) -> pd.DataFrame:
    block = df.copy()
    block["candidate_name"] = candidate["candidate_name"]
    block["model_key"] = candidate["model_key"]
    block["family"] = candidate["family"]
    block["feature_set"] = candidate["feature_set"]
    block["sampling_method"] = candidate["sampling_method"]
    block["search_type"] = candidate["search_type"]
    block["scoring_metric"] = SCORING_METRIC
    block["best_cv_score"] = best_cv_score
    block["best_params_json"] = json.dumps(_to_builtin(best_params), sort_keys=True)
    block["fit_time_sec"] = round(fit_time_sec, 2)
    block["use_xgboost_gpu"] = candidate["model_key"] == "xgboost" and USE_XGBOOST_GPU
    return block


def build_final_results_dataframe(
    results_05: pd.DataFrame,
    results_pr: pd.DataFrame,
    results_recall: pd.DataFrame,
    candidate: dict,
    best_cv_score: float,
    best_params: dict,
    fit_time_sec: float,
) -> pd.DataFrame:
    df_05 = _prepare_result_block(results_05, "seuil_0_5")
    df_pr = _prepare_result_block(results_pr, "seuil_cv_opt_prc")
    df_recall = _prepare_result_block(
        results_recall,
        f"cv_recall_cible_{TARGET_RECALL}",
    )

    df_all = pd.concat([df_05, df_pr, df_recall], ignore_index=True)
    df_all = _add_candidate_metadata(
        df=df_all,
        candidate=candidate,
        best_cv_score=best_cv_score,
        best_params=best_params,
        fit_time_sec=fit_time_sec,
    )

    ordered_cols = [
        "candidate_name",
        "model",
        "model_key",
        "family",
        "feature_set",
        "sampling_method",
        "search_type",
        "scoring_metric",
        "strategie_seuil",
        "threshold",
        "best_cv_score",
        "fit_time_sec",
        "accuracy",
        "precision_1",
        "recall_1",
        "f1_1",
        "prc_auc",
        "train_accuracy",
        "train_precision_1",
        "train_recall_1",
        "train_f1_1",
        "train_prc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "best_params_json",
        "use_xgboost_gpu",
    ]
    existing_cols = [col for col in ordered_cols if col in df_all.columns]
    return df_all[existing_cols].copy()


def build_search_summary_row(
    candidate: dict,
    best_cv_score: float,
    best_params: dict,
    fit_time_sec: float,
) -> dict:
    return {
        "candidate_name": candidate["candidate_name"],
        "model_key": candidate["model_key"],
        "family": candidate["family"],
        "feature_set": candidate["feature_set"],
        "sampling_method": candidate["sampling_method"],
        "search_type": candidate["search_type"],
        "scoring_metric": SCORING_METRIC,
        "best_cv_score": best_cv_score,
        "fit_time_sec": round(fit_time_sec, 2),
        "best_params_json": json.dumps(_to_builtin(best_params), sort_keys=True),
        "use_xgboost_gpu": candidate["model_key"] == "xgboost" and USE_XGBOOST_GPU,
    }


def tune_candidate(candidate: dict, prepared: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    preprocessor = prepared["preprocessor"]

    print("=" * 80)
    print(f"TUNING CANDIDATE: {candidate['candidate_name']}")
    print(f"FEATURE SET     : {candidate['feature_set']}")
    print(f"MODEL KEY       : {candidate['model_key']}")
    print(f"SAMPLING        : {candidate['sampling_method']}")
    print("=" * 80)

    fit_kwargs = {}
    start = perf_counter()

    if candidate["model_key"] == "elastic_net":
        search = train_model_with_gridsearch(
            model=build_elastic_net_logistic_regression_pipeline(
                preprocessor=preprocessor,
                random_state=SEED,
                sampling_method=candidate["sampling_method"],
            ),
            X_train=X_train,
            y_train=y_train,
            param_grid=get_final_elastic_net_param_grid(),
            scoring=SCORING_METRIC,
            cv=CV_FOLDS,
            n_jobs=-1,
            verbose=1,
        )
        tuned_model = search.best_estimator_
        best_cv_score = float(search.best_score_)
        best_params = search.best_params_

    elif candidate["model_key"] == "random_forest":
        search = train_model_with_randomized_search(
            model=build_random_forest_pipeline(
                preprocessor=preprocessor,
                random_state=SEED,
                sampling_method=candidate["sampling_method"],
            ),
            X_train=X_train,
            y_train=y_train,
            param_distributions=get_final_random_forest_param_distributions(),
            n_iter=RANDOM_FOREST_SEARCH_ITER,
            scoring=SCORING_METRIC,
            cv=CV_FOLDS,
            n_jobs=-1,
            verbose=1,
            random_state=SEED,
        )
        tuned_model = search.best_estimator_
        best_cv_score = float(search.best_score_)
        best_params = search.best_params_

    elif candidate["model_key"] == "xgboost":
        search = train_model_with_randomized_search(
            model=build_xgboost_pipeline(
                preprocessor=preprocessor,
                random_state=SEED,
                sampling_method=candidate["sampling_method"],
                use_gpu=USE_XGBOOST_GPU,
                n_jobs=-1,
            ),
            X_train=X_train,
            y_train=y_train,
            param_distributions=get_final_xgboost_param_distributions(),
            n_iter=XGBOOST_SEARCH_ITER,
            scoring=SCORING_METRIC,
            cv=CV_FOLDS,
            n_jobs=1,
            verbose=1,
            random_state=SEED,
        )
        tuned_model = search.best_estimator_
        best_cv_score = float(search.best_score_)
        best_params = search.best_params_

    else:
        raise ValueError(f"model_key non supporte: {candidate['model_key']}")

    fit_time_sec = perf_counter() - start

    print(f"Best CV score ({SCORING_METRIC}): {best_cv_score:.4f}")
    print(f"Best params: {best_params}")

    trained_models = {candidate["candidate_name"]: tuned_model}
    oof_proba = {
        candidate["candidate_name"]: get_oof_predicted_proba(
            model=tuned_model,
            X=X_train,
            y=y_train,
            fit_kwargs=fit_kwargs,
            cv=CV_FOLDS,
            random_state=SEED,
        )
    }

    results_05 = compare_models(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
        threshold=0.5,
        sort_by="prc_auc",
        X_train=X_train,
        y_train=y_train,
    )
    results_pr = compare_models_with_cv_pr_optimal_threshold(
        trained_models=trained_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metric="f1",
        sort_by="prc_auc",
        oof_proba_by_model=oof_proba,
        cv=CV_FOLDS,
        random_state=SEED,
    )
    results_recall = compare_models_with_cv_target_recall(
        trained_models=trained_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        target_recall=TARGET_RECALL,
        sort_by="precision_1",
        oof_proba_by_model=oof_proba,
        cv=CV_FOLDS,
        random_state=SEED,
    )

    final_results = build_final_results_dataframe(
        results_05=results_05,
        results_pr=results_pr,
        results_recall=results_recall,
        candidate=candidate,
        best_cv_score=best_cv_score,
        best_params=best_params,
        fit_time_sec=fit_time_sec,
    )

    _, cv_summary = cross_validate_model_specs(
        model_specs={
            candidate["candidate_name"]: {
                "model": tuned_model,
                "fit_kwargs": fit_kwargs,
            }
        },
        X=X_train,
        y=y_train,
        cv=CV_FOLDS,
        threshold=0.5,
        random_state=SEED,
    )
    if not cv_summary.empty:
        cv_summary = _add_candidate_metadata(
            df=cv_summary,
            candidate=candidate,
            best_cv_score=best_cv_score,
            best_params=best_params,
            fit_time_sec=fit_time_sec,
        )

    search_row = build_search_summary_row(
        candidate=candidate,
        best_cv_score=best_cv_score,
        best_params=best_params,
        fit_time_sec=fit_time_sec,
    )
    return search_row, final_results, cv_summary


def build_best_by_candidate(all_results: pd.DataFrame) -> pd.DataFrame:
    best_rows = (
        all_results.sort_values(
            ["candidate_name", "prc_auc", "f1_1"],
            ascending=[True, False, False],
        )
        .groupby("candidate_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_rows = best_rows.sort_values(
        ["best_cv_score", "prc_auc", "f1_1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return best_rows


def export_results(
    run_config: pd.DataFrame,
    search_summary: pd.DataFrame,
    all_results: pd.DataFrame,
    best_by_candidate: pd.DataFrame,
    best_overall: pd.DataFrame,
    cv_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{pd.Timestamp.now().strftime('%y%m%d')}-final_tuning_shortlist.xlsx"
    output_path = output_dir / file_name

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        run_config.to_excel(writer, sheet_name="run_config", index=False)
        search_summary.to_excel(writer, sheet_name="search_summary", index=False)
        all_results.to_excel(writer, sheet_name="all_results", index=False)
        best_by_candidate.to_excel(writer, sheet_name="best_by_candidate", index=False)
        best_overall.to_excel(writer, sheet_name="best_overall", index=False)
        if not cv_summary.empty:
            cv_summary.to_excel(writer, sheet_name="cv_summary_05", index=False)

    return output_path


def main():
    data_path = PROJECT_ROOT / "data" / "interim" / "data_eda.csv"
    output_dir = PROJECT_ROOT / "data" / "processed"

    print("\n=== FINAL TUNING CONFIG ===")
    print(f"SCORING_METRIC        : {SCORING_METRIC}")
    print(f"TARGET_RECALL         : {TARGET_RECALL}")
    print(f"CV_FOLDS              : {CV_FOLDS}")
    print(f"USE_XGBOOST_GPU       : {USE_XGBOOST_GPU}")
    print(f"XGBOOST_SEARCH_ITER   : {XGBOOST_SEARCH_ITER}")
    print(f"RANDOM_FOREST_ITER    : {RANDOM_FOREST_SEARCH_ITER}")
    print(f"CANDIDATES            : {[c['candidate_name'] for c in FINAL_TUNING_CANDIDATES]}")
    print()

    prepared_cache: dict[str, dict] = {}
    search_rows: list[dict] = []
    result_blocks: list[pd.DataFrame] = []
    cv_blocks: list[pd.DataFrame] = []

    for candidate in FINAL_TUNING_CANDIDATES:
        feature_set = candidate["feature_set"]
        if feature_set not in prepared_cache:
            prepared_cache[feature_set] = prepare_dataset(
                data_path=data_path,
                feature_set_name=feature_set,
                test_size=TEST_SIZE,
                seed=SEED,
            )

        search_row, final_results, cv_summary = tune_candidate(
            candidate=candidate,
            prepared=prepared_cache[feature_set],
        )
        search_rows.append(search_row)
        result_blocks.append(final_results)
        if not cv_summary.empty:
            cv_blocks.append(cv_summary)

    search_summary = pd.DataFrame(search_rows)
    all_results = pd.concat(result_blocks, ignore_index=True)
    best_by_candidate = build_best_by_candidate(all_results)
    best_overall = best_by_candidate.head(1).reset_index(drop=True)
    cv_summary = pd.concat(cv_blocks, ignore_index=True) if cv_blocks else pd.DataFrame()

    run_config = pd.DataFrame(
        [
            {
                "seed": SEED,
                "test_size": TEST_SIZE,
                "scoring_metric": SCORING_METRIC,
                "target_recall": TARGET_RECALL,
                "cv_folds": CV_FOLDS,
                "use_xgboost_gpu": USE_XGBOOST_GPU,
                "xgboost_search_iter": XGBOOST_SEARCH_ITER,
                "random_forest_search_iter": RANDOM_FOREST_SEARCH_ITER,
            }
        ]
    )

    print("\n=== BEST BY CANDIDATE ===")
    print(
        best_by_candidate[
            [
                "candidate_name",
                "feature_set",
                "strategie_seuil",
                "threshold",
                "precision_1",
                "recall_1",
                "f1_1",
                "prc_auc",
                "best_cv_score",
            ]
        ]
    )

    print("\n=== BEST OVERALL ===")
    print(best_overall)

    output_path = export_results(
        run_config=run_config,
        search_summary=search_summary,
        all_results=all_results,
        best_by_candidate=best_by_candidate,
        best_overall=best_overall,
        cv_summary=cv_summary,
        output_dir=output_dir,
    )
    print(f"\nFinal tuning exported: {output_path}")


if __name__ == "__main__":
    main()
