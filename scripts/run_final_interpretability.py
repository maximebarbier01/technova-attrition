from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names.*",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.lightgbm_model import build_lightgbm_pipeline
from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.data.split_data import make_train_test_split
from src.features.feature_engineering import make_feature_engineering
from src.features.features_selection import DROP_COLUMNS, TARGET, get_feature_set
from src.utils.visualization import (
    plot_categorical_feature_diagnostics,
    plot_numeric_feature_diagnostics,
    plot_probability_distrib_per_pred_type,
)

SEED = 51
TEST_SIZE = 0.2
DATE_TAG = pd.Timestamp.now().strftime("%y%m%d")

FINAL_MODEL = {
    "candidate_name": "best_lightgbm_raw_baseline",
    "feature_set": "raw_baseline",
    "sampling_method": "borderline",
    "threshold": 0.211717,
    "params": {
        "n_estimators": 700,
        "learning_rate": 0.01,
        "num_leaves": 63,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.01,
        "reg_lambda": 0.5,
        "class_weight": None,
    },
}
OUTPUT_PREFIX = f"{DATE_TAG}-final_lightgbm_raw_baseline"

NUMERIC_DIAGNOSTIC_FEATURES = [
    "revenu_mensuel",
    "age",
    "distance_domicile_travail",
    "annee_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "annees_depuis_la_derniere_promotion",
    "satisfaction_employee_environnement",
    "satisfaction_employee_nature_travail",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "note_evaluation_precedente",
    "note_evaluation_actuelle",
    "niveau_hierarchique_poste",
    "nb_formations_suivies",
    "nombre_participation_pee",
]

CATEGORICAL_DIAGNOSTIC_FEATURES = [
    "statut_marital",
    "frequence_deplacement",
    "departement",
    "poste",
    "revenu_bin",
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
        "preprocessor": preprocessor,
        "context_test_rows": df.loc[X_test.index].copy(),
    }


def fit_final_model(preprocessor, X_train: pd.DataFrame, y_train: pd.Series):
    model = build_lightgbm_pipeline(
        preprocessor=preprocessor,
        random_state=SEED,
        sampling_method=FINAL_MODEL["sampling_method"],
        n_estimators=FINAL_MODEL["params"]["n_estimators"],
        learning_rate=FINAL_MODEL["params"]["learning_rate"],
        num_leaves=FINAL_MODEL["params"]["num_leaves"],
        max_depth=FINAL_MODEL["params"]["max_depth"],
        min_child_samples=FINAL_MODEL["params"]["min_child_samples"],
        subsample=FINAL_MODEL["params"]["subsample"],
        colsample_bytree=FINAL_MODEL["params"]["colsample_bytree"],
        reg_alpha=FINAL_MODEL["params"]["reg_alpha"],
        reg_lambda=FINAL_MODEL["params"]["reg_lambda"],
        class_weight=FINAL_MODEL["params"]["class_weight"],
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    return model


def build_predictions_dataframe(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    context_test_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= FINAL_MODEL["threshold"]).astype(int)

    df = X_test.copy()
    context_columns = [
        "satisfaction_global",
        "revenu_bin",
        "age_bucket",
        "salary_gap_vs_poste_median",
        "satisfaction_gap_vs_poste_mean",
        "role_stagnation_ratio",
    ]
    if context_test_rows is not None:
        for column in context_columns:
            if column in context_test_rows.columns and column not in df.columns:
                df[column] = context_test_rows[column]
    df["y_true"] = y_test.values
    df["y_pred"] = y_pred
    df["y_proba"] = y_proba
    df["threshold"] = FINAL_MODEL["threshold"]
    df["margin_vs_threshold"] = df["y_proba"] - df["threshold"]
    df["abs_margin_vs_threshold"] = df["margin_vs_threshold"].abs()
    df["proba_rank_pct"] = (
        pd.Series(y_proba, index=df.index).rank(pct=True, method="average")
    )

    df["error_type"] = np.select(
        [
            (df["y_true"] == 1) & (df["y_pred"] == 1),
            (df["y_true"] == 0) & (df["y_pred"] == 0),
            (df["y_true"] == 0) & (df["y_pred"] == 1),
            (df["y_true"] == 1) & (df["y_pred"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )
    return df


def compute_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring="average_precision",
        n_repeats=20,
        random_state=SEED,
        n_jobs=1,
    )

    df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return df.reset_index(drop=True)


def save_permutation_importance_plot(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    top = importance_df.head(top_n).sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#FF857B",
        alpha=0.9,
    )
    ax.set_title("Permutation Importance - Modele final")
    ax.set_xlabel("Contribution à la détection des départs")
    ax.set_ylabel("Features")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def transform_feature_matrices(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = model.named_steps["prep"]
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train_t = pd.DataFrame(
        preprocessor.transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_test_t = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )
    return X_train_t, X_test_t


def compute_shap_explanations(
    model,
    X_train_t: pd.DataFrame,
    X_test_t: pd.DataFrame,
):
    tree_model = model.named_steps["model"]
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer(X_test_t)

    if getattr(shap_values, "values", np.array([])).ndim == 3:
        shap_values = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=X_test_t.values,
            feature_names=X_test_t.columns.tolist(),
        )

    return shap_values


def save_global_shap_outputs(
    shap_values,
    output_plot_path: Path,
    output_csv_path: Path,
) -> pd.DataFrame:
    shap_importance_df = pd.DataFrame(
        {
            "feature": shap_values.feature_names,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance_df.to_csv(output_csv_path, index=False)

    plt.figure(figsize=(11, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP global - Modele final", pad=16)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return shap_importance_df


def get_case_indices(predictions_df: pd.DataFrame) -> dict[str, int]:
    tp_idx = (
        predictions_df[
            (predictions_df["y_true"] == 1) & (predictions_df["y_pred"] == 1)
        ]
        .sort_values("y_proba", ascending=False)
        .index[0]
    )
    fn_idx = (
        predictions_df[predictions_df["error_type"] == "FN"]
        .sort_values("y_proba", ascending=True)
        .index[0]
    )
    fp_idx = (
        predictions_df[predictions_df["error_type"] == "FP"]
        .sort_values("y_proba", ascending=False)
        .index[0]
    )
    return {
        "tp_high_risk": tp_idx,
        "fn_silent_attrition": fn_idx,
        "fp_extreme_alert": fp_idx,
    }


def format_top_contributors(explanation_row, top_n: int = 5) -> tuple[str, str]:
    contrib = pd.Series(explanation_row.values, index=explanation_row.feature_names)
    top_positive = contrib[contrib > 0].sort_values(ascending=False).head(top_n)
    top_negative = contrib[contrib < 0].sort_values().head(top_n)

    pos_text = " | ".join(f"{name}={value:+.3f}" for name, value in top_positive.items())
    neg_text = " | ".join(f"{name}={value:+.3f}" for name, value in top_negative.items())
    return pos_text, neg_text


def save_local_waterfall_plot(explanation_row, output_path: Path, title: str) -> None:
    plt.figure(figsize=(11, 6))
    shap.plots.waterfall(explanation_row, max_display=12, show=False)
    plt.title(title, pad=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_local_cases_summary(
    predictions_df: pd.DataFrame,
    shap_values,
    case_indices: dict[str, int],
    x_test_index: list[int],
) -> pd.DataFrame:
    rows = []
    index_lookup = list(x_test_index)

    for case_name, idx in case_indices.items():
        explanation_row = shap_values[index_lookup.index(idx)]
        top_positive, top_negative = format_top_contributors(explanation_row)
        base = predictions_df.loc[idx]

        rows.append(
            {
                "case_name": case_name,
                "row_index": idx,
                "error_type": base["error_type"],
                "y_true": base["y_true"],
                "y_pred": base["y_pred"],
                "y_proba": base["y_proba"],
                "threshold": base["threshold"],
                "departement": base.get("departement"),
                "poste": base.get("poste"),
                "statut_marital": base.get("statut_marital"),
                "frequence_deplacement": base.get("frequence_deplacement"),
                "satisfaction_global": base.get("satisfaction_global"),
                "revenu_mensuel": base.get("revenu_mensuel"),
                "top_positive_shap": top_positive,
                "top_negative_shap": top_negative,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "interim" / "data_eda.csv"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "reports" / "figure"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_dataset(
        data_path=data_path,
        feature_set_name=FINAL_MODEL["feature_set"],
    )

    model = fit_final_model(
        preprocessor=prepared["preprocessor"],
        X_train=prepared["X_train"],
        y_train=prepared["y_train"],
    )

    predictions_df = build_predictions_dataframe(
        model=model,
        X_test=prepared["X_test"],
        y_test=prepared["y_test"],
        context_test_rows=prepared["context_test_rows"],
    )

    permutation_df = compute_permutation_importance(
        model=model,
        X_test=prepared["X_test"],
        y_test=prepared["y_test"],
    )
    permutation_csv_path = processed_dir / f"{OUTPUT_PREFIX}_permutation_importance.csv"
    permutation_png_path = figures_dir / f"{OUTPUT_PREFIX}_permutation_importance.png"
    permutation_df.to_csv(permutation_csv_path, index=False)
    save_permutation_importance_plot(permutation_df, permutation_png_path)

    X_train_t, X_test_t = transform_feature_matrices(
        model=model,
        X_train=prepared["X_train"],
        X_test=prepared["X_test"],
    )
    shap_values = compute_shap_explanations(
        model=model,
        X_train_t=X_train_t,
        X_test_t=X_test_t,
    )

    shap_global_csv_path = processed_dir / f"{OUTPUT_PREFIX}_shap_importance.csv"
    shap_global_png_path = figures_dir / f"{OUTPUT_PREFIX}_shap_beeswarm.png"
    shap_importance_df = save_global_shap_outputs(
        shap_values=shap_values,
        output_plot_path=shap_global_png_path,
        output_csv_path=shap_global_csv_path,
    )

    case_indices = get_case_indices(predictions_df)
    local_summary_df = build_local_cases_summary(
        predictions_df=predictions_df,
        shap_values=shap_values,
        case_indices=case_indices,
        x_test_index=X_test_t.index.tolist(),
    )
    local_summary_csv_path = processed_dir / f"{OUTPUT_PREFIX}_local_cases.csv"
    local_summary_df.to_csv(local_summary_csv_path, index=False)

    index_lookup = X_test_t.index.tolist()
    for case_name, idx in case_indices.items():
        explanation_row = shap_values[index_lookup.index(idx)]
        case_meta = local_summary_df[local_summary_df["case_name"] == case_name].iloc[0]
        title = (
            f"{case_name} - row {idx} - "
            f"{case_meta['departement']} / {case_meta['poste']}"
        )
        output_path = figures_dir / f"{OUTPUT_PREFIX}_{case_name}_waterfall.png"
        save_local_waterfall_plot(explanation_row, output_path, title)


    diagnostics_dir = figures_dir / "final_model_diagnostics"
    probability_plot_path = diagnostics_dir / f"{OUTPUT_PREFIX}_probability_errors_and_tp.png"
    numeric_diag_dir = diagnostics_dir / "numeric"
    categorical_diag_dir = diagnostics_dir / "categorical"

    plot_probability_distrib_per_pred_type(
        model=model,
        X=prepared["X_test"],
        y=prepared["y_test"],
        threshold=FINAL_MODEL["threshold"],
        categories_to_exclude=["true_negative"],
        save_path=probability_plot_path,
        show=False,
    )

    probability_departures_plot_path = diagnostics_dir / f"{OUTPUT_PREFIX}_probability_tp_vs_fn.png"
    plot_probability_distrib_per_pred_type(
        model=model,
        X=prepared["X_test"],
        y=prepared["y_test"],
        threshold=FINAL_MODEL["threshold"],
        categories_to_exclude=["true_negative", "false_positive"],
        save_path=probability_departures_plot_path,
        show=False,
    )

    numeric_diag_paths = plot_numeric_feature_diagnostics(
        model=model,
        X=prepared["X_test"],
        y=prepared["y_test"],
        num_features=NUMERIC_DIAGNOSTIC_FEATURES,
        threshold=FINAL_MODEL["threshold"],
        kind="kde",
        output_dir=numeric_diag_dir,
        filename_prefix=f"{OUTPUT_PREFIX}_numeric_",
        show=False,
    )

    categorical_diag_paths = plot_categorical_feature_diagnostics(
        model=model,
        X=prepared["X_test"],
        y=prepared["y_test"],
        cat_features=CATEGORICAL_DIAGNOSTIC_FEATURES,
        threshold=FINAL_MODEL["threshold"],
        normalize=True,
        top_n=8,
        output_dir=categorical_diag_dir,
        filename_prefix=f"{OUTPUT_PREFIX}_categorical_",
        show=False,
    )

    print("\n=== FINAL INTERPRETABILITY OUTPUTS ===")
    print(f"Permutation importance CSV : {permutation_csv_path}")
    print(f"Permutation importance PNG : {permutation_png_path}")
    print(f"SHAP global CSV            : {shap_global_csv_path}")
    print(f"SHAP beeswarm PNG          : {shap_global_png_path}")
    print(f"Local cases CSV            : {local_summary_csv_path}")
    print(f"Probability diag PNG       : {probability_plot_path}")
    print(f"Probability TP/FN PNG      : {probability_departures_plot_path}")
    print(f"Numeric diag PNG count     : {len(numeric_diag_paths)}")
    print(f"Categorical diag PNG count : {len(categorical_diag_paths)}")
    print("\n=== SELECTED LOCAL CASES ===")
    print(local_summary_df.to_string(index=False))
    print("\n=== TOP PERMUTATION FEATURES ===")
    print(permutation_df.head(10).to_string(index=False))
    print("\n=== TOP GLOBAL SHAP FEATURES ===")
    print(shap_importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
