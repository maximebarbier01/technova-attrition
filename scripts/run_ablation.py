from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.xgboost_model import build_xgboost_pipeline
from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.data.split_data import make_train_test_split
from src.features.feature_engineering import make_feature_engineering
from src.features.features_selection import DROP_COLUMNS, TARGET, get_feature_set
from src.modeling.compare import evaluate_binary_classifier
from src.modeling.train import train_model

DATA_PATH = PROJECT_ROOT / "data" / "interim" / "data_eda.csv"

TO_TEST = [
    "raw_baseline",
    "raw_plus_buckets",
    "fe_core",
    "fe_compact",
    "fe_fn_focus",
    "fe_business",
    "fe_full_robust",
    "fe_linear_clean",
]


def prepare_dataframe(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    return make_feature_engineering(df)


def run_ablation_study(
    df: pd.DataFrame,
    feature_set_names: list[str],
    model_builder,
    seed: int = 51,
    test_size: float = 0.2,
) -> pd.DataFrame:
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

        model = model_builder(preprocessor, seed)
        model = train_model(model, X_train, y_train)

        metrics = evaluate_binary_classifier(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            threshold=0.5,
        )

        metrics["feature_set"] = feature_set_name
        metrics["n_features"] = len(num_features) + len(cat_features)
        metrics["threshold"] = 0.5
        results.append(metrics)

    results_df = pd.DataFrame(results)

    return results_df.sort_values(
        ["prc_auc", "f1_1"],
        ascending=[False, False],
    ).reset_index(drop=True)



def xgb_builder(preprocessor, seed):
    return build_xgboost_pipeline(
        preprocessor=preprocessor,
        random_state=seed,
        sampling_method="borderline",
    )



def main() -> None:
    seed = 42
    df = prepare_dataframe(DATA_PATH)

    ablation_results = run_ablation_study(
        df=df,
        feature_set_names=TO_TEST,
        model_builder=xgb_builder,
        seed=seed,
    )

    print("\n=== ABLATION RESULTS (sorted by PRC AUC) ===")
    print(ablation_results.round(3))


if __name__ == "__main__":
    main()
