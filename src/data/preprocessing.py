from __future__ import annotations

from typing import Any, List, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

OUTLIER_RATIO_THRESHOLD = 0.05

ORDINAL_CATEGORY_ORDERS = {
    "frequence_deplacement": ["Aucun", "Occasionnel", "Frequent"],
    "age_bucket": ["young", "mid", "senior"],
    "niveau_education": [1, 2, 3, 4, 5],
}


def filter_existing_features(feature_list: List[str], df_columns: List[str]) -> List[str]:
    return [col for col in feature_list if col in df_columns]


def get_numeric_pipeline(scaler: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )


def get_binary_numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )


def get_nominal_categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
            ),
        ]
    )


def get_ordinal_categorical_pipeline(categories: Sequence[Sequence[Any]]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=list(categories),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def _is_binary_numeric_feature(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False

    unique_values = set(numeric.unique().tolist())
    return unique_values.issubset({0, 1})


def _has_outliers(series: pd.Series, threshold: float = OUTLIER_RATIO_THRESHOLD) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if numeric.empty or numeric.nunique() <= 2:
        return False

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return False

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_ratio = ((numeric < lower_bound) | (numeric > upper_bound)).mean()

    return bool(outlier_ratio >= threshold)


def _split_numeric_features(
    num_features: Sequence[str],
    X_reference: pd.DataFrame | None,
) -> tuple[list[str], list[str], list[str]]:
    if X_reference is None:
        return [], [], list(num_features)

    binary_features: list[str] = []
    robust_features: list[str] = []
    standard_features: list[str] = []

    for feature in num_features:
        if feature not in X_reference.columns:
            continue

        series = X_reference[feature]

        if _is_binary_numeric_feature(series):
            binary_features.append(feature)
        elif _has_outliers(series):
            robust_features.append(feature)
        else:
            standard_features.append(feature)

    return binary_features, robust_features, standard_features


def _split_categorical_features(
    cat_features: Sequence[str],
) -> tuple[list[str], list[str]]:
    ordinal_features = [
        feature for feature in cat_features if feature in ORDINAL_CATEGORY_ORDERS
    ]
    nominal_features = [
        feature for feature in cat_features if feature not in ORDINAL_CATEGORY_ORDERS
    ]
    return ordinal_features, nominal_features


def build_preprocessor(
    num_features: List[str],
    cat_features: List[str],
    X_reference: pd.DataFrame | None = None,
) -> ColumnTransformer:
    binary_num_features, robust_num_features, standard_num_features = (
        _split_numeric_features(
            num_features=num_features,
            X_reference=X_reference,
        )
    )
    ordinal_cat_features, nominal_cat_features = _split_categorical_features(
        cat_features=cat_features,
    )

    transformers = []

    if standard_num_features:
        transformers.append(
            ("num_standard", get_numeric_pipeline(StandardScaler()), standard_num_features)
        )

    if robust_num_features:
        transformers.append(
            ("num_robust", get_numeric_pipeline(RobustScaler()), robust_num_features)
        )

    if binary_num_features:
        transformers.append(
            ("num_binary", get_binary_numeric_pipeline(), binary_num_features)
        )

    if ordinal_cat_features:
        ordinal_categories = [
            ORDINAL_CATEGORY_ORDERS[feature] for feature in ordinal_cat_features
        ]
        transformers.append(
            (
                "cat_ordinal",
                get_ordinal_categorical_pipeline(ordinal_categories),
                ordinal_cat_features,
            )
        )

    if nominal_cat_features:
        transformers.append(
            (
                "cat_nominal",
                get_nominal_categorical_pipeline(),
                nominal_cat_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
