from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def filter_existing_features(feature_list: List[str], df_columns: List[str]) -> List[str]:
    return [col for col in feature_list if col in df_columns]


def get_numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def get_categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessor(
    num_features: List[str],
    cat_features: List[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", get_numeric_pipeline(), num_features),
            ("cat", get_categorical_pipeline(), cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )