from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare un DataFrame en features X et target y.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    feature_columns : Sequence[str]
        Colonnes utilisées comme variables explicatives.
    target_column : str
        Nom de la colonne cible.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X et y.
    """
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Colonnes features absentes du DataFrame : {missing_features}"
        )

    if target_column not in df.columns:
        raise ValueError(f"Colonne cible absente du DataFrame : {target_column}")

    X = df[list(feature_columns)].copy()
    y = df[target_column].copy()

    X.columns = X.columns.astype(str)

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Réalise un train/test split.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    test_size : float, default=0.2
        Proportion du jeu de test.
    random_state : int, default=42
        Seed pour reproductibilité.
    stratify : bool, default=True
        Active la stratification sur y, utile en classification.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    stratify_arg = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )


def make_train_test_split(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Enchaîne :
    1. séparation X / y
    2. train/test split

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    feature_columns : Sequence[str]
        Colonnes des features.
    target_column : str
        Colonne cible.
    test_size : float, default=0.2
        Taille du test set.
    random_state : int, default=42
        Seed de reproductibilité.
    stratify : bool, default=True
        Stratification sur y.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X, y = split_features_target(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    return split_train_test(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

def make_train_val_test_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    """
    Split en train / validation / test.

    val_size est exprimé sur le dataset total.
    Exemple :
    - test_size=0.2
    - val_size=0.2
    => 60% train / 20% val / 20% test
    """
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    stratify_y = y if stratify else None

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    # proportion de validation à prendre dans le bloc train_full
    val_relative_size = val_size / (1 - test_size)

    stratify_train_full = y_train_full if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=stratify_train_full,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test