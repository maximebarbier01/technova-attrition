from __future__ import annotations

import sys
from pathlib import Path

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing import build_preprocessor

#*****************************
#* 1. Baseline sans sampling *
#*****************************

def build_baseline_pipeline(model, num_features, cat_features):
    preprocessor = build_preprocessor(num_features, cat_features)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline

#**************************
#* 2. Pipeline avec SMOTE *
#**************************

def build_smote_pipeline(
    preprocessor,
    model,
    sampling_strategy: float = 0.3,
    k_neighbors: int = 5,
    random_state: int = 42,
):
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "sampling",
                SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                ),
            ),
            ("model", model),
        ]
    )

#************************************
#* 3. Pipeline avec BorderlineSMOTE *
#************************************

def build_borderline_smote_pipeline(
    preprocessor,
    model,
    sampling_strategy: float = 0.3,
    k_neighbors: int = 5,
    random_state: int = 42,
):
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "sampling",
                BorderlineSMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                ),
            ),
            ("model", model),
        ]
    )


#******************************************
#* 4. Pipeline avec SMOTE + undersampling *
#******************************************

def build_smote_under_pipeline(
    preprocessor,
    model,
    over_strategy: float = 0.3,
    under_strategy: float = 0.7,
    k_neighbors: int = 5,
    random_state: int = 42,
):
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "over",
                SMOTE(
                    sampling_strategy=over_strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                ),
            ),
            (
                "under",
                RandomUnderSampler(
                    sampling_strategy=under_strategy,
                    random_state=random_state,
                ),
            ),
            ("model", model),
        ]
    )

#***************************
#* 4. Pipeline avec ADASYN *
#***************************

def build_adasyn_pipeline(
    preprocessor,
    model,
    sampling_strategy: float = 0.3,
    n_neighbors: int = 5,
    random_state: int = 42,
):
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "sampling",
                ADASYN(
                    sampling_strategy=sampling_strategy,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                ),
            ),
            ("model", model),
        ]
    )

#******************************
#* 4. Pipeline avec SVM-SMOTE *
#******************************

def build_svmsmote_pipeline(
    preprocessor,
    model,
    sampling_strategy: float = 0.3,
    k_neighbors: int = 5,
    random_state: int = 42,
):
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "sampling",
                SVMSMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                ),
            ),
            ("model", model),
        ]
    )