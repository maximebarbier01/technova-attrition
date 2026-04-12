
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_preprocessor
from src.modeling.train import train_model
from src.modeling.compare import get_oof_predicted_proba


# exemple de création d'un dataset de test
from sklearn.datasets import make_blobs
from collections import Counter

# création des entrées et sorties
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)

# résumé de la forme des tableaux
print(X.shape, y.shape)

Counter(y)

# évaluer le modèle en moyennant la performance sur chaque fold
from numpy import mean
from numpy import std
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# example of random oversampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler


import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.modeling.compare import (
    compare_models,
    compare_models_with_cv_pr_optimal_threshold,
    compare_models_with_cv_target_recall,
    cross_validate_model_specs,
    get_oof_predicted_proba_by_model_specs,
    get_oof_predicted_proba
)
from src.modeling.model_specs import (
    get_baseline_model_specs,
    get_new_models_model_specs,
    get_optuna_model_specs,
    get_tuned_model_specs,
)
from src.modeling.train import train_model
from sklearn.linear_model import LogisticRegression
# création des entrées et sorties

X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)

model = LogisticRegression(max_iter=1000)

oof = get_oof_predicted_proba(
    model= model,
    X=X,
    y=y
)

oof.shape

find_best_threshold_from_proba(y)
