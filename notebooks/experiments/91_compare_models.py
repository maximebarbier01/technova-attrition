import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.split_data import make_train_test_split
from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.features.features_selection import NUM_FEATURES, CAT_FEATURES, TARGET, DROP_COLUMNS
from src.models.logistic_regression_model import build_logistic_regression_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost_model import build_xgboost_pipeline
from src.models.catboost_model import build_catboost_model
from src.models.train import train_model
from src.models.compare import compare_models, compare_models_with_optimal_threshold, find_best_threshold, evaluate_binary_classifier, compare_models_with_pr_optimal_threshold, compare_models_with_target_recall

# *****************************************
# *          IMPORT DES TABLES            *
# *****************************************

df = pd.read_csv("/home/maxime/projects/technova-attrition/data/interim/data.csv")
df = df.drop(columns=DROP_COLUMNS).copy()

# *****************************************
# *            SPLIT & SCALE              *
# *****************************************

feature_columns = NUM_FEATURES + CAT_FEATURES

X_train, X_test, y_train, y_test = make_train_test_split(
    df=df,
    feature_columns=feature_columns,
    target_column=TARGET,
    test_size=0.2,
    random_state=51,
    stratify=True,
)

num_features = filter_existing_features(NUM_FEATURES, X_train.columns.tolist())
cat_features = filter_existing_features(CAT_FEATURES, X_train.columns.tolist())

preprocessor = build_preprocessor(
    num_features=num_features,
    cat_features=cat_features,
)

# *****************************************
# *              MODELES                  *
# *****************************************

log_reg = build_logistic_regression_pipeline(preprocessor, random_state=51)
rf = build_random_forest_pipeline(preprocessor, random_state=51)
xgb = build_xgboost_pipeline(preprocessor, random_state=51)
cat = build_catboost_model(random_state=51)

model_specs = {
    "log_reg": {
        "model": log_reg,
        "fit_kwargs": {},
    },
    "random_forest": {
        "model": rf,
        "fit_kwargs": {},
    },
    "xgboost": {
        "model": xgb,
        "fit_kwargs": {},
    },
    "catboost": {
        "model": cat,
        "fit_kwargs": {"cat_features": cat_features},
    },
}

# *****************************************
# *             ENTRAINEMENT              *
# *****************************************

trained_models = {}

for model_name, spec in model_specs.items():
    print(f"Entraînement : {model_name}")
    trained_models[model_name] = train_model(
        model=spec["model"],
        X_train=X_train,
        y_train=y_train,
        **spec["fit_kwargs"],
    )

# *****************************************
# *        COMPARAISON SEUIL 0.5          *
# *****************************************

results_05 = compare_models(
    trained_models=trained_models,
    X_test=X_test,
    y_test=y_test,
    threshold=0.5,
    sort_by="roc_auc",
)

results_05.round(3)

# *****************************************
# *        COMPARAISON SEUIL OPT          *
# *****************************************

results_opt = compare_models_with_optimal_threshold(
    trained_models=trained_models,
    X_test=X_test,
    y_test=y_test,
    metric="f1",
    sort_by="f1_1",
)

results_opt.round(2)

# *****************************************
# *     COMPARAISON SEUIL OPT sur PRC     *
# *****************************************

results_pr = compare_models_with_pr_optimal_threshold(
    trained_models=trained_models,
    X_test=X_test,
    y_test=y_test,
    metric="f1",
    sort_by="f1_1",
)

results_pr.round(2)

# *****************************************
# *  COMPARAISON SEUIL POUR RECALL CIBLE  *
# *****************************************

from src.models.compare import compare_models_with_target_recall

results_recall = compare_models_with_target_recall(
    trained_models=trained_models,
    X_test=X_test,
    y_test=y_test,
    target_recall=0.70,
    sort_by="precision_1",
)

results_recall.round(3)

# *****************************************
# *        ZOOM SUR           *
# *****************************************

best_thresh_xgb, best_f1_xgb = find_best_threshold(
    model=trained_models["xgboost"],
    X_test=X_test,
    y_test=y_test,
)

xgb_metrics = evaluate_binary_classifier(
    model=trained_models["catboost"],
    X_test=X_test,
    y_test=y_test,
    threshold=best_thresh_xgb,
)

xgb_metrics = pd.DataFrame(xgb_metrics)

print(best_thresh_xgb, best_f1_xgb)
print(xgb_metrics)