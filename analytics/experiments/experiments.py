import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc

PROJECT_ROOT = Path.cwd().resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.split_data import make_train_test_split
from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.features.features_selection import TARGET, DROP_COLUMNS, get_feature_set
from src.features.feature_engineering import make_feature_engineering
from src.models.model_specs import get_baseline_model_specs, get_tuned_model_specs
from src.models.train import train_model, train_model_with_gridsearch, train_model_with_randomized_search
from src.models.compare import compare_models, compare_models_with_optimal_threshold, find_best_threshold, evaluate_binary_classifier, compare_models_with_pr_optimal_threshold, compare_models_with_target_recall
from src.utils.visualization import (
    plot_probability_distrib_per_pred_type, 
    plot_numeric_distributions_by_prediction_type, 
    plot_numeric_feature_diagnostics, 
    summarize_numeric_feature_diagnostics,
    plot_categorical_feature_diagnostics
)

# *****************************************
# *          IMPORT DES TABLES            *
# *****************************************

df = pd.read_csv("/home/maxime/projects/technova-attrition/data/interim/data_eda.csv")
df = df.drop(columns=DROP_COLUMNS).copy()

# *****************************************
# *        FEATURES ENGINEERING           *
# *****************************************

df = make_feature_engineering(df)

# *****************************************
# *           FEATURES SET                *
# *****************************************

feature_set_name = "fe_full_robust"
feature_config = get_feature_set(feature_set_name)

NUM_FEATURES = feature_config["num"]
CAT_FEATURES = feature_config["cat"]

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

#? PARAMETRES 

seed=51
scoring_metric = "average_precision"

baseline_specs = get_baseline_model_specs(
    preprocessor=preprocessor,
    cat_features=cat_features,
    seed=seed
)

tuned_specs = get_tuned_model_specs(
    preprocessor=preprocessor,
    cat_features=cat_features,
    X_train=X_train,
    y_train=y_train,
    seed=seed,
    scoring=scoring_metric
    
)

model_specs = {**baseline_specs, **tuned_specs}

# *****************************************
# *             ENTRAINEMENT              *
# *****************************************

trained_models = {}

for model_name, spec in model_specs.items():
    if spec["already_trained"]:
        trained_models[model_name] = spec["model"]
    else:
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

results_05.round(2)

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

#todo tracer la courbe precision rappel selon le modèle utilisé 

model = "best_random_forest_grid"

y_proba = trained_models[model].predict_proba(X_test)[:, 1]

precision_test, recall_test, thresholds_test = precision_recall_curve(
    y_test, y_proba
)

plt.plot(recall_test, precision_test, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"PRC de {model}")
plt.legend()
plt.show()

auc_test = auc(recall_test, precision_test)

# *****************************************
# *     COMPARAISON PAR RECALL CIBLE      *
# *****************************************

from src.models.compare import compare_models_with_target_recall

results_recall = compare_models_with_target_recall(
    trained_models=trained_models,
    X_test=X_test,
    y_test=y_test,
    target_recall=0.9,
    sort_by="precision_1",
)

results_recall.round(2)

# ******************************************
# *  DATASET AVEC TOUTES LES COMPARAISONS  *
# ******************************************

df1 = results_05[['model', 'threshold', 'precision_1', 'recall_1', 'f1_1',
                  'prc_auc', 'tn', 'fp', 'fn', 'tp']].copy()

df3 = results_pr[['model', 'best_threshold', 'precision_1', 'recall_1', 'f1_1',
                  'prc_auc', 'tn', 'fp', 'fn', 'tp']].copy()

df4 = results_recall[['model', 'best_threshold', 'precision_1', 'recall_1', 'f1_1',
                      'prc_auc', 'tn', 'fp', 'fn', 'tp']].copy()

# Harmonisation
df1 = df1.rename(columns={"threshold": "best_threshold"})

# Colonnes de contexte
df1["strategie_seuil"] = "seuil à 0.5"
df3["strategie_seuil"] = "seuil optimisé sur PRC"
df4["strategie_seuil"] = "recall cible à 0.9"

df1["feature_set"] = feature_set_name
df3["feature_set"] = feature_set_name
df4["feature_set"] = feature_set_name

# Concat
df_all_results = pd.concat([df1, df3, df4], ignore_index=True)

# Ordre des colonnes
cols = df_all_results.columns.tolist()

cols.insert(cols.index("model") + 1, cols.pop(cols.index("strategie_seuil")))
cols.insert(cols.index("strategie_seuil") + 1, cols.pop(cols.index("feature_set")))

df_all_results = df_all_results[cols]

#? EXPORT 

output_dir = Path("/home/maxime/projects/technova-attrition/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

file_path = output_dir / f"all_results_{feature_set_name}.xlsx"
df_all_results.to_excel(file_path, index=False)

print(f"Fichier exporté : {file_path}")

# *****************************************
# *           ZOOM SUR UN MODEL           *
# *****************************************

best_model = "random_forest"

best_thresh_xgb, best_f1_xgb = find_best_threshold(
    model=trained_models[best_model],
    X_test=X_test,
    y_test=y_test,
)

plot_probability_distrib_per_pred_type(
    model=trained_models[best_model],  # ou log_reg etc
    X=X_test,
    y=y_test,
    threshold=0.17,
    categories_to_exclude=["true_negative","false_positive"],
)

plot_probability_distrib_per_pred_type(
    model=trained_models[best_model],  # ou log_reg etc
    X=X_test,
    y=y_test,
    threshold=best_thresh_xgb,
    categories_to_exclude=["true_positive", "false_negative"],
)

plot_numeric_distributions_by_prediction_type(
    model=trained_models[best_model],
    X=X_test,
    y=y_test,
    num_features=NUM_FEATURES,
    threshold=best_thresh_xgb,
    features_per_row=3
)

plot_numeric_feature_diagnostics(
    model=trained_models[best_model],
    X=X_test,
    y=y_test,
    num_features=num_features,
    threshold=best_thresh_xgb,
    kind="kde",
)

plot_categorical_feature_diagnostics(
    model=trained_models[best_model],
    X=X_test,
    y=y_test,
    cat_features=CAT_FEATURES,
    threshold=best_thresh_xgb,
    top_n=20
)