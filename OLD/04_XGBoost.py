# =========================
# Standard library
# =========================
import importlib
import math
import os
import re
import sys
import time
# Pour mise en forme des résultats
import colorama

# =========================
# Data / scientific stack
# =========================
import numpy as np
import pandas as pd
import pingouin as pg

# =========================
# Visualisation
# =========================
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

# =========================
# Machine Learning
# =========================
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, classification_report, precision_recall_curve, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.model_selection import GridSearchCV
# =========================
# Local imports (src/)
# =========================
sys.path.append(os.path.abspath(".."))  # si notebook dans /notebooks

import src.outliers as of
import src.distrib_pred_type as dpt
import src.outliers_treatment as ot


# *****************************************
# *          IMPORT DES TABLES            *
# *****************************************

df_backup = pd.read_csv(
    "/home/maxime/projects/technova-attrition/data/interim/data.csv"
)
df_backup.drop(["Unnamed: 0", "id_employee", "eval_number", "code_sondage"], axis=1, inplace=True)

df = df_backup.copy()
df.info()
df.head(2)

df.columns

# *****************************************
# *        SELECTION DES FEATURES         *
# *****************************************

num_features = [
    "age",
    "revenu_mensuel",
    "annee_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "annees_depuis_la_derniere_promotion",
    "distance_domicile_travail",
    "niveau_hierarchique_poste",
    "satisfaction_employee_environnement",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "satisfaction_global",
    "note_evaluation_precedente",
    "note_evaluation_actuelle",
    "delta_evaluation",
    "salary_vs_level",
    "salary_vs_tenure",
    "promotion_speed",
    "experience_mismatch",
    "salary_gap_level",
    "career_stagnation",
]

cat_features = [
    "genre",
    "statut_marital",
    "departement",
    "poste",
    "niveau_education",
    "domaine_etude",
    "frequence_deplacement",
]

# *****************************************
# *          TARGET ET FEATURES           *
# *****************************************

col_sel = num_features + cat_features

X = df[col_sel].copy()
y = df["a_quitte_l_entreprise"]

# *****************************************
# *            SPLIT & PROC               *
# *****************************************

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features),
])

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("scale_pos_weight =", scale_pos_weight)

# *****************************************
# *         Parte 1 :: modele de base     *
# *****************************************

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", XGBClassifier(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=5.0,
        gamma=0.5,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])
# Fit
pipe.fit(X_train, y_train)

# Predict
y_pred_test = pipe.predict(X_test)
y_pred_train = pipe.predict(X_train)

#Matrice de confusion
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

# Metrics
print("====== TRAIN ======")
print(confusion_matrix_train)
print(classification_report(y_train, y_pred_train))

print("====== TEST ======")
print(confusion_matrix_test)
print(classification_report(y_test, y_pred_test))

print("ROC AUC:", roc_auc_score(y_test, y_proba))

#? ====== RESULTATS TEST ======
#?              precision    recall  f1-score   support
#?
#?           0       0.90      0.78      0.84       247
#?           1       0.32      0.53      0.40        47
#?
#?    accuracy                           0.74       294
#?   macro avg       0.61      0.66      0.62       294
#?weighted avg       0.80      0.74      0.77       294
#? ROC AUC: 0.7152209492635024

#? Visualisation

# Répartition de la proba de la classe 1 

y_proba = pipe.predict_proba(X_test)[:, 1] # proba de la classe 1 (à quitter l'entreprise)
sns.histplot(y_proba)

# ------ Probability Score Analysis ------
X_train["prediction"] = y_pred_train
X_train["probability_score"] = pipe.predict_proba(
    X_train[col_sel]
)[:, 1]

X_test["prediction"] = y_pred_test
X_test["probability_score"] = pipe.predict_proba(
    X_test[col_sel]
)[:, 1]

X_train["a_quitte_l_entreprise"] = y_train 
X_train["prediction_type"] = X_train.apply(
    lambda row: dpt.get_prediction_type(row["prediction"], row["a_quitte_l_entreprise"]),
    axis=1,
)

X_test["a_quitte_l_entreprise"] = y_test
X_test["prediction_type"] = X_test.apply(
    lambda row: dpt.get_prediction_type(row["prediction"], row["a_quitte_l_entreprise"]),
    axis=1,
)

color_palette = {
    "true_positive" : "green",
    "true_negative": "red",
    "false_positive": "blue",
    "false_negative": "orange"
}


dpt.plot_probability_disrtib_per_pred_type(X_test,categories_to_exclude=["true_negative","false_positive"])

dpt.plot_probability_disrtib_per_pred_type(X_test,categories_to_exclude=["true_positive","false_negative"])

dpt.plot_probability_disrtib_per_pred_type(X_train,categories_to_exclude=["true_negative","false_positive"])

# ******************************************
# *      Parte 2 :: Validation croisée     *
# ******************************************

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = cross_validate(
    pipe,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

for metric in scoring:
    scores = cv_results[f"test_{metric}"]
    print(f"{metric:<10} : mean={scores.mean():.3f} | std={scores.std():.3f}")

#? ==== Résultats CV test 

#? precision  : mean=0.320 | std=0.035
#? recall     : mean=0.586 | std=0.066
#? f1         : mean=0.414 | std=0.045
#? roc_auc    : mean=0.760 | std=0.027

# ******************************************
# *        Parte 3 :: test de seuil        *
# ******************************************

y_proba_test = pipe.predict_proba(X_test)[:, 1]

for threshold in [0.5, 0.6]:
    y_pred_thr = (y_proba_test >= threshold).astype(int)
    print(f"\n===== threshold = {threshold} =====")
    print(classification_report(y_test, y_pred_thr))

# ******************************************
# *      Parte 3 :: Randomized Search      *
# ******************************************

param_dist_xgb = {
    "model__n_estimators": [150, 250, 400, 600],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.03, 0.05, 0.08],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
    "model__min_child_weight": [3, 5, 8, 10],
    "model__gamma": [0, 0.3, 0.5, 1.0],
    "model__reg_alpha": [0, 0.5, 1.0, 3.0],
    "model__reg_lambda": [1.0, 3.0, 5.0, 10.0],
    "model__scale_pos_weight": [
        scale_pos_weight * 0.8,
        scale_pos_weight,
        scale_pos_weight * 1.2,
    ],
}

random_search_xgb = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist_xgb,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

random_search_xgb.fit(X_train, y_train)

print("Best params:", random_search_xgb.best_params_)
print("Best CV score:", random_search_xgb.best_score_)

best_model = random_search_xgb.best_estimator_

y_pred_test = best_model.predict(X_test)
y_proba_test = best_model.predict_proba(X_test)[:, 1]

print("====== TEST BEST XGB ======")
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
print("ROC AUC:", roc_auc_score(y_test, y_proba_test))

#? ====== TEST BEST XGB ======
#? [[204  43]
#? [ 28  19]]
#?                precision    recall  f1-score   support
#? 
#?            0       0.88      0.83      0.85       247
#?            1       0.31      0.40      0.35        47
#? 
#?     accuracy                           0.76       294
#?    macro avg       0.59      0.62      0.60       294
#? weighted avg       0.79      0.76      0.77       294
#? ROC AUC: 0.7062623826341631