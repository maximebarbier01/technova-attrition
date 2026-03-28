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


# =========================
# Local imports (src/)
# =========================
sys.path.append(os.path.abspath(".."))  # si notebook dans /notebooks

import src.outliers as of
import src.distrib_pred_type as dpt

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

df.frequence_deplacement

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

X = X.copy()
X.columns = X.columns.astype(str)
num_features = [c for c in map(str, num_features) if c in X.columns]
cat_features = [c for c in map(str, cat_features) if c in X.columns]
X.info()

seed = 51

# *****************************************
# *            SPLIT & PROC               *
# *****************************************

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# Num pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Cat pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Global preprocessor
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# ! *******************************************
# !    Parte 0 :: Lasso - select feature      *
# ! *******************************************

lasso_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=seed
    ))
])

from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 5, 10]
}

grid_lasso = GridSearchCV(
    lasso_pipeline,
    param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    refit=True
)

grid_lasso.fit(X_train, y_train)

print("Best params:", grid_lasso.best_params_)
print("Best CV:", grid_lasso.best_score_)

#? Résultats
#? Best params: {'model__C': 10}
#? Best CV: 0.7693236128314942

best_lasso = grid_lasso.best_estimator_

y_pred = best_lasso.predict(X_test)
y_proba = best_lasso.predict_proba(X_test)[:, 1]

feature_names = best_lasso.named_steps["prep"].get_feature_names_out()
coefficients = best_lasso.named_steps["model"].coef_[0]

print("====== TEST ======")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("ROC AUC:", roc_auc_score(y_test, y_proba))

#? ====== RESULTATS TEST ======
#? [[163  84]
#?  [  9  38]]
#?              precision    recall  f1-score   support
#?
#?           0       0.95      0.66      0.78       247
#?           1       0.31      0.81      0.45        47
#?
#?    accuracy                           0.68       294
#?   macro avg       0.63      0.73      0.61       294
#?weighted avg       0.85      0.68      0.73       294
#?
#? ROC AUC: 0.8048927556206391


best_thresh = 0
best_f1 = 0

for t in np.linspace(0.1, 0.9, 50):
    y_pred_t = (y_proba >= t).astype(int)
    score = f1_score(y_test, y_pred_t)
    
    if score > best_f1:
        best_f1 = score
        best_thresh = t

best_thresh, best_f1

y_pred_best = (y_proba >= best_thresh).astype(int)

print(f"\n====== REG LOG - THRESHOLD {best_thresh:.2f} ======")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

#? ====== REG LOG - THRESHOLD 0.70 ======
#? [[231  16]
#?  [ 27  20]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.90      0.94      0.91       247
#?            1       0.56      0.43      0.48        47
#? 
#?     accuracy                           0.85       294
#?    macro avg       0.73      0.68      0.70       294
#? weighted avg       0.84      0.85      0.85       294
#?
#? ROC AUC: 0.8048927556206391

#? VISUALISATION 

y_pred_train = best_lasso.predict(X_train)

y_pred_test = best_lasso.predict(X_test)

# ------ Probability Score Analysis ------
X_train["prediction"] = y_pred_train
X_train["probability_score"] = best_lasso.predict_proba(
    X_train[col_sel]
)[:, 1]

X_test["prediction"] = y_pred_test
X_test["probability_score"] = best_lasso.predict_proba(
    X_test[col_sel])[
        :, 1
]

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

dpt.plot_probability_disrtib_per_pred_type(X_test,categories_to_exclude=["true_positive","false_negative"])

dpt.plot_probability_disrtib_per_pred_type(X_train,categories_to_exclude=["true_negative","false_positive"])


# ! *****************************************
# ! *         Parte 1 :: modele de base     *
# ! *****************************************


# *****************************************
# *        SELECTION DES FEATURES         *
# *****************************************

num_features = [
    "revenu_mensuel",
    "niveau_hierarchique_poste",
    "salary_vs_level",
    "annees_dans_l_entreprise",
    "satisfaction_global",
    "annees_depuis_la_derniere_promotion",
    "annees_dans_le_poste_actuel"

]

cat_features = [
    "poste",
    "departement",
    "domaine_etude",
    "frequence_deplacement",
    "statut_marital",
    "niveau_education",
]

# *****************************************
# *          TARGET ET FEATURES           *
# *****************************************

col_sel = num_features + cat_features

X = df[col_sel].copy()
y = df["a_quitte_l_entreprise"]

X = X.copy()
X.columns = X.columns.astype(str)
num_features = [c for c in map(str, num_features) if c in X.columns]
cat_features = [c for c in map(str, cat_features) if c in X.columns]
X.info()

# *****************************************
# *            SPLIT & PROC               *
# *****************************************

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Num pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Cat pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Global preprocessor
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])


pipe = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])
# Fit
pipe.fit(X_train, y_train)

# Predict
y_pred_test = pipe.predict(X_test)
y_pred_train = pipe.predict(X_train)
y_proba = pipe.predict_proba(X_test)[:, 1] # proba de la classe 1 (à quitter l'entreprise)

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

#? ====== TEST ======
#? [[182  65]
#?  [ 17  30]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.91      0.74      0.82       247
#?            1       0.32      0.64      0.42        47
#? 
#?     accuracy                           0.72       294
#?    macro avg       0.62      0.69      0.62       294
#? weighted avg       0.82      0.72      0.75       294

#? ROC AUC: 0.7173744508570936

# ------ Probability Score Analysis ------
X_train["prediction"] = y_pred_train
X_train["probability_score"] = pipe.predict_proba(
    X_train[col_sel]
)[:, 1]

X_test["prediction"] = y_pred_test
X_test["probability_score"] = pipe.predict_proba(
    X_test[col_sel])[
        :, 1
]

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

dpt.plot_probability_disrtib_per_pred_type(X_test,categories_to_exclude=["true_positive","false_negative"])

dpt.plot_probability_disrtib_per_pred_type(X_train,categories_to_exclude=["true_negative","false_positive"])
