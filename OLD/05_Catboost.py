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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier
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
    "satisfaction_employee_environnement",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "satisfaction_global",
    "note_evaluation_precedente",
    "note_evaluation_actuelle",
#    "delta_evaluation",
    "salary_vs_level",
 #   "salary_vs_tenure",
    "promotion_speed",
    "experience_mismatch",
    "salary_gap_level",
    "career_stagnation",
    "niveau_hierarchique_poste",
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

from collections import Counter

counter = Counter(y_train)
counter

neg = 986
pos = 190

class_weights = [1, neg / pos]
class_weights

# *****************************************
# *         Parte 1 :: modele de base     *
# *****************************************

cat_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        class_weights=class_weights,
        random_seed=42,
        verbose=0
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

#? =========================
#? Evaluation at 0.5
#? =========================
y_proba = cat_model.predict_proba(X_test)[:, 1]
y_pred = cat_model.predict(X_test) #(y_proba >= 0.5).astype(int)

print("====== CATBOOST - THRESHOLD 0.5 ======")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

#? ====== CATBOOST - THRESHOLD 0.5 ======
#? [[233  14]
#?  [ 28  19]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.90      0.92      0.91       247
#?            1       0.53      0.45      0.48        47

#?     accuracy                           0.85       294
#?    macro avg       0.71      0.68      0.70       294
#? weighted avg       0.84      0.84      0.84       294
#? 
#? ROC AUC: 0.8311654750624515


seeds = [0, 21, 42, 51, 99]

for s in seeds:
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        class_weights=class_weights,
        random_seed=s,
        verbose=0
    )
    
    model.fit(X_train, y_train, cat_features=cat_features)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(s, roc_auc_score(y_test, y_proba))

#? =========================
#?  Threshold tuning
#? =========================
best_thresh = 0
best_score = 0

for t in np.linspace(0.1, 0.9, 50):
    y_pred_t = (y_proba >= t).astype(int)
    score = f1_score(y_test, y_pred_t)

    if score > best_score:
        best_score = score
        best_thresh = t

print("\nBest threshold:", best_thresh)
print("Best F1:", best_score)

y_pred_best = (y_proba >= best_thresh).astype(int)

print(f"\n====== CATBOOST - THRESHOLD {best_thresh:.2f} ======")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
print("ROC AUC:", roc_auc_score(y_test, y_proba))


#? Best threshold: 0.49183673469387756
#? Best F1: 0.5319148936170213
#? 
#? ====== CATBOOST - THRESHOLD 0.33 ======
#? [[169  78]
#?  [ 13  34]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.93      0.84      0.89       247
#?            1       0.45      0.68      0.54        47
#? 
#?     accuracy                           0.82       294
#?    macro avg       0.69      0.76      0.71       294
#? weighted avg       0.86      0.82      0.83       294
#?
#? ROC AUC: 0.8405547420105091
#? precision_recall_curve :
#? =========================
#?  Probability Score Analysis
#? =========================

y_proba_test = cat_model.predict_proba(X_test)[:, 1]
y_proba_train = cat_model.predict_proba(X_train)[:, 1]

y_pred_test = (y_proba_test >= best_thresh).astype(int)
y_pred_train = (y_proba_train >= best_thresh).astype(int)

X_train["prediction"] = y_pred_train
X_train["probability_score"] = y_proba_train

X_test["prediction"] = y_pred_test
X_test["probability_score"] = y_proba_test

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

dpt.plot_probability_disrtib_per_pred_type(X_test)

sns.histplot(y_proba_test)


# *****************************************
# *       Parte 2 :: Early Stop           *
# *****************************************

cat_model = CatBoostClassifier(
    iterations=2000,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=class_weights,
    random_seed=42,
    verbose=100
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=100
)


#? =========================
#? Evaluation at 0.5
#? =========================
y_proba = cat_model.predict_proba(X_test)[:, 1]
y_pred = cat_model.predict(X_test) #(y_proba >= 0.5).astype(int)

print("====== CATBOOST - THRESHOLD 0.5 ======")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

#? ====== CATBOOST - THRESHOLD 0.5 ======
#? [[211  36]
#?  [ 14  33]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.94      0.85      0.89       247
#?            1       0.48      0.70      0.57        47
#? 
#?     accuracy                           0.83       294
#?    macro avg       0.71      0.78      0.73       294
#? weighted avg       0.86      0.83      0.84       294
#? 
#? ROC AUC: 0.8473598070462572

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

print(f"\n====== CATBOOST - THRESHOLD {best_thresh:.2f} ======")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
print("ROC AUC:", roc_auc_score(y_test, y_proba))


#? ====== CATBOOST - THRESHOLD 0.49 ======
#? [[207  40]
#?  [ 13  34]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.94      0.84      0.89       247
#?            1       0.46      0.72      0.56        47
#? 
#?     accuracy                           0.82       29
#?    macro avg       0.70      0.78      0.72       294
#? weighted avg       0.86      0.82      0.83       294
#? 
#? ROC AUC: 0.8473598070462572

# *************************************************************
# *    Parte 3 :: Analyse des faux négatifs vs true positif   *
# *************************************************************

#? Overview 

X_test["prediction"] = y_pred_test
X_test["probability_score"] = y_proba_test

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

#? Vision par modalités numériques 

results = X_test.copy()
results["y_true"] = y_test.values
results["y_proba"] = y_proba
results["y_pred"] = y_pred_best

# Faux négatifs
fn = results[(results["y_true"] == 1) & (results["y_pred"] == 0)]
fn.shape

# Faux négatifs
tp = results[(results["y_true"] == 1) & (results["y_pred"] == 1)]
tp.shape

tn = results[(results["y_true"] == 0) & (results["y_pred"] == 0)]
tn.shape

fn[num_features].mean()
tp[num_features].mean()

for col in num_features:
    plt.figure(figsize=(6,3))
    sns.kdeplot(fn[col], label="FN", fill=True, color="blue")
    sns.kdeplot(tp[col], label="TP", fill=True, color="green")
    sns.kdeplot(tn[col], label="TN", fill=True, color="red")
    plt.title(col)
    plt.legend()
    plt.show()




# ********************************************
# *    Parte 4 :: Nouvelles features model   *
# ********************************************

num_features = [
    "age",
    "revenu_mensuel",
#    "annee_experience_totale",
#    "annees_dans_l_entreprise",
#    "annees_dans_le_poste_actuel",
    "annees_depuis_la_derniere_promotion",
#    "distance_domicile_travail",
    "satisfaction_employee_environnement",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "satisfaction_global",
    "note_evaluation_precedente",
    "note_evaluation_actuelle",
#    "delta_evaluation",
    "salary_vs_level",
#   "salary_vs_tenure",
    "promotion_speed",
    "experience_mismatch",
    "salary_gap_level",
    "career_stagnation",
    "niveau_hierarchique_poste",
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

col_sel = num_features + cat_features

X = df[col_sel].copy()
y = df["a_quitte_l_entreprise"]

X = X.copy()
X.columns = X.columns.astype(str)
num_features = [c for c in map(str, num_features) if c in X.columns]
cat_features = [c for c in map(str, cat_features) if c in X.columns]
X.info()

seed = 51

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

from collections import Counter

counter = Counter(y_train)
counter

neg = 986
pos = 190

class_weights = [1, neg / pos]
class_weights

cat_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        class_weights=class_weights,
        random_seed=42,
        verbose=0
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

#? =========================
#? Evaluation at 0.5
#? =========================
y_proba = cat_model.predict_proba(X_test)[:, 1]
y_pred = cat_model.predict(X_test) #(y_proba >= 0.5).astype(int)

print("====== CATBOOST - THRESHOLD 0.5 ======")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))