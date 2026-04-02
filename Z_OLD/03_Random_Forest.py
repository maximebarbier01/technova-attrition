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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# =========================
# Local imports (src/)
# =========================
sys.path.append(os.path.abspath(".."))  # si notebook dans /notebooks

import src.utils.outliers as of
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

df.shape

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

# *****************************************
# *         Parte 1 :: modele de base     *
# *****************************************

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
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


dpt.plot_probability_disrtib_per_pred_type(X_test,categories_to_exclude=["true_negative","false_positive"])

dpt.plot_probability_disrtib_per_pred_type(X_test,categories_to_exclude=["true_positive","false_negative"])

dpt.plot_probability_disrtib_per_pred_type(X_train,categories_to_exclude=["true_negative","false_positive"])

# ******************************************
# *    Parte 2 :: Model sans overfitting   *
# ******************************************

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    ))
])
# Fit
pipe.fit(X_train, y_train)

# Predict
y_pred_test = pipe.predict(X_test)
y_pred_train = pipe.predict(X_train)
y_proba_test = pipe.predict_proba(X_test)[:, 1] # proba de la classe 1 (à quitter l'entreprise)
y_proba_train = pipe.predict_proba(X_train)[:, 1] # proba de la classe 1 (à quitter l'entreprise)

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
#? [[227  20]
#?  [ 26  21]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.90      0.92      0.91       247
#?            1       0.51      0.45      0.48        47
#? 
#?     accuracy                           0.84       294
#?    macro avg       0.70      0.68      0.69       294
#? weighted avg       0.84      0.84      0.84       294

#? ROC AUC: 0.8255233008872426

#? ------ Changement de threshold ------

threshold = 0.36  # test
print(f"====== TEST SEUIL à {threshold} ======")
y_pred_test = (y_proba_test >= threshold).astype(int)
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
print("ROC AUC:", roc_auc_score(y_test, y_pred_test))

#? ====== TEST SEUIL à 0.36 ======
#? [[186  61]
#?  [  7  40]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.96      0.75      0.85       247
#?            1       0.40      0.85      0.54        47
#? 
#?     accuracy                           0.77       294
#?    macro avg       0.68      0.80      0.69       294
#? weighted avg       0.87      0.77      0.80       294
#? 
#? ROC AUC: 0.8020501335170988

y_pred_train = (y_proba_train >= threshold).astype(int)
print(confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))


#? ------ Calculer la permutation importance ------

perm = permutation_importance(
    estimator=pipe,
    X=X_test,
    y=y_test,
    n_repeats=10,
    random_state=seed,
    scoring="roc_auc",
    n_jobs=-1
)

perm_importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)

perm_importance.head(20)

top_n = 20
plot_df = perm_importance.head(top_n).sort_values("importance_mean")

plt.figure(figsize=(10, 12))
plt.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_std"])
plt.title(f"Top {top_n} - Permutation Importance")
plt.xlabel("Baisse moyenne de performance (ROC AUC)")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()


#? ------ Modèle avec top features ------

top_features = perm_importance.loc[
    perm_importance["importance_mean"] > 0
, "feature"].tolist()

top_features[:10]

X_top = X[top_features]

num_features_top = [col for col in num_features if col in top_features]
cat_features_top = [col for col in cat_features if col in top_features]

# Full model
y_pred_test = pipe.predict(X_test)
y_proba_test = pipe.predict_proba(X_test)[:, 1]

y_pred_train = pipe.predict(X_train)
y_proba_train = pipe.predict_proba(X_train)[:, 1]

print("FULL MODEL")
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_full))
print("ROC AUC:", roc_auc_score(y_test, y_proba_full))

#? FULL MODEL
#? [[227  20]
#?  [ 26  21]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.84      0.89      0.86       247
#?            1       0.18      0.13      0.15        47
#? 
#?     accuracy                           0.77       294
#?    macro avg       0.51      0.51      0.51       294
#? weighted avg       0.74      0.77      0.75       294
#? 
#? ROC AUC: 0.5186493238004994

#? ------ Changement de threshold ------

best_thresh = 0
best_score = 0

for t in np.linspace(0.1, 0.9, 50):
    y_pred = (y_proba >= t).astype(int)
    score = f1_score(y_test, y_pred)
    
    if score > best_score:
        best_score = score
        best_thresh = t

best_thresh, best_score

threshold = best_thresh  # test

y_pred_test = (y_proba_test >= threshold).astype(int)
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
print("ROC AUC:", roc_auc_score(y_test, y_pred_test))

y_pred_train = (y_proba_train >= threshold).astype(int)
print(confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))

#? ====== FULL MODEL - THRESHOLD 0.36 ======
#? [[204  43]
#?  [ 12  35]]
#?               precision    recall  f1-score   support
#? 
#?            0       0.94      0.83      0.88       247
#?            1       0.45      0.74      0.56        47
#? 
#?     accuracy                           0.81       294
#?    macro avg       0.70      0.79      0.72       294
#? weighted avg       0.87      0.81      0.83       294

#? ROC AUC: 0.7201309328968903

#? ------ Probability Score Analysis ------
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


dpt.plot_probability_disrtib_per_pred_type(X_test,categories_to_exclude=["true_negative","false_positive"])

dpt.plot_probability_disrtib_per_pred_type(X_test,categories_to_exclude=["true_positive","false_negative"])

sns.histplot(y_proba_train)