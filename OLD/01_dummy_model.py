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
from sklearn.dummy import DummyClassifier
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
    "revenu_mensuel",  # ou log_revenu (pas les 2)
    "nombre_experiences_precedentes",
    "annee_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "annees_depuis_la_derniere_promotion",
    "distance_domicile_travail",
    "augementation_salaire_precedente",
    "delta_evaluation",
    "evaluation_declined",
    "satisfaction_global"
]

cat_features = [
    "genre",
    "statut_marital",
    "departement",
    "poste",
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

# *****************************************
# *         Parte 1 :: modele de base     *
# *****************************************
pipeline = ImbPipeline([
    ("prep", preprocessor),
   # ("smote", SMOTE(random_state=42)),
    ("model", DummyClassifier(strategy="most_frequent"))
])

# Fit
pipeline.fit(X_train, y_train)

# Predict
y_pred_test = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)
y_proba = pipeline.predict_proba(X_test)[:, 1] # proba de la classe 1 (à quitter l'entreprise)

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
X_train["probability_score"] = pipeline.predict_proba(
    X_train[col_sel]
)[:, 1]

X_test["prediction"] = y_pred_test
X_test["probability_score"] = pipeline.predict_proba(
    X_test[col_sel])[
        :, 1
]

def get_prediction_type(prediction, target):
    if prediction == 1 and target == 1:
        return "true_positive"
    elif prediction == 0 and target == 0:
        return "true_negative"
    elif prediction == 1 and target == 0:
        return "false_positive"
    elif prediction == 0 and target == 1:
        return "false_negative"
    else:
        return "unknow"

X_train["a_quitte_l_entreprise"] = y_train 
X_train["prediction_type"] = X_train.apply(
    lambda row: get_prediction_type(row["prediction"], row["a_quitte_l_entreprise"]),
    axis=1,
)

X_test["a_quitte_l_entreprise"] = y_test
X_test["prediction_type"] = X_test.apply(
    lambda row: get_prediction_type(row["prediction"], row["a_quitte_l_entreprise"]),
    axis=1,
)

color_palette = {
    "true_positive" : "green",
    "true_negative": "red",
    "false_positive": "blue",
    "false_negative": "orange"
}

def plot_probability_disrtib_per_pred_type(
        X,
        color_palette=color_palette,
        categories_to_exclude=["true_negative"]
    ):
    X_filtered = X[~X["prediction_type"].isin(categories_to_exclude)]
    sns.displot(data=X_filtered, x="probability_score", hue="prediction_type",palette=color_palette)
    plt.legend(loc="upper right")
    plt.show()

plot_probability_disrtib_per_pred_type(X_test)

plot_probability_disrtib_per_pred_type(X_train)