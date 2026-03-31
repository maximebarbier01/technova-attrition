# =========================
# Standard library
# =========================
import os
import sys
from pathlib import Path
import time
import math
import re
import importlib

# =========================
# Data / scientific stack
# =========================
import numpy as np
import pandas as pd

# =========================
# Visualisation
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# =========================
# Stats
# =========================
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import pingouin as pg
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import (
    LabelEncoder,
    MultiLabelBinarizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

# =========================
# Local imports (src/)
# =========================
PROJECT_ROOT = Path.cwd().resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print(PROJECT_ROOT)

sys.path.append(os.path.abspath(".."))  # si notebook dans /notebooks

import src.utils.outliers as of
import src.distrib_pred_type as dpt

df_back_up = pd.read_csv("/home/maxime/projects/technova-attrition/data/interim/data_eda.csv")
df_back_up.head(2)

df = df_back_up.copy()
df.shape
df.drop("Unnamed: 0",axis=1,inplace=True)

