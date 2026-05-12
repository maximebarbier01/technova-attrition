from __future__ import annotations

import pandas as pd

df = pd.read_csv("/home/maxime/projects/technova-attrition/data/interim/data_eda.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(1).T)