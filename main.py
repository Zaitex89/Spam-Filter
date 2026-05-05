import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split

path = kagglehub.dataset_download("balaka18/email-spam-classification-dataset-csv")
df = pd.read_csv(os.path.join(path, "emails.csv"))
df = df.rename(columns={'Prediction': 'label'})

X = df.drop(columns=['Email No.', 'label'])
y = df['label']

df['total_words'] = X.sum(axis=1)
df['unika_ord']   = (X > 0).sum(axis=1)

selector = VarianceThreshold(threshold=0.01)  # justera vid behov
X_reduced = selector.fit_transform(X)

kept_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_reduced, columns=kept_features)

print(f"Features före: {len(X.columns)+sum(~selector.get_support())}")
print(f"Features efter: {len(X.columns)}")

# MaxAbsScaler bevarar glesheten (sparar 0-värden)
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)


# 1. GRUNDLÄGGANDE INFO
print("=" * 50)
print("DATASET ÖVERSIKT")
print("=" * 50)
print(f"Totalt antal mail:     {len(df)}")
print(f"Antal ord-features:    {X.shape[1]}")
print(f"Saknade värden:        {df.isnull().sum().sum()}")
print(f"Duplicerade rader:     {df.duplicated().sum()}")
print(y.value_counts())
print(y.value_counts(normalize=True).mul(100).round(1))