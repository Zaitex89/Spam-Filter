import pandas as pd
import numpy as np
import kagglehub
import os

path = kagglehub.dataset_download("balaka18/email-spam-classification-dataset-csv")
df = pd.read_csv(os.path.join(path, "emails.csv"))
df = df.rename(columns={'Prediction': 'label'})

X = df.drop(columns=['Email No.', 'label'])
y = df['label']

df['total_words'] = X.sum(axis=1)
df['unika_ord']   = (X > 0).sum(axis=1)

# 1. GRUNDLÄGGANDE INFO
print("=" * 50)
print("DATASET ÖVERSIKT")
print("=" * 50)
print(f"Totalt antal mail:     {len(df)}")
print(f"Antal ord-features:    {X.shape[1]}")
print(f"Saknade värden:        {df.isnull().sum().sum()}")
print(f"Duplicerade rader:     {df.duplicated().sum()}")
