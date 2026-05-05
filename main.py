import pandas as pd
import kagglehub
import os
from eda import run_eda
from ml import run_ml

# Ladda data
path = kagglehub.dataset_download("balaka18/email-spam-classification-dataset-csv")
df   = pd.read_csv(os.path.join(path, "emails.csv"))
df   = df.rename(columns={'Prediction': 'label'})

X = df.drop(columns=['Email No.', 'label'])
y = df['label']

# Rensa features
X = X[[col for col in X.columns if len(col) >= 4]]

stoppord = [
    'the', 'and', 'for', 'you', 'our', 'his', 'her', 'are', 'com',
    'ect', 'hou', 'this', 'that', 'with', 'have', 'will', 'your',
    'from', 'they', 'been', 'not', 'all', 'pro', 'men', 'est',
    'act', 'rom', 'was', 'but', 'has', 'can', 'its', 'one', 'who',
    'may', 'any', 'out', 'get', 'new', 'see', 'had', 'did', 'say'
]
X = X[[col for col in X.columns if col not in stoppord]]

df['total_words'] = X.sum(axis=1)

# Kör EDA
run_eda(X, y, df)

# Kör ML
run_ml(X, y)