import pandas as pd
import numpy as np
import kagglehub
import os

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

# Hjälpfunktion för att printa ord-listor
def print_ord(serie):
    for ord, freq in zip(serie.index.tolist(), serie.values.tolist()):
        print(f"  {str(ord)} - {str(int(freq) if freq > 1 else round(freq, 1))}")


print("=" * 50)
print("DATASET ÖVERSIKT")
print("=" * 50)
print(f"  Totalt antal mail:   {len(df)}")
print(f"  Antal ord-features:  {X.shape[1]}")
print(f"  Saknade värden:      {df.isnull().sum().sum()}")
print(f"  Duplicerade rader:   {df.duplicated().sum()}")


print("\n" + "=" * 50)
print("KLASSFÖRDELNING")
print("=" * 50)
print(f"  Äkta mail:   {(y==0).sum()}  ({(y==0).mean()*100:.1f}%)")
print(f"  Skräppost:   {(y==1).sum()}  ({(y==1).mean()*100:.1f}%)")


print("\n" + "=" * 50)
print("HUR MÅNGA ORD HAR MAILEN I SNITT?")
print("=" * 50)
print(f"  Äkta mail:   {df[df['label']==0]['total_words'].mean():.0f} ord")
print(f"  Skräppost:   {df[df['label']==1]['total_words'].mean():.0f} ord")


print("\n" + "=" * 50)
print("TOPP 15 VANLIGASTE ORD I SPAM")
print("=" * 50)
print_ord(X[y==1].sum().sort_values(ascending=False).head(15))

print("\n" + "=" * 50)
print("TOPP 15 VANLIGASTE ORD I ÄKTA MAIL")
print("=" * 50)
print_ord(X[y==0].sum().sort_values(ascending=False).head(15))


spam_freq = X[y==1].mean()
ham_freq  = X[y==0].mean()

print("\n" + "=" * 50)
print("ORD SOM STARKAST INDIKERAR SPAM")
print("=" * 50)
print_ord((spam_freq / (ham_freq + 0.0001)).sort_values(ascending=False).head(15))

print("\n" + "=" * 50)
print("ORD SOM STARKAST INDIKERAR ÄKTA MAIL")
print("=" * 50)
print_ord((ham_freq / (spam_freq + 0.0001)).sort_values(ascending=False).head(15))