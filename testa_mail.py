import pickle
import pandas as pd
import kagglehub
import os

# Load model
with open("modell.pkl", "rb") as f:
    modell = pickle.load(f)

# Load column name (same as training)
path = kagglehub.dataset_download("balaka18/email-spam-classification-dataset-csv")
df   = pd.read_csv(os.path.join(path, "emails.csv"))
df   = df.rename(columns={'Prediction': 'label'})

X = df.drop(columns=['Email No.', 'label'])
X = X[[col for col in X.columns if len(col) >= 4]]

stoppord = [
    'the', 'and', 'for', 'you', 'our', 'his', 'her', 'are', 'com',
    'ect', 'hou', 'this', 'that', 'with', 'have', 'will', 'your',
    'from', 'they', 'been', 'not', 'all', 'pro', 'men', 'est',
    'act', 'rom', 'was', 'but', 'has', 'can', 'its', 'one', 'who',
    'may', 'any', 'out', 'get', 'new', 'see', 'had', 'did', 'say'
]
X = X[[col for col in X.columns if col not in stoppord]]
kolumner = X.columns.tolist()

# Functions to test a mail
def testa_mail(text):
    # Count how many times every word is caught in the text
    ord_i_mail = text.lower().split()
    rad = {}
    for col in kolumner:
        rad[col] = ord_i_mail.count(col)

    mail_df = pd.DataFrame([rad])
    prediction = modell.predict(mail_df)[0]
    sannolikhet = modell.predict_proba(mail_df)[0]

    print("\n" + "=" * 50)
    print("MAILANALYS")
    print("=" * 50)
    print(f"  Text:        {text[:60]}...")
    print(f"  Resultat:    {'SKRÄPPOST' if prediction == 1 else 'ÄKTA MAIL'}")
    print(f"  Säkerhet:    {max(sannolikhet)*100:.1f}%")

# Test own emails
testa_mail("Free money! Click here now to claim your prize and win cash!")
testa_mail("Hi, can we schedule a meeting tomorrow to discuss the deal?")
testa_mail("Congratulations you have won a free iPhone click here now")
testa_mail("Please find attached the report from the Enron meeting today")