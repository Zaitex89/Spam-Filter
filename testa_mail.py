import pickle
import pandas as pd
import kagglehub
import os

# Load model
with open("modell.pkl", "rb") as f:
    model = pickle.load(f)

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
columns = X.columns.tolist()

# Functions to test a mail
def test_email(text):
    # Count how many times every word is caught in the text
    words_in_email = text.lower().split()
    row = {}
    for col in columns:
        row[col] = words_in_email.count(col)

    mail_df = pd.DataFrame([row])
    prediction = model.predict(mail_df)[0]
    probability = model.predict_proba(mail_df)[0]

    print("\n" + "=" * 50)
    print("EMAIL ANALYSIS")
    print("=" * 50)
    print(f"  Text:        {text[:60]}...")
    print(f"  Result:      {'SPAM' if prediction == 1 else 'LEGITIMATE'}")
    print(f"  Confidence:  {max(probability)*100:.1f}%")

# Test own emails
test_email("Free money! Click here now to claim your prize and win cash!")
test_email("Hi, can we schedule a meeting tomorrow to discuss the deal?")
test_email("Congratulations you have won a free iPhone click here now")
test_email("Please find attached the report from the Enron meeting today")