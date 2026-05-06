import pickle
import pandas as pd
import kagglehub
import os
from emails import SPAM_EMAILS, LEGITIMATE_EMAILS

# Load model
with open("modell.pkl", "rb") as f:
    model = pickle.load(f)

# Load column names (same as training)
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

# Function to test a mail
def test_email(text, threshold=0.8):
    # Count how many times every word is caught in the text
    words_in_email = text.lower().split()
    row = {col: words_in_email.count(col) for col in columns}

    mail_df     = pd.DataFrame([row])
    probability = model.predict_proba(mail_df)[0]
    prediction  = 1 if probability[1] >= threshold else 0

    return {
        'result':     'SPAM' if prediction == 1 else 'LEGITIMATE',
        'spam_score': probability[1] * 100,
        'correct':    None
    }

# Run tests and print summary
def run_tests(emails, expected_label):
    correct = 0
    for email in emails:
        result = test_email(email)
        if result['result'] == expected_label:
            correct += 1

    print(f"  Correct:      {correct}/{len(emails)}")
    print(f"  Wrong:        {len(emails) - correct}/{len(emails)}")
    print(f"  Accuracy:     {correct/len(emails)*100:.0f}%")

print("=" * 50)
print("SPAM EMAIL RESULTS")
print("=" * 50)
run_tests(SPAM_EMAILS, "SPAM")

print("\n" + "=" * 50)
print("LEGITIMATE EMAIL RESULTS")
print("=" * 50)
run_tests(LEGITIMATE_EMAILS, "LEGITIMATE")