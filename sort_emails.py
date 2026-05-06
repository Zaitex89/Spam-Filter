import pickle
import pandas as pd
import kagglehub
import os
import emails as emails_module
from emails import INBOX

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

# Save email to emails.py
def save_to_inbox(text):
    emails_module.INBOX.append(text)
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emails.py")
    with open(filepath, "w") as f:
        f.write("INBOX = [\n")
        for email in emails_module.INBOX:
            clean = email.replace('"', '\\"')
            f.write(f'    "{clean}",\n')
        f.write("]\n")
    print(f"Saved! emails.py now has {len(emails_module.INBOX)} emails")
    print(f"File location: {filepath}")

# Sort emails
def sort_inbox(emails, threshold=0.8):
    spam       = []
    legitimate = []

    for email in emails:
        words       = email.lower().split()
        row         = {col: words.count(col) for col in columns}
        mail_df     = pd.DataFrame([row])
        probability = model.predict_proba(mail_df)[0]
        prediction  = 1 if probability[1] >= threshold else 0

        if prediction == 1:
            spam.append((email, probability[1] * 100))
        else:
            legitimate.append((email, probability[1] * 100))

    return spam, legitimate

spam, legitimate = sort_inbox(INBOX)

print("=" * 50)
print(f"INBOX SORTED — {len(INBOX)} emails")
print("=" * 50)

print(f"\n LEGITIMATE ({len(legitimate)})")
print("-" * 50)
for email, score in legitimate:
    print(f"  [{score:.0f}% spam score]  {email[:55]}...")

print(f"\n SPAM ({len(spam)})")
print("-" * 50)
for email, score in spam:
    print(f"  [{score:.0f}% spam score]  {email[:55]}...")

def test_email_interactive(threshold=0.8):
    print("\n" + "=" * 75)
    print("INTERACTIVE SPAM FILTER")
    print("=" * 75)
    print("Type an email below and press Enter to analyze it.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            text = input("Your email: ").strip()

            if text.lower() in ['exit', 'quit', 'q']:
                print("Exiting spam filter. Have a great day!")
                break

            if not text:
                print("Please write an email...")
                continue

            # Prepare bag-of-words
            words       = text.lower().split()
            row         = {col: words.count(col) for col in columns}
            mail_df     = pd.DataFrame([row])
            probability = model.predict_proba(mail_df)[0][1]
            prediction  = 1 if probability >= threshold else 0

            # Show result
            print("-" * 70)
            if prediction == 1:
                print(f"SPAM DETECTED! ({probability*100:.1f}% spam probability)")
            else:
                print(f"LEGITIMATE EMAIL ({probability*100:.1f}% spam probability)")
            print("-" * 70)
            print(f"Message: {text[:150]}{'...' if len(text) > 150 else ''}\n")

            # Save to emails.py
            save_to_inbox(text)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_email_interactive(threshold=0.8)