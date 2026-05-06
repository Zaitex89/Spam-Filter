# Enron Spam Filter

A machine learning spam filter trained on the Enron email dataset.
The model classifies emails as spam or legitimate using Logistic Regression.

---

## Installation

1. Clone the repository
2. Create a virtual environment and install dependencies:

pip install pandas numpy scikit-learn kagglehub

---

## Project Structure

Spam-Filter/
├── main.py          ← trains and evaluates all models
├── eda.py           ← exploratory data analysis
├── ml.py            ← machine learning models
├── sort_emails.py   ← sorts inbox into spam and legitimate
├── testa_mail.py    ← tests individual emails
├── emails.py        ← add your own emails here
└── modell.pkl       ← saved model (created after running main.py)

---

## How to Run

### Step 1 — Train the model (required first)

python main.py

This will:
- Load and clean the dataset from Kaggle
- Run exploratory data analysis (EDA)
- Train and evaluate all models (Logistic Regression, Naive Bayes, Random Forest, KNN + PCA)
- Save the best model as modell.pkl

> You must run main.py before anything else — modell.pkl must exist first.

---

### Step 2 — Sort your inbox

python sort_emails.py

This will:
- Load the saved model from modell.pkl
- Read all emails from emails.py
- Sort them into LEGITIMATE and SPAM
- Print spam score for each email

To add your own emails, open emails.py and add them to the INBOX list:

INBOX = [
    "Your email here",
    "Another email here",
]

---

### Optional — Test individual emails

python testa_mail.py

This will run accuracy statistics on the emails in emails.py.

---

## Models Evaluated

| Model                  | Accuracy | Precision | Recall | F1    |
|------------------------|----------|-----------|--------|-------|
| Logistic Regression    | 98.2%    | 97.3%     | 96.3%  | 96.8% |
| Naive Bayes            | 95.6%    | 91.4%     | 93.2%  | 92.3% |
| Random Forest          | 96.3%    | 98.5%     | 88.5%  | 93.2% |
| KNN + PCA (50 comp.)   | 94.0%    | 92.7%     | 85.8%  | 89.1% |

All models evaluated with threshold = 0.8

---

## Dataset

- **Source:** [Kaggle — Email Spam Classification Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)
- **Size:** 5172 emails
- **Legitimate:** 3672 (71.0%)
- **Spam:** 1500 (29.0%)
- **Features:** 2565 word frequency columns