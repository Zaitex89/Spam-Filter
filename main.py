import kagglehub
import pandas as pd
import os

# Ladda ner dataset
path = kagglehub.dataset_download("wcukierski/enron-email-dataset")
print("Path to dataset files:", path)

# ── Kolla vad som finns i mappen ──────────────────────────
print("\nFiler i mappen:")
for f in os.listdir(path):
    size_mb = os.path.getsize(os.path.join(path, f)) / 1024 / 1024
    print(f"  {f}  ({size_mb:.1f} MB)")

# ── Läs in CSV ────────────────────────────────────────────
csv_path = os.path.join(path, "emails.csv")
df = pd.read_csv(csv_path)

# ── Grundläggande kontroll ────────────────────────────────
print("\nAntal rader och kolumner:", df.shape)
print("\nKolumnnamn:", df.columns.tolist())
print("\nFörsta raden - file-kolumn:")
print(df['file'].iloc[0])
print("\nFörsta raden - message-kolumn (första 500 tecken):")
print(df['message'].iloc[0][:500])