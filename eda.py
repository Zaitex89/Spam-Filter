def run_eda(X, y, df):

    def print_word(serie):
        for ord, freq in zip(serie.index.tolist(), serie.values.tolist()):
            print(f"  {str(ord)} - {str(int(freq))} times")

    def print_ratio(serie, typ):
        for ord, val in zip(serie.index.tolist(), serie.values.tolist()):
            print(f"  {str(ord)} - {str(round(val, 1))}x more common in {typ}")

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"  Total emails:        {len(df)}")
    print(f"  Word features:       {X.shape[1]}")
    print(f"  Missing values:      {df.isnull().sum().sum()}")
    print(f"  Duplicate rows:      {df.duplicated().sum()}")

    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION")
    print("=" * 50)
    print(f"  Legitimate mail:   {(y==0).sum()}  ({(y==0).mean()*100:.1f}%)")
    print(f"  Spam:              {(y==1).sum()}  ({(y==1).mean()*100:.1f}%)")

    print("\n" + "=" * 50)
    print("AVERAGE WORDS PER EMAIL")
    print("=" * 50)
    print(f"  Legitimate mail:   {df[df['label']==0]['total_words'].mean():.0f} words")
    print(f"  Spam:              {df[df['label']==1]['total_words'].mean():.0f} words")

    print("\n" + "=" * 50)
    print("TOP 15 MOST COMMON WORDS IN SPAM")
    print("=" * 50)
    print_word(X[y==1].sum().sort_values(ascending=False).head(15))

    print("\n" + "=" * 50)
    print("TOP 15 MOST COMMON WORDS IN LEGITIMATE MAIL")
    print("=" * 50)
    print_word(X[y==0].sum().sort_values(ascending=False).head(15))

    spam_freq = X[y==1].mean()
    ham_freq  = X[y==0].mean()

    print("\n" + "=" * 50)
    print("WORDS MOST INDICATIVE OF SPAM")
    print("=" * 50)
    print_ratio((spam_freq / (ham_freq + 0.0001)).sort_values(ascending=False).head(15), "spam")

    print("\n" + "=" * 50)
    print("WORDS MOST INDICATIVE OF LEGITIMATE MAIL")
    print("=" * 50)
    print_ratio((ham_freq / (spam_freq + 0.0001)).sort_values(ascending=False).head(15), "legitimate mail")