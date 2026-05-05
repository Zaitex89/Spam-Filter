def run_eda(X, y, df):

    def print_ord(serie):
        for ord, freq in zip(serie.index.tolist(), serie.values.tolist()):
            print(f"  {str(ord)} - {str(int(freq))} gånger")

    def print_ratio(serie, typ):
        for ord, val in zip(serie.index.tolist(), serie.values.tolist()):
            print(f"  {str(ord)} - {str(round(val, 1))}x vanligare i {typ}")

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
    print_ratio((spam_freq / (ham_freq + 0.0001)).sort_values(ascending=False).head(15), "spam")

    print("\n" + "=" * 50)
    print("ORD SOM STARKAST INDIKERAR ÄKTA MAIL")
    print("=" * 50)
    print_ratio((ham_freq / (spam_freq + 0.0001)).sort_values(ascending=False).head(15), "äkta mail")