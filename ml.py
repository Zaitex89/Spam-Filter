from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

def run_ml(X, y):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training data:  {X_train.shape[0]} emails")
    print(f"Test data:      {X_test.shape[0]} emails")

    # Help function
    def evaluate(name, y_test, y_pred):
        print("\n" + "=" * 50)
        print(name)
        print("=" * 50)
        print(f"  Accuracy:   {accuracy_score(y_test, y_pred)*100:.1f}%")
        print(f"  Precision:  {precision_score(y_test, y_pred)*100:.1f}%")
        print(f"  Recall:     {recall_score(y_test, y_pred)*100:.1f}%")
        print(f"  F1-score:   {f1_score(y_test, y_pred)*100:.1f}%")
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion matrix:")
        print(f"  Correct legitimate:   {cm[0][0]}")
        print(f"  Legitimate as spam:   {cm[0][1]}")
        print(f"  Missed spam:          {cm[1][0]}")
        print(f"  Correct spam:         {cm[1][1]}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    evaluate("LOGISTIC REGRESSION", y_test, lr.predict(X_test))

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    evaluate("NAIVE BAYES", y_test, nb.predict(X_test))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    evaluate("RANDOM FOREST", y_test, rf.predict(X_test))

    # KNN + PCA
    # PCA reduces from 2565 features to 50 dimensions
    pca = PCA(n_components=50, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)
    evaluate("KNN + PCA (50 components)", y_test, knn.predict(X_test_pca))

    # Saves best model
    with open("modell.pkl", "wb") as f:
        pickle.dump(lr, f)
    print("\nModel saved as modell.pkl")