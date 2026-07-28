"""
Train a sentiment classifier on the IMDB 50k Movie Reviews dataset.

Pipeline: HTML/regex cleaning -> TF-IDF (unigrams+bigrams) -> Logistic Regression
Also trains a Multinomial Naive Bayes and a Linear SVM for comparison, and saves
the best-performing model.

Run:
    python train.py
"""
import time
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc,
)

from preprocessing import clean_text

DATA_PATH = "data/IMDB Dataset.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset="review").reset_index(drop=True)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    print(f"Loaded {len(df)} unique reviews. Class balance:\n{df['label'].value_counts()}")
    return df


def main():
    df = load_data()

    print("Cleaning text...")
    t0 = time.time()
    df["clean_review"] = df["review"].apply(clean_text)
    print(f"Cleaning done in {time.time() - t0:.1f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
        "linear_svm": LinearSVC(random_state=RANDOM_STATE),
        "naive_bayes": MultinomialNB(),
    }

    results = {}
    fitted = {}
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = {"accuracy": acc, "f1": f1, "train_time": time.time() - t0}
        fitted[name] = clf
        print(f"[{name}] acc={acc:.4f} f1={f1:.4f} ({time.time()-t0:.1f}s)")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (f1={results[best_name]['f1']:.4f})")

    preds = best_model.predict(X_test_vec)
    print("\nClassification report (best model):")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve (works for LR and SVM via decision_function; NB via predict_proba)
    try:
        if hasattr(best_model, "predict_proba"):
            scores = best_model.predict_proba(X_test_vec)[:, 1]
        else:
            scores = best_model.decision_function(X_test_vec)
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{MODELS_DIR}/roc_curve.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Skipping ROC curve: {e}")

    # Save artifacts
    joblib.dump(best_model, f"{MODELS_DIR}/sentiment_model.joblib")
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.joblib")

    with open(f"{MODELS_DIR}/model_info.txt", "w") as f:
        f.write(f"Best model: {best_name}\n")
        for name, r in results.items():
            f.write(f"{name}: accuracy={r['accuracy']:.4f}, f1={r['f1']:.4f}\n")

    print(f"\nSaved model + vectorizer to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
