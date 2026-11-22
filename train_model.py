"""Train TF-IDF + Logistic Regression on a labeled CSV.
Expected columns: text,label. Labels should be real/fake (or another consistent binary pair).
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_news.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", default="models/fake_news_model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[[args.text_column, args.label_column]].dropna()
    df[args.text_column] = df[args.text_column].astype(str).str.strip()
    df[args.label_column] = df[args.label_column].astype(str).str.lower().str.strip()
    df = df[df[args.text_column].str.len() > 10]
    if df[args.label_column].nunique() < 2:
        raise SystemExit("Training needs at least two distinct label classes.")
    if len(df) < 20:
        print("WARNING: Very small dataset. Metrics will be unstable; collect more labeled examples.")
    stratify = df[args.label_column] if df[args.label_column].value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df[args.text_column], df[args.label_column], test_size=0.2,
        random_state=42, stratify=stratify
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2),
                                  max_features=30000, min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": model.named_steps["tfidf"],
                 "classifier": model.named_steps["classifier"]}, output)
    print(f"\nSaved model to {output}")

if __name__ == "__main__":
    main()
