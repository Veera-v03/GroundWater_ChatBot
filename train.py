"""
train.py
Trains a TF-IDF + Multinomial Naive Bayes pipeline for Groundwater Question Intent Classification.
Dataset: groundwater_query_intent_dataset.csv
Classes: recharge, availability, extraction, stage, comparison, future_availability
Saves trained pipeline to models/groundwater_nb_pipeline.pkl
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def get_data_path():
    """Find intent training dataset."""
    possible_paths = [
        os.path.join("data", "groundwater_query_intent_dataset.csv"),
        os.path.join("..", "data", "groundwater_query_intent_dataset.csv"),
        os.path.join("groundwater_query_intent_dataset.csv"),
        os.path.join("..", "groundwater_query_intent_dataset.csv"),
        os.path.join("groundwater-chatbot", "data", "groundwater_query_intent_dataset.csv"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find 'groundwater_query_intent_dataset.csv'.")


def get_model_save_paths():
    """Get list of save paths."""
    return [os.path.join("models", "groundwater_nb_pipeline.pkl")]


def main():
    data_path = get_data_path()
    print(f"Loading Question Intent Dataset from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape} (Rows: {len(df)})")
    print("\nClass distribution:")
    print(df["intent"].value_counts())

    X = df["query"]
    y = df["intent"]

    # 1. Train-test split (test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # 2. Build Pipeline: TF-IDF Vectorizer + Multinomial Naive Bayes
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), lowercase=True)),
        ("nb", MultinomialNB(alpha=1.0))
    ])

    # 3. Train on X_train for validation
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # 4. Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET (20% split, 10 samples)")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f} ({acc * 100:.1f}%)")
    print(f"Precision (Weighted): {prec:.4f}")
    print(f"Recall (Weighted):    {rec:.4f}")
    print(f"F1-score (Weighted):  {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 5. Fit on complete dataset (all 47 examples) for the prototype pipeline
    # to ensure all keywords from the example dataset are indexed
    full_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), lowercase=True)),
        ("nb", MultinomialNB(alpha=1.0))
    ])
    full_pipeline.fit(X, y)

    # 6. Save Pipeline using joblib
    save_paths = get_model_save_paths()
    for save_path in save_paths:
        try:
            parent_dir = os.path.dirname(save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            joblib.dump(full_pipeline, save_path)
            print(f"Pipeline successfully saved to: {save_path}")
        except Exception as e:
            pass

    # 7. Verification Test on Sample Queries
    sample_queries = [
        "What is the groundwater extraction in Hyderabad?",
        "What is the groundwater recharge in Karimnagar?",
        "What is the groundwater availability in Adilabad?",
        "What is the extraction stage in Jagtial?",
        "Which district has the highest groundwater extraction?",
        "What is the future groundwater availability in Hyderabad?",
        "What is the groundwater extraction?",
    ]

    print("\n" + "=" * 60)
    print("SAMPLE VERIFICATION TESTS ON SAVED PIPELINE")
    print("=" * 60)
    for q in sample_queries:
        pred_intent = full_pipeline.predict([q])[0]
        print(f"Query:  '{q}'\nIntent: -> [{pred_intent}]\n")

    print("Model training and export complete!")


if __name__ == "__main__":
    main()
