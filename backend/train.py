import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
VECTORIZER_PATH = Path(__file__).resolve().parent / "vectorizer.pkl"
METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"
DATASET_CANDIDATES = [DATASET_DIR / "bbc_news.csv", DATASET_DIR / "bbc-news-data.csv", DATASET_DIR / "BBC News Train.csv", DATASET_DIR / "bbc_news_sample.csv"]

def load_dataset():
    for path in DATASET_CANDIDATES:
        if path.exists():
            data = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("No dataset found. Add the BBC News Dataset CSV to the dataset folder or keep the included bbc_news_sample.csv file.")
    columns = {column.lower().strip(): column for column in data.columns}
    text_column = columns.get("text") or columns.get("article") or columns.get("content")
    label_column = columns.get("category") or columns.get("label")
    if text_column is None or label_column is None:
        raise ValueError("Dataset must contain text/article/content and category/label columns.")
    data = data[[text_column, label_column]].rename(columns={text_column: "text", label_column: "category"}).dropna()
    data["category"] = data["category"].str.strip().str.title()
    data["clean_text"] = data["text"].apply(preprocess_text)
    return data[data["clean_text"].str.len() > 0]

def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42),
    }

def main():
    data = load_dataset()
    stratify = data["category"] if data["category"].value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(data["clean_text"], data["category"], test_size=0.2, random_state=42, stratify=stratify)
    results = {}
    best_name = None
    best_pipeline = None
    best_accuracy = -1.0
    for name, model in build_models().items():
        pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=12000, ngram_range=(1, 2))), ("model", model)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="weighted", zero_division=0)
        results[name] = {"accuracy": round(float(accuracy), 4), "precision": round(float(precision), 4), "recall": round(float(recall), 4), "f1_score": round(float(f1), 4), "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True)}
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_name = name
            best_pipeline = pipeline
    joblib.dump(best_pipeline.named_steps["model"], MODEL_PATH)
    joblib.dump(best_pipeline.named_steps["tfidf"], VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps({"best_model": best_name, "categories": sorted(data["category"].unique().tolist()), "dataset_rows": int(len(data)), "model_results": results}, indent=2), encoding="utf-8")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")

if __name__ == "__main__":
    main()
