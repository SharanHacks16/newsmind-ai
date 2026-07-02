import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from train import METRICS_PATH, load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
GRAPH_DIR = BASE_DIR / "graphs"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
VECTORIZER_PATH = Path(__file__).resolve().parent / "vectorizer.pkl"

def ensure_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Run python train.py before python evaluate.py.")
    GRAPH_DIR.mkdir(exist_ok=True)

def plot_accuracy_comparison(metrics):
    results = metrics.get("model_results", {})
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] * 100 for name in names]
    plt.figure(figsize=(9, 5))
    sns.barplot(x=names, y=accuracies, hue=names, palette="viridis", legend=False)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Model")
    plt.ylim(0, 100)
    plt.xticks(rotation=12)
    for index, value in enumerate(accuracies):
        plt.text(index, value + 1, f"{value:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "accuracy_comparison.png", dpi=220)
    plt.close()

def plot_confusion_matrix(y_test, predictions, labels):
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "confusion_matrix.png", dpi=220)
    plt.close()

def plot_dataset_distribution(data):
    plt.figure(figsize=(8, 5))
    order = data["category"].value_counts().index
    sns.countplot(data=data, x="category", order=order, hue="category", palette="mako", legend=False)
    plt.title("Dataset Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Article Count")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "dataset_distribution.png", dpi=220)
    plt.close()

def plot_wordcloud(data):
    words = " ".join(data["clean_text"].tolist())
    cloud = WordCloud(width=1400, height=800, background_color="white", colormap="plasma").generate(words)
    plt.figure(figsize=(11, 6))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "wordcloud.png", dpi=220)
    plt.close()

def plot_top_words(data):
    tokens = " ".join(data["clean_text"].tolist()).split()
    word_counts = pd.Series(tokens).value_counts().head(20).sort_values()
    plt.figure(figsize=(9, 7))
    sns.barplot(x=word_counts.values, y=word_counts.index, hue=word_counts.index, palette="rocket", legend=False)
    plt.title("Top Frequent Words")
    plt.xlabel("Frequency")
    plt.ylabel("Stemmed Word")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "top_words.png", dpi=220)
    plt.close()

def main():
    ensure_artifacts()
    data = load_dataset()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    stratify = data["category"] if data["category"].value_counts().min() >= 2 else None
    _, x_test, _, y_test = train_test_split(data["clean_text"], data["category"], test_size=0.2, random_state=42, stratify=stratify)
    predictions = model.predict(vectorizer.transform(x_test))
    labels = sorted(data["category"].unique().tolist())
    plot_accuracy_comparison(metrics)
    plot_confusion_matrix(y_test, predictions, labels)
    plot_dataset_distribution(data)
    plot_wordcloud(data)
    plot_top_words(data)
    (GRAPH_DIR / "classification_report.txt").write_text(classification_report(y_test, predictions, zero_division=0), encoding="utf-8")
    print(f"Evaluation graphs and report saved to {GRAPH_DIR}")

if __name__ == "__main__":
    main()
