import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)
df_q = pd.read_csv(os.path.join(DATA_DIR, "combined_questions.csv"))
df_map = pd.read_csv(os.path.join(DATA_DIR, "answer_map.csv"))
df_q["cleaned"] = df_q["question"].str.strip().str.lower()
df_map["question"] = df_map["question"].str.strip().str.lower()
df_map["answer"] = df_map["answer"].str.strip().str.lower()
answer_map = dict(zip(df_map["question"], df_map["answer"]))
filtered_df = df_q[df_q["cleaned"].isin(answer_map.keys())].copy()
filtered_df["answer"] = filtered_df["cleaned"].map(answer_map)
questions = filtered_df["cleaned"].tolist()
answers = filtered_df["answer"].tolist()
print(f"Загружено {len(questions)} вопросов с ответами")


def semantic_recall_at_k(embeddings, labels, k=5):
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    match = 0
    for i, neigh in enumerate(indices):
        neigh = neigh[1:]
        if any(labels[i] == labels[j] for j in neigh):
            match += 1
    return match / len(labels)


MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "custom_kaggle": os.path.join(BASE_DIR, "models", "custom_encoder_kaggle"),
}


results = []


for name, path in MODELS.items():
    print(f"\n🔍 Модель: {name}")
    model = SentenceTransformer(path)
    embeddings = model.encode(questions, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    recall5 = semantic_recall_at_k(embeddings, answers, k=5)
    results.append({"model": name, "recall@5": recall5})
    print(f"Recall@5: {recall5:.4f}")


print("\nМодель: TF-IDF")
vectorizer = TfidfVectorizer().fit(questions)
tfidf_matrix = vectorizer.transform(questions)
nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(tfidf_matrix)
_, indices = nn.kneighbors(tfidf_matrix)
match = 0
for i, neigh in enumerate(indices):
    top5 = neigh[1:]
    if any(answers[i] == answers[j] for j in top5):
        match += 1
recall5 = match / len(questions)
results.append({"model": "TF-IDF", "recall@5": recall5})
print(f"TF-IDF Recall@5: {recall5:.4f}")


df_results = pd.DataFrame(results)
csv_path = os.path.join(DATA_DIR, "encoder_comparison_recall.csv")
df_results.to_csv(csv_path, index=False)
print(f"\nТаблица сохранена: {csv_path}")


plt.figure(figsize=(8, 5))
sns.barplot(data=df_results, x="model", y="recall@5", palette="Set2")
plt.ylim(0, 1)
plt.ylabel("Semantic Recall@5")
plt.title("Semantic Recall@5 Comparison")
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(FIGURE_DIR, "encoder_comparison_recall.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"График сохранён: {plot_path}")
