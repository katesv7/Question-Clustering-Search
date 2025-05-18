import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.cluster import DBSCAN
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "figures")
CSV_PATH = os.path.join(DATA_DIR, "dbscan_semantic_grid_results.csv")
PLOT_PATH = os.path.join(FIG_DIR, "dbscan_semantic_recall_plot.png")
os.makedirs(FIG_DIR, exist_ok=True)

print("Загрузка эмбеддингов и вопросов...")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "train_embeddings.npy")
QUESTIONS_PATH = os.path.join(DATA_DIR, "train_questions.csv")
ANSWER_MAP_PATH = os.path.join(DATA_DIR, "answer_map.csv")

embeddings = np.load(EMBEDDINGS_PATH)
df_questions = pd.read_csv(QUESTIONS_PATH)
df_answers = pd.read_csv(ANSWER_MAP_PATH)

questions = df_questions["question"].dropna().tolist()
answer_map = {
    q.strip().lower(): a.strip().lower()
    for q, a in zip(df_answers["question"], df_answers["answer"])
}
answers = [answer_map.get(q.strip().lower(), "") for q in questions]


def semantic_recall_at_k(labels, true_answers, k=5):
    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[lbl].append(i)
    recalls = []
    for indices in clusters.values():
        if len(indices) < 2:
            continue
        cluster_answers = [true_answers[i] for i in indices]
        majority = max(set(cluster_answers), key=cluster_answers.count)
        match_count = sum(1 for a in cluster_answers if a == majority)
        recalls.append(match_count / len(cluster_answers))
    return np.mean(recalls) if recalls else 0.0



umap_model = umap.UMAP(n_components=10, random_state=42)
emb_umap = umap_model.fit_transform(embeddings)
print("UMAP завершён")


results = []
print("Grid Search DBSCAN по semantic recall@5")
for eps in np.arange(0.05, 0.61, 0.05):
    for min_samples in [3, 5, 10]:
        try:
            model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            labels = model.fit_predict(emb_umap)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            score = semantic_recall_at_k(labels, answers, k=5) if n_clusters > 1 else 0.0

            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "semantic_recall@5": score,
                "clusters": n_clusters,
                "noise": n_noise
            })
        except Exception as e:
            print(f"Ошибка при eps={eps}, min_samples={min_samples}: {e}")
            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "semantic_recall@5": None,
                "clusters": None,
                "noise": None
            })

df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False)
print(f"Результаты сохранены в: {CSV_PATH}")


plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="eps", y="semantic_recall@5", hue="min_samples", marker="o")
plt.title("Semantic Recall@5 после DBSCAN + UMAP")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
print(f"График сохранён в: {PLOT_PATH}")
