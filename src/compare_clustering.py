
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "combined_embeddings_custom_encoder.npy")
RESULT_CSV = os.path.join(DATA_DIR, "clustering_comparison_results_custom_encoder.csv")
RESULT_PLOT = os.path.join(FIG_DIR, "clustering_comparison_plot_custom_encoder.png")
embeddings = np.load(EMBEDDINGS_PATH)
results = []


for k in [50, 100, 150]:
    try:
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric="cosine")
        results.append({
            "method": "KMeans",
            "param": f"k={k}",
            "silhouette": score,
            "clusters": k,
            "noise": 0
        })
        print(f"KMeans k={k}, silhouette={score:.4f}")
    except Exception as e:
        print(f"KMeans k={k} — ошибка: {e}")


for eps in [0.1, 0.2, 0.3]:
    try:
        model = DBSCAN(eps=eps, min_samples=5, metric="cosine")
        labels = model.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = list(labels).count(-1)
        score = silhouette_score(embeddings, labels, metric="cosine") if n_clusters > 1 else None
        results.append({
            "method": "DBSCAN",
            "param": f"eps={eps}",
            "silhouette": score,
            "clusters": n_clusters,
            "noise": noise
        })
        print(f"DBSCAN eps={eps}, clusters={n_clusters}, silhouette={score}")
    except Exception as e:
        print(f"DBSCAN eps={eps} — ошибка: {e}")


for min_size in [5, 10, 20]:
    try:
        model = hdbscan.HDBSCAN(min_cluster_size=min_size, metric="euclidean", approx_min_span_tree=True)
        labels = model.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = list(labels).count(-1)
        score = silhouette_score(embeddings, labels, metric="cosine") if n_clusters > 1 else None
        results.append({
            "method": "HDBSCAN",
            "param": f"min_size={min_size}",
            "silhouette": score,
            "clusters": n_clusters,
            "noise": noise
        })
        print(f"HDBSCAN min_size={min_size}, clusters={n_clusters}, silhouette={score}")
    except Exception as e:
        print(f"HDBSCAN min_size={min_size} — ошибка: {e}")

df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print(f"\nРезультаты сохранены в: {RESULT_CSV}")


plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="param", y="silhouette", hue="method")
plt.title("Сравнение кластеризаций по Silhouette Score (Custom Encoder)")
plt.xticks(rotation=20)
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULT_PLOT, dpi=200)
print(f"График сохранён в: {RESULT_PLOT}")
plt.show()

