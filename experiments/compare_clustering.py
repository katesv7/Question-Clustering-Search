import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm
import os

EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
OUTPUT_CSV = "experiments/clustering_comparison.csv"
os.makedirs("experiments", exist_ok=True)


X = np.load(EMBEDDINGS_PATH)
print(f"Загружено эмбеддингов: {X.shape}")


def evaluate_clustering(X, labels, method_name):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    if n_clusters <= 1:
        return {
            "method": method_name,
            "clusters": n_clusters,
            "noise": n_noise,
            "silhouette": -1,
            "db_index": -1
        }

    mask = labels != -1  
    try:
        sil = silhouette_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])
    except:
        sil, db = -1, -1

    return {
        "method": method_name,
        "clusters": n_clusters,
        "noise": n_noise,
        "silhouette": sil,
        "db_index": db
    }


results = []


for eps in [0.1, 0.15, 0.2]:
    model = DBSCAN(eps=eps, min_samples=3, metric="cosine")
    labels = model.fit_predict(X)
    results.append(evaluate_clustering(X, labels, f"DBSCAN (eps={eps})"))


optics = OPTICS(min_samples=3, metric="cosine")
labels = optics.fit_predict(X)
results.append(evaluate_clustering(X, labels, "OPTICS"))


for k in [10, 20, 30, 50]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    results.append(evaluate_clustering(X, labels, f"KMeans (k={k})"))


for k in [10, 20, 30, 50]:
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X)
    results.append(evaluate_clustering(X, labels, f"Agglomerative (k={k})"))


df = pd.DataFrame(results)
df = df.sort_values(by="silhouette", ascending=False)
df.to_csv(OUTPUT_CSV, index=False)

print("\nСравнение кластеризаторов:")
print(df.round(4))
