import os
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def cluster_embeddings(embeddings, method="dbscan", eps=0.3, min_samples=5, xi=0.05, min_cluster_size=5):
    if method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    elif method == "optics":
        model = OPTICS(metric="cosine", xi=xi, min_samples=min_cluster_size)
    else:
        raise ValueError("Unsupported method. Choose 'dbscan' or 'optics'.")

    print(f"Запуск кластеризации: {method.upper()}")
    labels = model.fit_predict(embeddings)
    return labels

def load_inputs(embeddings_path, questions_path):
    embeddings = np.load(embeddings_path)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f.readlines()]
    assert len(questions) == embeddings.shape[0], "📛 Несоответствие вопросов и эмбеддингов!"
    return questions, embeddings

def save_clusters(output_path, questions, labels):
    df = pd.DataFrame({"question": questions, "cluster": labels})
    df["is_noise"] = df["cluster"] == -1
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Сохранено в: {output_path}")
    return df

def main(embeddings_path, questions_path, output_path, method, eps, min_samples, xi, min_cluster_size):
    questions, embeddings = load_inputs(embeddings_path, questions_path)
    labels = cluster_embeddings(
        embeddings,
        method=method,
        eps=eps,
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size
    )

    df = save_clusters(output_path, questions, labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Кластеров: {n_clusters} | Шумов: {n_noise} / {len(labels)}")

    if n_clusters > 1:
        silhouette = silhouette_score(embeddings, labels, metric="cosine")
        print(f"Silhouette Score: {silhouette:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster question embeddings")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to .npy embeddings file")
    parser.add_argument("--questions", type=str, required=True, help="Path to .txt question list")
    parser.add_argument("--output", type=str, default="data/processed/clustered_questions.csv", help="Path to save result")
    parser.add_argument("--method", type=str, choices=["dbscan", "optics"], default="dbscan", help="Clustering method")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN: max distance between neighbors")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN: min samples per core point")
    parser.add_argument("--xi", type=float, default=0.05, help="OPTICS: steepness for cluster separation")
    parser.add_argument("--min_cluster_size", type=int, default=5, help="OPTICS: minimum cluster size")

    args = parser.parse_args()
    main(
        args.embeddings,
        args.questions,
        args.output,
        args.method,
        args.eps,
        args.min_samples,
        args.xi,
        args.min_cluster_size
    )
