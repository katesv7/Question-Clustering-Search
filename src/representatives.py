import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
import matplotlib.pyplot as plt
import umap

def get_cluster_center(embeddings, method="mean"):
    if method == "mean":
        return np.mean(embeddings, axis=0)
    elif method == "median":
        return np.median(embeddings, axis=0)
    else:
        raise ValueError("Метод центра должен быть 'mean' или 'median'")

def find_top_k_representatives(questions, embeddings, center, top_k=1):
    sims = cosine_similarity([center], embeddings)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [questions[i] for i in top_idx]

def visualize_clusters(df, embeddings, output_path="data/processed/umap_clusters.png"):
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(embeddings)

    df_vis = df.copy()
    df_vis["x"] = X_2d[:, 0]
    df_vis["y"] = X_2d[:, 1]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_vis["x"], df_vis["y"], c=df_vis["cluster"], cmap="tab20", s=8, alpha=0.6)
    plt.title("Кластеры вопросов (UMAP проекция)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Визуализация сохранена: {output_path}")


def run(cluster_file, embeddings_path, output_file, center_method="mean", top_k=1, plot=True):
    df = pd.read_csv(cluster_file)
    embeddings = np.load(embeddings_path)

    if len(df) != len(embeddings):
        raise ValueError("Размерность эмбеддингов не совпадает с количеством вопросов")

    df["embedding_index"] = df.index
    df = df[df["cluster"] != -1]

    representatives = []
    for cluster_id, group in df.groupby("cluster"):
        idxs = group["embedding_index"].values
        cluster_embeddings = embeddings[idxs]
        cluster_questions = group["question"].tolist()

        center = get_cluster_center(cluster_embeddings, method=center_method)
        top_questions = find_top_k_representatives(cluster_questions, cluster_embeddings, center, top_k=top_k)

        representatives.append({
            "cluster": cluster_id,
            "representative_question": "; ".join(top_questions),
            "size": len(group)
        })

    df_result = pd.DataFrame(representatives)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_result.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Репрезентативные вопросы сохранены: {output_file}")
    print(df_result.head())

    if plot:
        visualize_clusters(df, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced representative question selection")
    parser.add_argument("--clusters", type=str, required=True, help="Path to clustered_questions.csv")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to question_embeddings.npy")
    parser.add_argument("--output", type=str, default="data/processed/representative_questions.csv")
    parser.add_argument("--center_method", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--top_k", type=int, default=1, help="How many representative questions per cluster")
    parser.add_argument("--no_plot", action="store_true", help="Disable UMAP visualization")

    args = parser.parse_args()
    run(
        cluster_file=args.clusters,
        embeddings_path=args.embeddings,
        output_file=args.output,
        center_method=args.center_method,
        top_k=args.top_k,
        plot=not args.no_plot
    )
