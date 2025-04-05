import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_embeddings(path):
    emb = np.load(path)
    print(f"{os.path.basename(path)}: {emb.shape}")
    return emb

def evaluate_embeddings(embeddings, top_k=5):
    norms = np.linalg.norm(embeddings, axis=1)
    sim_matrix = cosine_similarity(embeddings)

    top_k_scores = []
    for i in range(sim_matrix.shape[0]):
        sims = np.delete(sim_matrix[i], i)
        top_k_avg = np.sort(sims)[-top_k:].mean()
        top_k_scores.append(top_k_avg)

    return {
        "count": len(embeddings),
        "dim": embeddings.shape[1],
        "avg_norm": np.mean(norms),
        f"avg_top{top_k}_cosine": np.mean(top_k_scores),
        "similarity_distribution": top_k_scores,
    }

def plot_distribution(scores_dict):
    plt.figure(figsize=(10, 6))
    for model_name, scores in scores_dict.items():
        sns.kdeplot(scores, label=model_name, fill=True, linewidth=1.5)
    plt.title("📈 Cosine Similarity Distribution (Top-K avg)")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(embedding_paths, model_names, top_k=5):
    results = []
    sims_for_plot = {}

    for path, name in zip(embedding_paths, model_names):
        emb = load_embeddings(path)
        eval_result = evaluate_embeddings(emb, top_k=top_k)
        sims_for_plot[name] = eval_result.pop("similarity_distribution")
        eval_result["model"] = name
        results.append(eval_result)

    df = pd.DataFrame(results).set_index("model")
    print("\n📋 Сравнительная таблица моделей:\n")
    print(df.round(4))

    plot_distribution(sims_for_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple question embedding models")
    parser.add_argument("--embeddings", nargs="+", required=True, help="List of paths to .npy embeddings")
    parser.add_argument("--models", nargs="+", required=True, help="List of model names")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K neighbors to average")

    args = parser.parse_args()
    assert len(args.embeddings) == len(args.models), "Кол-во моделей и эмбеддингов должно совпадать"
    main(args.embeddings, args.models, args.top_k)
