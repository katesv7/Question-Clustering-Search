
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


TOP_K = 5
N_CLUSTERS = 100
FILES_TO_EVALUATE = [
    "train_embeddings_custom_encoder.npy",
    "test_embeddings_custom_encoder.npy"
]

def load_embeddings(path):
    embeddings = np.load(path)
    print(f"Загружено {embeddings.shape[0]} векторов размерности {embeddings.shape[1]}")
    return embeddings

def evaluate_embeddings(embeddings, top_k=5, n_clusters=100):
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Средняя норма векторов: {np.mean(norms):.4f}")

    sim_matrix = cosine_similarity(embeddings)
    top_k_scores = []
    for i in range(sim_matrix.shape[0]):
        sims = np.delete(sim_matrix[i], i)
        top_k_avg = np.sort(sims)[-top_k:].mean()
        top_k_scores.append(top_k_avg)
    avg_top_k = np.mean(top_k_scores)
    print(f"Средняя похожесть с Top-{top_k} соседями: {avg_top_k:.4f}")

    try:
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        cluster_labels = km.fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette score: {silhouette:.4f}")
    except Exception as e:
        silhouette = None
        print(f"Ошибка silhouette: {e}")

    return {
        "count": len(embeddings),
        "dim": embeddings.shape[1],
        "avg_norm": np.mean(norms),
        f"avg_top{top_k}_cosine": avg_top_k,
        "silhouette": silhouette,
        "similarity_distribution": top_k_scores
    }

def plot_similarity_distribution(scores, title, save_path):
    sns.histplot(scores, bins=30, kde=True, color="skyblue")
    plt.title(title)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"График сохранён: {save_path}")

if __name__ == "__main__":
    all_results = {}

    for file_name in FILES_TO_EVALUATE:
        file_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}")
            continue

        print(f"\n=== Оценка файла: {file_name} ===")
        embeddings = load_embeddings(file_path)
        result = evaluate_embeddings(embeddings, top_k=TOP_K, n_clusters=N_CLUSTERS)

        tag = file_name.replace(".npy", "")
        all_results[tag] = result

        eval_csv_path = os.path.join(DATA_DIR, f"embedding_eval_{tag}.csv")
        fig_path = os.path.join(FIGURE_DIR, f"similarity_dist_{tag}.png")

        pd.DataFrame([result]).to_csv(eval_csv_path, index=False, encoding="utf-8-sig")
        plot_similarity_distribution(result["similarity_distribution"],
                                     f"Top-{TOP_K} Similarity — {tag}", fig_path)

        print(f"Оценка для {tag} завершена. Метрики: {eval_csv_path}")


    if "train_embeddings_custom_encoder" in all_results and "test_embeddings_custom_encoder" in all_results:
        train = all_results["train_embeddings_custom_encoder"]
        test = all_results["test_embeddings_custom_encoder"]

        comparison = {
            "Metric": ["avg_top5_cosine", "silhouette"],
            "Train": [train["avg_top5_cosine"], train["silhouette"]],
            "Test": [test["avg_top5_cosine"], test["silhouette"]],
            "Difference (Test - Train)": [
                test["avg_top5_cosine"] - train["avg_top5_cosine"],
                test["silhouette"] - train["silhouette"] if train["silhouette"] and test["silhouette"] else None
            ]
        }

        comp_df = pd.DataFrame(comparison)
        print("\nСравнение качества эмбеддингов:")
        print(comp_df.to_string(index=False))

