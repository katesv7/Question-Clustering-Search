
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)


def cluster_kmeans(embeddings, n_clusters=100):
    print(f"Кластеризация: KMeans (n_clusters={n_clusters})")
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(embeddings)
    return labels


def find_best_k(embeddings, k_range=range(20, 201, 20)):
    print("Подбор лучшего k по silhouette_score...")
    best_k = None
    best_score = -1
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric="cosine")
        print(f"k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Лучшее k: {best_k} (silhouette={best_score:.4f})")
    return best_k


def process_split(split: str, n_clusters=100):
    print(f"\nОбработка: {split.upper()}")

    embeddings_path = os.path.join(DATA_DIR, f"{split}_embeddings_custom_encoder.npy")
    questions_path = os.path.join(DATA_DIR, f"{split}_questions_custom_encoder.csv")
    output_path = os.path.join(DATA_DIR, f"clustered_{split}_questions_custom_encoder.csv")

    embeddings = np.load(embeddings_path)
    df = pd.read_csv(questions_path)
    questions = df["question"].dropna().tolist()

    if len(questions) != embeddings.shape[0]:
        raise ValueError(f"Несоответствие размеров: {len(questions)} вопросов vs {embeddings.shape[0]} эмбеддингов")

    dists = pairwise_distances(embeddings[:1000])
    avg_dist = np.mean(dists)
    print(f"Средняя попарная дистанция (1000 примеров): {avg_dist:.4f}")

    best_k = find_best_k(embeddings)
    labels = cluster_kmeans(embeddings, n_clusters=best_k)

    df_out = pd.DataFrame({"question": questions, "cluster": labels})
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {output_path}")

    silhouette = silhouette_score(embeddings, labels, metric="cosine")
    print(f"Silhouette Score: {silhouette:.4f}")

if __name__ == "__main__":
    for split in ["train", "test"]:
        process_split(split)


    print("\nОбъединяем train и test в combined...")

    train_emb = np.load(os.path.join(DATA_DIR, "train_embeddings_custom_encoder.npy"))
    test_emb = np.load(os.path.join(DATA_DIR, "test_embeddings_custom_encoder.npy"))
    combined_emb = np.vstack([train_emb, test_emb])
    combined_emb_path = os.path.join(DATA_DIR, "combined_embeddings_custom_encoder.npy")
    np.save(combined_emb_path, combined_emb)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_questions_custom_encoder.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_questions_custom_encoder.csv"))
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_questions_path = os.path.join(DATA_DIR, "combined_questions_custom_encoder.csv")
    combined_df.to_csv(combined_questions_path, index=False, encoding="utf-8-sig")

    print(f"Сохранено: {combined_emb_path} и {combined_questions_path}")


    print(f"\nОбработка: COMBINED")
    embeddings = np.load(combined_emb_path)
    df = pd.read_csv(combined_questions_path)
    questions = df["question"].dropna().tolist()
    best_k = find_best_k(embeddings)
    labels = cluster_kmeans(embeddings, n_clusters=best_k)

    clustered_combined_path = os.path.join(DATA_DIR, "clustered_combined_questions_custom_encoder.csv")
    df_out = pd.DataFrame({"question": questions, "cluster": labels})
    df_out.to_csv(clustered_combined_path, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {clustered_combined_path}")

    silhouette = silhouette_score(embeddings, labels, metric="cosine")
    print(f"Silhouette Score (COMBINED): {silhouette:.4f}")
