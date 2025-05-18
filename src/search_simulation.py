import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os


def load_data(embeddings_path, cluster_file, reps_file):
    embeddings = np.load(embeddings_path)
    df_clusters = pd.read_csv(cluster_file)
    df_clusters = df_clusters[df_clusters["cluster"] != -1].reset_index(drop=True)
    df_reps = pd.read_csv(reps_file)

    return embeddings, df_clusters, df_reps


def run_simulation(embeddings, df_clusters, df_reps, top_k=5):
    questions = df_clusters["question"].tolist()
    clusters = df_clusters["cluster"].tolist()
    valid_idx = df_clusters.index.tolist()
    full_embeddings = embeddings[valid_idx]
    rep_q_to_cluster = {}
    rep_questions = []

    for row in df_reps.itertuples():
        c = row.cluster
        for q in row.representative_question.split("; "):
            rep_q_to_cluster[q] = c
            rep_questions.append(q)

    rep_embeddings = []
    rep_clusters = []
    for q in rep_questions:
        i = df_clusters[df_clusters["question"] == q].index
        if len(i) == 0:
            continue
        rep_embeddings.append(embeddings[i[0]])
        rep_clusters.append(df_clusters.loc[i[0], "cluster"])

    rep_embeddings = np.stack(rep_embeddings)

    print(f"Полных вопросов: {len(full_embeddings)} | Центров: {len(rep_embeddings)}")

    t0 = time.time()
    full_sim = cosine_similarity(full_embeddings)
    t_full = time.time() - t0

    t0 = time.time()
    center_sim = cosine_similarity(full_embeddings, rep_embeddings)
    t_center = time.time() - t0

    results = []
    for i in range(len(full_embeddings)):
        q = questions[i]
        c_true = clusters[i]
        s_full = full_sim[i]
        best_full_idx = np.argsort(s_full)[-2]
        score_full = s_full[best_full_idx]
        s_centers = center_sim[i]
        top_k_idx = np.argsort(s_centers)[-top_k:][::-1]
        score_center = s_centers[top_k_idx[0]]
        pred_cluster = rep_clusters[top_k_idx[0]]
        precision_at_1 = int(pred_cluster == c_true)
        recall_at_k = int(c_true in [rep_clusters[j] for j in top_k_idx])

        results.append({
            "score_full": score_full,
            "score_center": score_center,
            "score_diff": abs(score_full - score_center),
            "precision@1": precision_at_1,
            f"recall@{top_k}": recall_at_k
        })

    return pd.DataFrame(results), t_full, t_center


def analyze(df, t_full, t_center, top_k, save_path=None):
    print("\nScore diff:")
    print(df["score_diff"].describe())

    p1 = df["precision@1"].mean()
    r_k = df[f"recall@{top_k}"].mean()

    print(f"\nprecision@1: {p1:.3f}")
    print(f"recall@{top_k}: {r_k:.3f}")
    print(f"\nВремя поиска по всем: {t_full:.2f} сек")
    print(f"Время по центрам: {t_center:.2f} сек (ускорение ×{t_full / t_center:.1f})")


    plt.figure(figsize=(8, 4))
    sns.histplot(df["score_diff"], bins=30, kde=True, color="skyblue")
    plt.title("Разница score (все vs центры)")
    plt.xlabel("abs(score_full - score_center)")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"График сохранен в: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "processed", "train_embeddings.npy")
    CLUSTERS_CSV = os.path.join(BASE_DIR, "data", "processed", "clustered_train_questions.csv")
    REPS_CSV = os.path.join(BASE_DIR, "data", "processed", "representative_train_questions.csv")
    PLOT_SAVE_PATH = os.path.join(BASE_DIR, "figures", "score_diff_hist_train.png")
    TOP_K = 5

    emb, df_c, df_r = load_data(EMBEDDINGS_PATH, CLUSTERS_CSV, REPS_CSV)
    df_res, t1, t2 = run_simulation(emb, df_c, df_r, top_k=TOP_K)
    analyze(df_res, t1, t2, top_k=TOP_K, save_path=PLOT_SAVE_PATH)
