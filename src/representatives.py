import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
    return [questions[i] for i in top_idx], [embeddings[i] for i in top_idx]


def select_representatives(split="train", center_method="mean", top_k=1):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data", "processed")
    cluster_file = os.path.join(data_dir, f"clustered_{split}_questions_custom_encoder.csv")
    embeddings_path = os.path.join(data_dir, f"{split}_embeddings_custom_encoder.npy")
    output_csv = os.path.join(data_dir, f"representative_{split}_questions_custom_encoder.csv")
    output_txt = os.path.join(data_dir, f"representative_{split}_questions_custom_encoder.txt")
    output_npy = os.path.join(data_dir, f"representative_{split}_embeddings_custom_encoder.npy")
    output_centers = os.path.join(data_dir, f"cluster_centers_{split}_custom_encoder.npy")
    dummy_names_path = os.path.join(data_dir, f"cluster_center_names_{split}_custom_encoder.csv")

    print(f"\nОбработка: {split.upper()} (method: {center_method}, top_k={top_k})")

    df = pd.read_csv(cluster_file)
    embeddings = np.load(embeddings_path)

    if len(df) != len(embeddings):
        raise ValueError("Размерность эмбеддингов не совпадает с количеством вопросов")

    df["embedding_index"] = df.index
    df = df[df["cluster"] != -1]

    representatives = []
    all_rep_embeddings = []
    rep_questions_all = []
    cluster_centers = []

    for cluster_id, group in df.groupby("cluster"):
        idxs = group["embedding_index"].values
        cluster_embeddings = embeddings[idxs]
        cluster_questions = group["question"].tolist()

        center = get_cluster_center(cluster_embeddings, method=center_method)
        cluster_centers.append(center)

        top_questions, top_embeddings = find_top_k_representatives(
            cluster_questions, cluster_embeddings, center, top_k=top_k
        )

        representatives.append({
            "cluster": cluster_id,
            "representative_question": "; ".join(top_questions),
            "size": len(group)
        })
        all_rep_embeddings.extend(top_embeddings)
        rep_questions_all.extend(top_questions)


    os.makedirs(data_dir, exist_ok=True)
    df_result = pd.DataFrame(representatives)
    df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with open(output_txt, "w", encoding="utf-8") as f:
        for q in rep_questions_all:
            f.write(q.strip() + "\n")

    np.save(output_npy, np.array(all_rep_embeddings))
    np.save(output_centers, np.array(cluster_centers))
    dummy_names = [f"center_{i}" for i in range(len(cluster_centers))]
    pd.DataFrame({"question": dummy_names}).to_csv(dummy_names_path, index=False)

    print(f"Сохранено для split='{split}':")
    print(f"CSV с вопросами кластеров: {output_csv}")
    print(f"TXT репрезентативные вопросы: {output_txt}")
    print(f"NPY embeddings (вопросы): {output_npy}")
    print(f"NPY cluster centers (mean): {output_centers}")
    print(f"CSV dummy имена центров: {dummy_names_path}")

    return {
        "questions_csv": output_csv,
        "questions_txt": output_txt,
        "embeddings_npy": output_npy,
        "centers_npy": output_centers,
        "dummy_names_csv": dummy_names_path
    }


if __name__ == "__main__":
    for split in ["train", "test"]:
        select_representatives(split=split, center_method="mean", top_k=1)
