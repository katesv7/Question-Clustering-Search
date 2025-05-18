import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import pandas as pd
from sklearn.metrics import ndcg_score



def evaluate_embeddings(embeddings, top_k=5, n_clusters=100):
    """
    Оценка эмбеддингов:
    – Средняя норма векторов
    – Средняя похожесть на топ-k соседей
    – Silhouette score после кластеризации
    """
    norms = np.linalg.norm(embeddings, axis=1)
    sim_matrix = cosine_similarity(embeddings)

    top_k_scores = []
    for i in range(sim_matrix.shape[0]):
        sims = np.delete(sim_matrix[i], i)
        top_k_avg = np.sort(sims)[-top_k:].mean()
        top_k_scores.append(top_k_avg)

    try:
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        cluster_labels = km.fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, cluster_labels)
    except Exception as e:
        print(f"Не удалось посчитать silhouette: {e}")
        silhouette = None

    return {
        "count": len(embeddings),
        "dim": embeddings.shape[1],
        "avg_norm": np.mean(norms),
        f"avg_top{top_k}_cosine": np.mean(top_k_scores),
        "silhouette": silhouette,
        "similarity_distribution": top_k_scores,
    }

def evaluate_search(
    queries,
    search_fn,
    top_k=10,
    true_answer_map=None,
    return_details=False
):
    """
    Расширенная оценка поиска:
    Precision@1, Recall@k, MRR@k, NDCG@k, Latency
    """
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    ndcgs = []
    latencies = []
    details = []

    for q in queries:
        gold_answer = true_answer_map.get(q.strip().lower(), "").strip().lower() if true_answer_map else None
        if not gold_answer:
            continue

        t0 = time.time()
        results = search_fn(q, top_k=top_k)
        t1 = time.time()

        latencies.append(t1 - t0)

        found = False
        ranks = np.zeros(top_k)

        for idx, r in enumerate(results):
            retrieved_answer = r.get("answer", "").strip().lower()
            true_answer = gold_answer.strip().lower()
            if retrieved_answer == true_answer:
                if idx == 0:
                    hits_at_1 += 1
                hits_at_k += 1
                reciprocal_ranks.append(1 / (idx + 1))
                ranks[idx] = 1
                found = True
                break
        if not found:
            reciprocal_ranks.append(0)

        ndcgs.append(ndcg_score([ranks], [np.ones(top_k)]))

        if return_details:
            top = results[0] if results else {}
            details.append({
                "query": q,
                "top_question": top.get("question"),
                "top_answer": top.get("answer"),
                "score": top.get("score"),
                "latency": t1 - t0,
                "hit@1": int(ranks[0]),
                "hit@k": int(hits_at_k),
                "rr": reciprocal_ranks[-1],
                "ndcg": ndcgs[-1]
            })

    total = len(queries)
    summary = {
        "precision@1": hits_at_1 / total if total else 0.0,
        f"recall@{top_k}": hits_at_k / total if total else 0.0,
        f"mrr@{top_k}": np.mean(reciprocal_ranks),
        f"ndcg@{top_k}": np.mean(ndcgs),
        "avg_latency_sec": np.mean(latencies),
        "count": total
    }

    if return_details:
        print("Summary:\n", summary)
        print("Details:\n", pd.DataFrame(details).head())
        return summary, pd.DataFrame(details)

    print("Summary:\n", summary)
    return summary

