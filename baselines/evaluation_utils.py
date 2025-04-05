import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import clean_text


def evaluate_search(
    queries,
    search_fn,
    top_k=5,
    return_details=False
):
    """
    Оценивает функцию поиска на списке запросов
    :param queries: список строк (вопросов)
    :param search_fn: функция поиска, возвращающая список словарей [{question, answer, score}]
    :param top_k: сколько результатов учитывать
    :param return_details: вернуть ли таблицу всех результатов
    :return: средняя метрика, время
    """
    all_scores = []
    times = []
    results = []

    for q in queries:
        t0 = time.time()
        res = search_fn(q, top_k=top_k)
        t1 = time.time()

        if not res:
            continue

        top_score = res[0]["score"]
        all_scores.append(top_score)
        times.append(t1 - t0)

        if return_details:
            results.append({
                "query": q,
                "top_score": top_score,
                "top_question": res[0]["question"],
                "latency": t1 - t0
            })

    summary = {
        "avg_top_score": np.mean(all_scores),
        "avg_latency_sec": np.mean(times),
        "count": len(all_scores)
    }

    if return_details:
        return summary, pd.DataFrame(results)
    return summary
