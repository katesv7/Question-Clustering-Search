import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.utils import clean_text


class ClusteredRetriever:
    def __init__(self,
                 model_name: str,
                 full_embeddings_path: str,
                 full_questions_path: str,
                 answer_map_path: str,
                 cluster_embeddings_path: str = None,
                 cluster_questions_path: str = None):

        print(f"Загружаем модель: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.full_embeddings = np.load(full_embeddings_path)
        df_q = pd.read_csv(full_questions_path)
        self.full_questions = df_q["question"].dropna().tolist()

        if len(self.full_questions) != len(self.full_embeddings):
            raise ValueError("Несовпадение количества эмбеддингов и вопросов")


        df_map = pd.read_csv(answer_map_path)
        self.answer_map = dict(zip(
            df_map["question"].astype(str).str.strip().str.lower(),
            df_map["answer"].astype(str).str.strip().str.lower()
        ))


        if cluster_embeddings_path and cluster_questions_path:
            self.cluster_embeddings = np.load(cluster_embeddings_path)
            df_clust = pd.read_csv(cluster_questions_path)
            self.cluster_questions = df_clust["representative_question"].dropna().tolist()

            if len(self.cluster_embeddings) != len(self.cluster_questions):
                raise ValueError("Несовпадение количества кластерных эмбеддингов и репрезентативных вопросов")
        else:
            self.cluster_embeddings = None
            self.cluster_questions = None

    def encode(self, query: str):
        cleaned = clean_text(query)
        return self.model.encode([cleaned], normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 5, mode: str = "full"):
        vec = self.encode(query)

        if mode == "full":
            sims = cosine_similarity(vec, self.full_embeddings)[0]
            questions = self.full_questions
        elif mode == "clustered":
            if self.cluster_embeddings is None or self.cluster_questions is None:
                raise ValueError("Кластерные данные не загружены.")
            sims = cosine_similarity(vec, self.cluster_embeddings)[0]
            questions = self.cluster_questions
        else:
            raise ValueError("Аргумент mode должен быть 'full' или 'clustered'.")

        top_idx = np.argsort(sims)[-top_k:][::-1]
        results = []

        for rank, idx in enumerate(top_idx, start=1):
            q = questions[idx]
            a = self.answer_map.get(q.strip().lower(), "(ответ не найден)")
            results.append({
                "rank": rank,
                "question": q,
                "answer": a,
                "score": float(sims[idx])
            })

        return results
