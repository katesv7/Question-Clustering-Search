import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import clean_text

class ClusteredRetriever:
    def __init__(self,
                 model_name: str,
                 full_embeddings_path: str,
                 full_questions_path: str,
                 answer_map_path: str,
                 cluster_embeddings_path: str = None,
                 cluster_questions_path: str = None):

        self.model = SentenceTransformer(model_name)

        self.full_embeddings = np.load(full_embeddings_path)
        with open(full_questions_path, "r", encoding="utf-8") as f:
            self.full_questions = [line.strip() for line in f.readlines()]

        import pandas as pd
        df = pd.read_csv(answer_map_path)
        self.answer_map = dict(zip(df["question"], df["answer"]))

        if cluster_embeddings_path and cluster_questions_path:
            self.cluster_embeddings = np.load(cluster_embeddings_path)
            with open(cluster_questions_path, "r", encoding="utf-8") as f:
                self.cluster_questions = [line.strip() for line in f.readlines()]
        else:
            self.cluster_embeddings = None
            self.cluster_questions = None

    def encode(self, query: str):
        return self.model.encode([clean_text(query)], normalize_embeddings=True)

    def retrieve(self, query: str, top_k=5, mode="full"):
        vec = self.encode(query)

        if mode == "full":
            sims = cosine_similarity(vec, self.full_embeddings)[0]
            questions = self.full_questions
        elif mode == "clustered":
            if self.cluster_embeddings is None:
                raise ValueError("Кластерные эмбеддинги не загружены!")
            sims = cosine_similarity(vec, self.cluster_embeddings)[0]
            questions = self.cluster_questions
        else:
            raise ValueError("mode должен быть 'full' или 'clustered'")

        top_idx = np.argsort(sims)[-top_k:][::-1]
        results = []

        for rank, idx in enumerate(top_idx, 1):
            q = questions[idx]
            a = self.answer_map.get(q, "(ответ не найден)")
            results.append({
                "rank": rank,
                "question": q,
                "answer": a,
                "similarity": float(sims[idx])
            })

        return results
