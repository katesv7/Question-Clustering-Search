import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "train_questions.csv")
ANSWER_MAP_PATH = os.path.join(BASE_DIR, "data", "processed", "answer_map.csv")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

class SemanticBaseline:
    def __init__(self, questions_path: str, answer_map_path: str,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

        df_q = pd.read_csv(questions_path)
        self.questions = df_q["question"].astype(str).apply(clean_text).tolist()

        df_map = pd.read_csv(answer_map_path)
        self.answer_map = dict(zip(df_map["question"], df_map["answer"]))
        self.answers = [self.answer_map.get(q, "(ответ не найден)") for q in df_q["question"]]

        self.model = SentenceTransformer(model_name)
        print(f"Генерация эмбеддингов модели: {model_name}")
        self.embeddings = self.model.encode(self.questions, normalize_embeddings=True, show_progress_bar=True)

    def retrieve(self, query: str, top_k: int = 5):
        query_vec = self.model.encode([clean_text(query)], normalize_embeddings=True)
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = np.argsort(sims)[-top_k:][::-1]

        return [
            {
                "rank": i + 1,
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "score": float(sims[idx])
            }
            for i, idx in enumerate(top_idx)
        ]


if __name__ == "__main__":
    searcher = SemanticBaseline(QUESTIONS_PATH, ANSWER_MAP_PATH)

    print("Semantic baseline (SentenceTransformer). Введите вопрос (или 'exit'):")

    while True:
        query = input("\nВопрос: ").strip()
        if query.lower() == "exit":
            break

        results = searcher.retrieve(query, top_k=5)
        print("\nТоп-результаты:")
        for r in results:
            print(f"{r['rank']}. {r['question']}")
            print(f"  {r['answer']}")
            print(f"  Cosine similarity: {r['score']:.4f}")
