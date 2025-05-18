import os
import pandas as pd
from rank_bm25 import BM25Okapi
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

class BM25Baseline:
    def __init__(self, questions_path: str, answer_map_path: str):
        df_q = pd.read_csv(questions_path)
        self.questions = df_q["question"].astype(str).apply(clean_text).tolist()

        df_map = pd.read_csv(answer_map_path)
        self.answer_map = dict(zip(df_map["question"], df_map["answer"]))
        self.answers = [self.answer_map.get(q, "(ответ не найден)") for q in df_q["question"]]

        self.tokenized_corpus = [q.split() for q in self.questions]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5):
        query_tokens = clean_text(query).split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "rank": i + 1,
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "score": float(scores[idx])
            }
            for i, idx in enumerate(top_indices)
        ]


# === Пример запуска === #
if __name__ == "__main__":
    searcher = BM25Baseline(QUESTIONS_PATH, ANSWER_MAP_PATH)

    print("BM25 поиск по train вопросам. Введите вопрос (или 'exit'):")

    while True:
        query = input("\nВопрос: ").strip()
        if query.lower() == "exit":
            break

        results = searcher.retrieve(query, top_k=5)
        print("\nТоп-результаты:")
        for r in results:
            print(f"{r['rank']}.  {r['question']}")
            print(f" {r['answer']}")
            print(f"BM25 score: {r['score']:.4f}")
