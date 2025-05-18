import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "train_questions.csv")
ANSWER_MAP_PATH = os.path.join(BASE_DIR, "data", "processed", "answer_map.csv")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

class TFIDFBaseline:
    def __init__(self, questions_path: str, answer_map_path: str):
        df_q = pd.read_csv(questions_path)
        self.questions = df_q["question"].astype(str).apply(clean_text).tolist()

        df_map = pd.read_csv(answer_map_path)
        self.answer_map = dict(zip(df_map["question"], df_map["answer"]))
        self.answers = [self.answer_map.get(q, "(ответ не найден)") for q in df_q["question"]]

        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def search(self, query: str, top_k: int = 5):
        query_clean = clean_text(query)
        query_vec = self.vectorizer.transform([query_clean])
        sims = cosine_similarity(query_vec, self.question_vectors)[0]

        top_indices = sims.argsort()[-top_k:][::-1]
        return [
            {
                "rank": i + 1,
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "score": float(sims[idx])
            }
            for i, idx in enumerate(top_indices)
        ]

if __name__ == "__main__":
    model = TFIDFBaseline(QUESTIONS_PATH, ANSWER_MAP_PATH)

    print("TF-IDF поиск по train вопросам. Введите запрос, чтобы найти похожие.")

    while True:
        query = input("\nВведите запрос (или 'exit'): ").strip()
        if query.lower() == "exit":
            break

        results = model.search(query, top_k=5)
        print("\nТоп-результаты:")
        for r in results:
            print(f"{r['rank']}. Вопрос: {r['question']}")
            print(f"Ответ: {r['answer']}")
            print(f"Сходство: {r['score']:.4f}")
