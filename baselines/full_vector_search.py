import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text

MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
QUESTIONS_PATH = "data/processed/question_texts.txt"
ANSWER_MAP_PATH = "data/processed/answer_map.csv"

print("Загрузка данных...")
embeddings = np.load(EMBEDDINGS_PATH)
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f.readlines()]
df_answers = pd.read_csv(ANSWER_MAP_PATH)
answer_map = dict(zip(df_answers["question"], df_answers["answer"]))


model = SentenceTransformer(MODEL_NAME)


def full_vector_search(query, top_k=5):
    vec = model.encode([clean_text(query)], normalize_embeddings=True)
    sims = cosine_similarity(vec, embeddings)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]

    results = []
    for i in top_idx:
        q = questions[i]
        a = answer_map.get(q, "(ответ не найден)")
        results.append({
            "question": q,
            "answer": a,
            "score": float(sims[i])
        })
    return results


if __name__ == "__main__":
    while True:
        query = input("\nВведите вопрос (или 'exit'): ").strip()
        if query.lower() == "exit":
            break

        results = full_vector_search(query)
        print(f"\n🔍 Результаты (Full Vector Search):")
        for r in results:
            print(f"— {r['question']}")
            print(f"  {r['answer']}")
            print(f"  {r['score']:.4f}\n")
