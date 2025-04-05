import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import clean_text


MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
QUESTIONS_PATH = "data/processed/question_texts.txt"
ANSWER_MAP_PATH = "data/processed/answer_map.csv"

N_CENTERS = 1000  
TOP_K = 5       


print("Загрузка данных...")
embeddings = np.load(EMBEDDINGS_PATH)
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f.readlines()]
df_answers = pd.read_csv(ANSWER_MAP_PATH)
answer_map = dict(zip(df_answers["question"], df_answers["answer"]))


random.seed(42)
center_idx = random.sample(range(len(questions)), N_CENTERS)
center_questions = [questions[i] for i in center_idx]
center_embeddings = embeddings[center_idx]


model = SentenceTransformer(MODEL_NAME)


def random_center_search(query, top_k=TOP_K):
    vec = model.encode([clean_text(query)], normalize_embeddings=True)
    sims = cosine_similarity(vec, center_embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for i in top_idx:
        q = center_questions[i]
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
        results = random_center_search(query)
        print(f"\nРезультаты (Random Centers, {N_CENTERS}):")
        for r in results:
            print(f"— {r['question']}")
            print(f"  {r['answer']}")
            print(f"  {r['score']:.4f}\n")
