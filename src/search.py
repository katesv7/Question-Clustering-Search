import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_embeddings, load_questions, load_model, clean_text

def search(query, mode, model_name, k, embeddings_path, questions_path, answer_map_path, cluster_only=False):
    model = load_model(model_name)
    query_clean = clean_text(query)
    query_vec = model.encode([query_clean], normalize_embeddings=True)

    all_embeddings = load_embeddings(embeddings_path)
    all_questions = load_questions(questions_path)
    answer_map = pd.read_csv(answer_map_path)

    sims = cosine_similarity(query_vec, all_embeddings)[0]
    top_idx = np.argsort(sims)[-k:][::-1]

    print(f"\nРезультаты для запроса: '{query}'\n")
    for rank, idx in enumerate(top_idx, 1):
        q = all_questions[idx]
        a_row = answer_map[answer_map["question"] == q]
        a = a_row["answer"].iloc[0] if not a_row.empty else "(ответ не найден)"
        print(f"{rank}. {q}\n   💬 {a}\n   📏 Similarity: {sims[idx]:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск похожих вопросов")
    parser.add_argument("--query", type=str, required=True, help="Текст запроса")
    parser.add_argument("--mode", type=str, choices=["full", "clustered"], default="clustered", help="Поиск по всей базе или центрам")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base", help="Модель эмбеддинга")
    parser.add_argument("--k", type=int, default=5, help="Сколько результатов возвращать")

    parser.add_argument("--embeddings", type=str,
        default="data/processed/question_embeddings.npy")
    parser.add_argument("--questions", type=str,
        default="data/processed/question_texts.txt")
    parser.add_argument("--answer_map", type=str,
        default="data/processed/answer_map.csv")

    args = parser.parse_args()

    search(
        query=args.query,
        mode=args.mode,
        model_name=args.model,
        k=args.k,
        embeddings_path=args.embeddings,
        questions_path=args.questions,
        answer_map_path=args.answer_map,
        cluster_only=(args.mode == "clustered")
    )
