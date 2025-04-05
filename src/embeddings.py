import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

DEFAULT_INPUT = "data/processed/questions_answers_augmented.csv"
DEFAULT_EMBEDDINGS_OUT = "data/processed/question_embeddings.npy"
DEFAULT_TEXTS_OUT = "data/processed/question_texts.txt"
DEFAULT_ANSWER_MAP = "data/processed/answer_map.csv"
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
NORMALIZE_EMBEDDINGS = True

def encode_questions(model_name, questions, normalize=True):
    print(f"Загружаем модель: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Кодируем {len(questions)} вопросов...")
    embeddings = model.encode(
        questions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Средняя норма векторов: {np.mean(norms):.4f}")

    return embeddings

def run(input_path, emb_out_path, txt_out_path, map_out_path, model_name):
    df = pd.read_csv(input_path)
    print(f"Всего строк: {len(df)}")

    df = df.dropna(subset=["question", "answer"]).drop_duplicates()

    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    print(f"Уникальных вопросов: {len(set(questions))} / {len(questions)}")
    
    embeddings = encode_questions(model_name, questions, normalize=NORMALIZE_EMBEDDINGS)

    os.makedirs(os.path.dirname(emb_out_path), exist_ok=True)
    np.save(emb_out_path, embeddings)

    with open(txt_out_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.strip() + "\n")

    pd.DataFrame({"question": questions, "answer": answers}).to_csv(map_out_path, index=False, encoding="utf-8-sig")

    print(f"Сохранено:\n– embeddings: {emb_out_path}\n– texts: {txt_out_path}\n– map: {map_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode questions into embeddings with mapping.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="CSV with columns [question, answer]")
    parser.add_argument("--embeddings", type=str, default=DEFAULT_EMBEDDINGS_OUT, help="Path to save .npy")
    parser.add_argument("--texts", type=str, default=DEFAULT_TEXTS_OUT, help="Path to save .txt")
    parser.add_argument("--mapping", type=str, default=DEFAULT_ANSWER_MAP, help="Path to save question-answer map")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model name")

    args = parser.parse_args()
    run(args.input, args.embeddings, args.texts, args.mapping, args.model)
