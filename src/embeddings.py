import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from src.utils import get_custom_encoder_path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_FILE_CLEAN = os.path.join(BASE_DIR, "data", "processed", "questions_answers_clean.csv")
QA_FILE_AUG = os.path.join(BASE_DIR, "data", "processed", "questions_answers_augmented.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_NAME = get_custom_encoder_path("custom_encoder_kaggle")
TEST_SIZE = 0.2
RANDOM_SEED = 42
NORMALIZE_EMBEDDINGS = True


def encode_questions(model_name, questions, normalize=True):
    print(f"Загружаем модель: {model_name}")
    # model = SentenceTransformer(model_name)
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


def main():
    if os.path.exists(QA_FILE_AUG):
        source_file = QA_FILE_AUG
    elif os.path.exists(QA_FILE_CLEAN):
        source_file = QA_FILE_CLEAN
    else:
        raise FileNotFoundError("Не найден ни один файл: questions_answers_clean.csv или questions_answers_augmented.csv")

    print(f"Используем файл: {source_file}")
    df = pd.read_csv(source_file)

    if "answer" not in df.columns or "question" not in df.columns:
        raise ValueError("Файл должен содержать колонки 'question' и 'answer'")

    df = df.dropna(subset=["question", "answer"]).drop_duplicates()
    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    print(f"Загружено уникальных вопросов: {len(set(questions))}")

    train_q, test_q = train_test_split(questions, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    pd.DataFrame({"question": train_q}).to_csv(os.path.join(OUTPUT_DIR, "train_questions_custom_encoder.csv"), index=False)
    pd.DataFrame({"question": test_q}).to_csv(os.path.join(OUTPUT_DIR, "test_questions_custom_encoder.csv"), index=False)
    train_embeddings = encode_questions(MODEL_NAME, train_q, normalize=NORMALIZE_EMBEDDINGS)
    test_embeddings = encode_questions(MODEL_NAME, test_q, normalize=NORMALIZE_EMBEDDINGS)
    np.save(os.path.join(OUTPUT_DIR, "train_embeddings_custom_encoder.npy"), train_embeddings)
    np.save(os.path.join(OUTPUT_DIR, "test_embeddings_custom_encoder.npy"), test_embeddings)

    with open(os.path.join(OUTPUT_DIR, "question_texts_custom_encoder.txt"), "w", encoding="utf-8") as f:
        for q in train_q + test_q:
            f.write(q.strip() + "\n")

    answer_map = df[df["question"].isin(train_q + test_q)][["question", "answer"]]
    answer_map.to_csv(os.path.join(OUTPUT_DIR, "answer_map_custom_encoder.csv"), index=False, encoding="utf-8-sig")

    print("\nВсё сохранено:")
    print("train_questions.csv + test_questions.csv")
    print("train_embeddings.npy + test_embeddings.npy")
    print("answer_map.csv")
    print("question_texts.txt")

if __name__ == "__main__":
    main()
