import re
from bs4 import BeautifulSoup
import json
import pandas as pd
from tqdm import tqdm
import os


def clean_text(text: str) -> str:
    """
    Удаляет HTML-теги и приводит текст к нижнему регистру.
    """
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_jsonl(filepath: str, max_rows=None) -> list:
    """
    Загружает jsonl-файл и возвращает список словарей.
    """
    data = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def save_df(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Сохранено в: {path}")


def print_banner(title: str):
    print(f"\n{'=' * 40}\n {title}\n{'=' * 40}")


def extract_question_answer_pairs(data: list, html_clean=True) -> list:
    """
    Извлекает пары вопрос-ответ из загруженного jsonl.
    """
    pairs = []
    for item in tqdm(data, desc="Извлечение пар"):
        question = item.get("question_text", "").strip()
        annotations = item.get("annotations", [])
        candidates = item.get("long_answer_candidates", [])
        tokens = item.get("document_text", "").split(" ")

        if not annotations or not candidates or not question:
            continue

        long_ans = annotations[0].get("long_answer", {})
        idx = long_ans.get("candidate_index", -1)

        if idx == -1 or idx >= len(candidates):
            continue

        start = candidates[idx].get("start_token")
        end = candidates[idx].get("end_token")

        if start is None or end is None or end <= start:
            continue

        answer = " ".join(tokens[start:end]).strip()
        if html_clean:
            answer = BeautifulSoup(answer, "html.parser").get_text()

        if len(answer) < 5:
            continue

        pairs.append((question, answer))
    return pairs


def get_custom_encoder_path(model_name="custom_encoder_kaggle"):
    """
    Возвращает путь к кастомной модели SentenceTransformer.
    :param model_name: Название папки внутри models/
    :return: Строка пути
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

    return model_path
