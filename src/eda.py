import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from bs4 import BeautifulSoup
import os

INPUT_FILE = "data/raw/small_nq_new.jsonl"
OUTPUT_FILE = "data/processed/questions_answers_clean.csv"
MAX_ROWS = None           
HTML_CLEAN = True         

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def clean_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()

def extract_question_answer_pairs():
    pairs = []
    errors = 0

    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in tqdm(enumerate(f), desc="🔍 Обработка строк"):
            if MAX_ROWS and i >= MAX_ROWS:
                break
            try:
                data = json.loads(line.strip())
                question = data.get("question_text", "").strip()
                annotations = data.get("annotations", [])
                candidates = data.get("long_answer_candidates", [])
                tokens = data.get("document_text", "").split(" ")

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
                if HTML_CLEAN:
                    answer = clean_html(answer)

                if len(answer) < 5:
                    continue

                question = clean_text(question)
                answer = clean_text(answer)

                pairs.append((question, answer))

            except json.JSONDecodeError:
                errors += 1
                continue

    print(f"Извлечено пар: {len(pairs)} | Ошибок JSON: {errors}")
    return pairs

def filter_multi_question_answers(pairs):
    df = pd.DataFrame(pairs, columns=["question", "answer"])
    df = df.drop_duplicates()

    counts = df.groupby("answer")["question"].nunique()
    valid_answers = counts[counts >= 2].index
    df = df[df["answer"].isin(valid_answers)]

    print(f"Ответов с ≥2 вопросами: {df['answer'].nunique()}")
    print(f"Финальный размер: {df.shape}")
    return df

def visualize_distribution(df):
    df["q_per_answer"] = df.groupby("answer")["question"].transform("nunique")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["q_per_answer"], bins=20, kde=False, color="skyblue")
    plt.title("📊 Количество вопросов на 1 ответ")
    plt.xlabel("Вопросов на ответ")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    pairs = extract_question_answer_pairs()
    df = filter_multi_question_answers(pairs)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {OUTPUT_FILE}")

    visualize_distribution(df)

if __name__ == "__main__":
    main()
