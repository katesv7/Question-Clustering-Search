import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "questions_answers_clean.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "questions_answers_augmented.csv")

MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
PARAPHRASES_PER_QUESTION = 1
MIN_QUESTIONS_PER_ANSWER = 3
MAX_ANSWERS_TO_AUGMENT = 5000

paraphraser = pipeline("text2text-generation", model=MODEL_NAME, tokenizer=MODEL_NAME, max_length=256, use_fast=False)

df = pd.read_csv(INPUT_FILE)
print(f"Загружено строк: {len(df)}")

answer_to_questions = df.groupby("answer")["question"].apply(list).to_dict()

small_groups = {a: list(set(q)) for a, q in answer_to_questions.items() if 1 <= len(set(q)) <= 2}
limited_groups = dict(list(small_groups.items())[:MAX_ANSWERS_TO_AUGMENT])
print(f"Будет обработано ответов: {len(limited_groups)}")


new_rows = []
for answer, questions in tqdm(limited_groups.items(), desc="Аугментация"):
    seen = set(questions)
    while len(seen) < MIN_QUESTIONS_PER_ANSWER:
        for q in list(seen):
            if len(seen) >= MIN_QUESTIONS_PER_ANSWER:
                break
            try:
                paraphrases = paraphraser(f"paraphrase: {q}", num_return_sequences=PARAPHRASES_PER_QUESTION, do_sample=True)
                for p in paraphrases:
                    gen_q = p["generated_text"].strip().lower()
                    if gen_q not in seen:
                        seen.add(gen_q)
            except Exception as e:
                print(f"Ошибка генерации для '{q}': {e}")
                continue
    new_rows.extend([(q, answer) for q in list(seen)[:MIN_QUESTIONS_PER_ANSWER]])

df_aug = pd.DataFrame(new_rows, columns=["question", "answer"])
df_final = pd.concat([df, df_aug], ignore_index=True).drop_duplicates()
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"После аугментации: {len(df_final)} строк (новых добавлено: {len(df_aug)})")
print(f"Сохранено в: {OUTPUT_FILE}")
