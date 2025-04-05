import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import argparse

DEFAULT_INPUT = "data/processed/questions_answers_clean.csv"
DEFAULT_OUTPUT = "data/processed/questions_answers_augmented.csv"
NUM_PARAPHRASES = 2
MAX_QUESTIONS_PER_ANSWER = 5

def load_paraphraser(model_name="Vamsi/T5_Paraphrase_Paws", device=-1):
    return pipeline("text2text-generation", model=model_name, device=device)

def generate_paraphrases(paraphraser, question, num_return_sequences=2):
    input_text = f"paraphrase: {question} </s>"
    outputs = paraphraser(
        input_text,
        max_length=64,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.7,
        top_k=120,
        top_p=0.95,
    )
    return [o['generated_text'].strip().lower() for o in outputs]

def run_augmentation(input_path, output_path, num_paraphrases, max_per_answer):
    df = pd.read_csv(input_path)
    print(f"Загружено пар: {len(df)}")

    answer_to_questions = df.groupby("answer")["question"].apply(list).to_dict()
    paraphraser = load_paraphraser()

    new_rows = []

    for answer, questions in tqdm(answer_to_questions.items(), desc="🔁 Обогащение"):
        seen = set(questions)
        new_qs = list(questions)

        if len(new_qs) >= max_per_answer:
            new_rows.extend([(q, answer) for q in new_qs[:max_per_answer]])
            continue

        to_generate = max_per_answer - len(new_qs)

        for q in questions:
            if to_generate <= 0:
                break
            try:
                paras = generate_paraphrases(paraphraser, q, num_return_sequences=num_paraphrases)
                for p in paras:
                    if p not in seen:
                        new_qs.append(p)
                        seen.add(p)
                        to_generate -= 1
                        if to_generate == 0:
                            break
            except Exception as e:
                print(f"⚠ Ошибка при генерации для '{q}': {e}")
                continue

        new_rows.extend([(q, answer) for q in new_qs[:max_per_answer]])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_aug = pd.DataFrame(new_rows, columns=["question", "answer"]).drop_duplicates()
    df_aug.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Обогащённый датасет сохранён: {output_path} ({len(df_aug)} пар)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question paraphrasing for data augmentation.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to input CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save augmented CSV")
    parser.add_argument("--num_paraphrases", type=int, default=NUM_PARAPHRASES, help="Paraphrases per question")
    parser.add_argument("--max_questions", type=int, default=MAX_QUESTIONS_PER_ANSWER, help="Max questions per answer")

    args = parser.parse_args()
    run_augmentation(args.input, args.output, args.num_paraphrases, args.max_questions)
