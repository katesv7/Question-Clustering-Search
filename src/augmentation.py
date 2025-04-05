import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import argparse

DEFAULT_INPUT = "data/processed/questions_answers_clean.csv"
DEFAULT_OUTPUT = "data/processed/questions_answers_augmented.csv"
PARAPHRASES_PER_QUESTION = 2
MIN_REQUIRED = 4 

def load_paraphraser(model_name="Vamsi/T5_Paraphrase_Paws", device=-1):
    return pipeline("text2text-generation", model=model_name, device=device)

def generate_paraphrases(model, question, num_return_sequences=2):
    input_text = f"paraphrase: {question} </s>"
    outputs = model(
        input_text,
        max_length=64,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.7,
        top_k=120,
        top_p=0.95,
    )
    return [o["generated_text"].strip().lower() for o in outputs]

def run_augmentation(input_path, output_path, min_required, num_paraphrases):
    df = pd.read_csv(input_path)
    print(f"Загружено пар: {len(df)}")

    answer_to_questions = df.groupby("answer")["question"].apply(list).to_dict()
    new_rows = []

    paraphraser = load_paraphraser()

    for answer, questions in tqdm(answer_to_questions.items(), desc="🔁 Аугментация"):
        unique_q = list(set(questions))
        seen = set(unique_q)

        if len(unique_q) >= min_required:
            new_rows.extend([(q, answer) for q in unique_q])
            continue

        to_add = min_required - len(unique_q)

        for q in unique_q:
            if to_add <= 0:
                break
            try:
                paras = generate_paraphrases(paraphraser, q, num_paraphrases)
                for para in paras:
                    if para not in seen:
                        seen.add(para)
                        unique_q.append(para)
                        to_add -= 1
                        if to_add <= 0:
                            break
            except Exception as e:
                print(f"⚠ Ошибка генерации для '{q}': {e}")
                continue

        new_rows.extend([(q, answer) for q in unique_q[:min_required]])

    df_aug = pd.DataFrame(new_rows, columns=["question", "answer"])
    df_aug = df_aug.drop_duplicates()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_aug.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Обогащённый датасет сохранён: {output_path} ({len(df_aug)} пар)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective paraphrasing augmentation.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to input CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save augmented CSV")
    parser.add_argument("--min_required", type=int, default=MIN_REQUIRED, help="Minimum questions per answer")
    parser.add_argument("--num_paraphrases", type=int, default=PARAPHRASES_PER_QUESTION, help="Per question")

    args = parser.parse_args()
    run_augmentation(args.input, args.output, args.min_required, args.num_paraphrases)
