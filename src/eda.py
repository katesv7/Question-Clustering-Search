import os
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = "/Users/ek.son/Desktop/diplom/Diplom"
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "small_nq_new.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "questions_answers_clean.csv")
PLOT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(PLOT_DIR, exist_ok=True)
MAX_ROWS = None
HTML_CLEAN = True
ANSWER_LENGTH_LIMIT = 5000
MIN_ANSWER_LENGTH = 30


def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

question_answer_pairs = []
errors = 0

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in tqdm(enumerate(f), desc="Обработка JSONL"):
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

            question_answer_pairs.append((question, answer))
        except json.JSONDecodeError:
            errors += 1
            continue

df = pd.DataFrame(question_answer_pairs, columns=["question", "answer"])
df.drop_duplicates(inplace=True)
df["question"] = df["question"].apply(clean_text)
df["answer"] = df["answer"].apply(clean_text)
df = df[df["answer"].str.len().between(MIN_ANSWER_LENGTH, ANSWER_LENGTH_LIMIT)]
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


def save_and_show_plot(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=200)
    plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["question"].str.len(), bins=50, color="skyblue", ax=ax)
ax.set_title("Длина вопросов")
ax.set_xlabel("Символов в вопросе")
ax.set_ylabel("Количество")
ax.grid(True)
plt.tight_layout()
save_and_show_plot(fig, "question_lengths.png")


fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["answer"].str.len(), bins=50, color="salmon", ax=ax)
ax.set_title("Длина ответов (фильтрованные)")
ax.set_xlabel("Символов в ответе")
ax.set_ylabel("Количество")
ax.grid(True)
plt.tight_layout()
save_and_show_plot(fig, "answer_lengths.png")


answer_groups = df.groupby("answer")["question"].nunique()
multi_q_answers = answer_groups[answer_groups > 1]
long_answers = df[df["answer"].isin(multi_q_answers.index)]["answer"]
all_text = " ".join(long_answers)
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
ax.set_title("Облако слов из часто повторяющихся ответов")
plt.tight_layout()
save_and_show_plot(fig, "wordcloud.png")


fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(answer_groups, bins=30, color="mediumseagreen", ax=ax)
ax.set_title("Количество вопросов на один ответ")
ax.set_xlabel("Число уникальных вопросов")
ax.set_ylabel("Количество ответов")
ax.grid(True)
plt.tight_layout()
save_and_show_plot(fig, "questions_per_answer.png")


answer_counts = df["answer"].value_counts()
cumulative_questions = answer_counts.cumsum()
total_questions = len(df)
coverage_percent = cumulative_questions / total_questions * 100

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(coverage_percent) + 1), coverage_percent)
ax.axhline(80, color='red', linestyle='--', label="80% покрытие")
ax.set_title("Сколько разных ответов нужно, чтобы покрыть N% всех вопросов в датасете?")
ax.set_xlabel("Количество уникальных ответов (по убыванию частоты)")
ax.set_ylabel("Покрытие вопросов (%)")
ax.grid(True)
ax.legend()
plt.tight_layout()
save_and_show_plot(fig, "answer_coverage_plot.png")

num_answers_80_percent = (coverage_percent < 80).sum() + 1
print(f"Ответов, покрывающих 80% вопросов: {num_answers_80_percent}")


def plot_top_ngrams(corpus, ngram_range=(1, 2), top_n=20, title="Top N-grams", color="steelblue", filename="ngrams.png"):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]

    ngrams_df = pd.DataFrame(words_freq, columns=["ngram", "count"])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=ngrams_df, x="count", y="ngram", color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Частота")
    plt.tight_layout()
    save_and_show_plot(fig, filename)

questions = df["question"].dropna().tolist()
answers = df["answer"].dropna().tolist()

plot_top_ngrams(questions, (2, 2), 20, "Top Bigrams in Questions", "skyblue", "bigrams_questions.png")
plot_top_ngrams(questions, (3, 3), 20, "Top Trigrams in Questions", "skyblue", "trigrams_questions.png")
plot_top_ngrams(answers, (2, 2), 20, "Top Bigrams in Answers", "salmon", "bigrams_answers.png")
plot_top_ngrams(answers, (3, 3), 20, "Top Trigrams in Answers", "salmon", "trigrams_answers.png")
