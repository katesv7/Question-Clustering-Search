# # import os
# # import random
# # import pandas as pd
# # from tqdm import tqdm
# # from sentence_transformers import SentenceTransformer, InputExample, losses
# # from torch.utils.data import DataLoader
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# #
# # # === Пути === #
# # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
# # MODEL_DIR = os.path.join(BASE_DIR, "models", "custom_encoder")
# # os.makedirs(MODEL_DIR, exist_ok=True)
# #
# # # === Загрузка данных === #
# # questions_path = os.path.join(DATA_DIR, "train_questions.csv")
# # clusters_path = os.path.join(DATA_DIR, "clustered_train_questions.csv")
# # answers_path = os.path.join(DATA_DIR, "answer_map_train_centers.csv")
# #
# # df_questions = pd.read_csv(questions_path)
# # df_clusters = pd.read_csv(clusters_path)
# # df_answers = pd.read_csv(answers_path)
# #
# # # === Мапа: cluster → answer === #
# # answer_map = {
# #     q.strip().lower(): a.strip().lower()
# #     for q, a in zip(df_answers["question"], df_answers["answer"])
# # }
# #
# # # === Объединение вопросов с кластерами и очистка === #
# # df_merged = df_questions.merge(df_clusters, on="question", how="left")
# # df_merged["cluster_str"] = df_merged["cluster"].apply(lambda x: f"center_{x}")
# # df_merged["answer"] = df_merged["cluster_str"].map(answer_map)
# # df_merged["question_clean"] = df_merged["question"].str.strip().str.lower()
# # df_filtered = df_merged[df_merged["answer"].notnull()].copy()
# # df_filtered = df_filtered.drop_duplicates(subset=["question_clean"])
# #
# # print(f"✅ Фильтровано: {len(df_filtered)} уникальных вопросов")
# #
# # # === Группировка по ответам === #
# # groups = df_filtered.groupby("answer")["question_clean"].apply(list)
# # all_questions = df_filtered["question_clean"].tolist()
# # question_to_answer = dict(zip(df_filtered["question_clean"], df_filtered["answer"]))
# #
# # # === Векторизация для hard negatives === #
# # print("🔠 Векторизация вопросов для подбора негативов...")
# # vectorizer = TfidfVectorizer().fit(all_questions)
# # tfidf_matrix = vectorizer.transform(all_questions)
# #
# # # Быстрая мапа вопрос → индекс
# # question_index = {q: i for i, q in enumerate(all_questions)}
# #
# # # === Генерация пар === #
# # pairs = []
# # max_negatives_per_question = 1
# #
# # print("🔧 Генерация пар (позитивные + hard негативные)...")
# # for group in tqdm(groups, desc="Обработка кластеров"):
# #     unique_group = list(set(group))
# #     if len(unique_group) < 2:
# #         continue
# #
# #     # Позитивные пары
# #     for i in range(len(unique_group)):
# #         for j in range(i + 1, len(unique_group)):
# #             pairs.append(InputExample(texts=[unique_group[i], unique_group[j]], label=1.0))
# #
# #     # Негативы: для каждого вопроса в группе ищем похожие, но с другим ответом
# #     for q in unique_group:
# #         q_idx = question_index[q]
# #         similarities = cosine_similarity(tfidf_matrix[q_idx], tfidf_matrix).flatten()
# #         sorted_indices = similarities.argsort()[::-1]
# #
# #         neg_count = 0
# #         for idx in sorted_indices:
# #             candidate = all_questions[idx]
# #             if candidate == q:
# #                 continue
# #             if question_to_answer[candidate] != question_to_answer[q]:
# #                 pairs.append(InputExample(texts=[q, candidate], label=0.0))
# #                 neg_count += 1
# #                 if neg_count >= max_negatives_per_question:
# #                     break
# #
# # print(f"✅ Всего сгенерировано пар: {len(pairs)}")
# #
# # # === Ограничение для отладки === #
# # pairs = pairs[:100000]
# #
# # # === Обучение модели === #
# # print("📦 Загружаем базовую модель...")
# # base_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
# #
# # train_dataloader = DataLoader(pairs, shuffle=True, batch_size=32)
# # train_loss = losses.CosineSimilarityLoss(model=base_model)
# #
# # print("🚀 Запуск обучения (1 эпоха)...")
# # base_model.fit(
# #     train_objectives=[(train_dataloader, train_loss)],
# #     epochs=1,
# #     warmup_steps=100,
# #     show_progress_bar=True
# # )
# #
# # # === Сохранение модели === #
# # base_model.save(MODEL_DIR)
# # print(f"✅ Модель сохранена в: {MODEL_DIR}")
# import os
# import pandas as pd
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from torch.utils.data import DataLoader
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
#
# # === Пути === #
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
# MODEL_DIR = os.path.join(BASE_DIR, "models", "custom_encoder_improved")
# os.makedirs(MODEL_DIR, exist_ok=True)
#
# # === Загрузка всех данных === #
# full_answer_map_path = os.path.join(DATA_DIR, "answer_map.csv")
# df_full = pd.read_csv(full_answer_map_path)
# df_full["question_clean"] = df_full["question"].str.strip().str.lower()
# df_full["answer_clean"] = df_full["answer"].str.strip().str.lower()
# df_full = df_full.drop_duplicates(subset=["question_clean"])
#
# # === Создание подмножества на 5000 строк === #
# df_subset = df_full.sample(n=5000, random_state=42)
# subset_path = os.path.join(DATA_DIR, "answer_map_subset_5000.csv")
# df_subset.to_csv(subset_path, index=False)
# print(f"✅ Сохранено подмножество в: {subset_path}")
#
# # === Подготовка вопросов и ответов === #
# questions = df_subset["question_clean"].tolist()
# answers = df_subset["answer_clean"].tolist()
# question_to_answer = dict(zip(questions, answers))
#
# # === Векторизация TF-IDF === #
# print("🔠 Векторизация TF-IDF...")
# vectorizer = TfidfVectorizer().fit(questions)
# tfidf_matrix = vectorizer.transform(questions)
#
# # === Поиск и генерация пар === #
# print("🔧 Генерация позитивных и hard негативных пар...")
# pairs = []
# for i, q in enumerate(tqdm(questions)):
#     q_answer = answers[i]
#     q_vec = tfidf_matrix[i]
#     sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
#
#     # Позитивные пары
#     for j in range(i + 1, len(questions)):
#         if answers[j] == q_answer:
#             pairs.append(InputExample(texts=[q, questions[j]], label=1.0))
#
#     # Hard negatives: cosine-сходные, но с другим ответом
#     similar_indices = sims.argsort()[::-1]
#     added = 0
#     for j in similar_indices:
#         if answers[j] != q_answer and 0.4 < sims[j] < 0.9:
#             pairs.append(InputExample(texts=[q, questions[j]], label=0.0))
#             added += 1
#         if added >= 1:
#             break
#
# print(f"✅ Всего пар: {len(pairs)}")
#
# # === Ограничение === #
# pairs = pairs[:100000]
#
# # === Обучение модели === #
# print("📦 Загружаем модель MiniLM...")
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#
# train_dataloader = DataLoader(pairs, shuffle=True, batch_size=32)
# train_loss = losses.CosineSimilarityLoss(model=model)
#
# print("🚀 Обучение (1 эпоха)...")
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=1,
#     warmup_steps=100,
#     show_progress_bar=True
# )
#
# # === Сохранение модели === #
# model.save(MODEL_DIR)
# print(f"✅ Модель сохранена в: {MODEL_DIR}")
#
# # === Оценка Recall@1 и Recall@5 === #
# print("📊 Оценка Recall@1 и Recall@5...")
# embeddings = model.encode(questions, normalize_embeddings=True)
# nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(embeddings)
# _, indices = nn.kneighbors(embeddings)
#
# match_1, match_5 = 0, 0
# for i, neigh in enumerate(indices):
#     top5 = neigh[1:]  # исключаем самого себя
#     if answers[i] == answers[top5[0]]:
#         match_1 += 1
#     if any(answers[i] == answers[j] for j in top5):
#         match_5 += 1
#
# recall1 = match_1 / len(questions)
# recall5 = match_5 / len(questions)
#
# print(f"🎯 Recall@1 = {recall1:.4f}")
# print(f"🎯 Recall@5 = {recall5:.4f}")
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "custom_encoder_mnr")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


df_full = pd.read_csv(os.path.join(DATA_DIR, "answer_map.csv"))
df_full["question_clean"] = df_full["question"].str.strip().str.lower()
df_full["answer_clean"] = df_full["answer"].str.strip().str.lower()
df_full = df_full.drop_duplicates(subset=["question_clean"])
df_subset = df_full.sample(n=30000, random_state=42)
df_subset.to_csv(os.path.join(DATA_DIR, "answer_map_subset_30000.csv"), index=False)

print(f"Подмножество: {len(df_subset)} вопросов")
questions = df_subset["question_clean"].tolist()
answers = df_subset["answer_clean"].tolist()
question_to_answer = dict(zip(questions, answers))

print("TF-IDF векторизация...")
vectorizer = TfidfVectorizer().fit(questions)
tfidf_matrix = vectorizer.transform(questions)

print(" Генерация (query, pos, hard_neg)...")
examples = []
for i, q in enumerate(tqdm(questions)):
    pos_candidates = [q2 for j, q2 in enumerate(questions) if answers[j] == answers[i] and q2 != q]
    if not pos_candidates:
        continue
    q_pos = pos_candidates[0]
    sims = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
    for idx in sims.argsort()[::-1]:
        q_neg = questions[idx]
        if answers[idx] != answers[i] and 0.3 < sims[idx] < 0.95:
            examples.append(InputExample(texts=[q, q_pos, q_neg]))
            break

print(f"Всего триплетов: {len(examples)}")


print("Загружаем paraphrase-mpnet-base-v2")
model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

train_dataloader = DataLoader(examples, shuffle=True, batch_size=32)
train_loss = MultipleNegativesRankingLoss(model=model)

print("Обучение (3 эпохи)...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True
)


model.save(MODEL_DIR)
print(f"Модель сохранена в: {MODEL_DIR}")


print("Recall@5")
embeddings = model.encode(questions, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(embeddings)
_, indices = nn.kneighbors(embeddings)

match_1, match_5 = 0, 0
for i, neigh in enumerate(indices):
    top5 = neigh[1:]
    if answers[i] == answers[top5[0]]:
        match_1 += 1
    if any(answers[i] == answers[j] for j in top5):
        match_5 += 1

recall1 = match_1 / len(questions)
recall5 = match_5 / len(questions)


df_result = pd.DataFrame({"Metric": ["Recall@1", "Recall@5"], "Value": [recall1, recall5]})
df_result.to_csv(os.path.join(DATA_DIR, "evaluation_metrics_mnr.csv"), index=False)

plt.figure(figsize=(6, 4))
plt.bar(["Recall@1", "Recall@5"], [recall1, recall5], color=["skyblue", "steelblue"])
plt.title("Semantic Recall@k (MNR + MPNet)")
plt.ylim(0, 1)
plt.ylabel("Recall")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "recall_plot_mnr.png"), dpi=300)
plt.show()

print(f"Recall@1 = {recall1:.4f}")
print(f"Recall@5 = {recall5:.4f}")
