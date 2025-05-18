# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
# from clustered_retriever import ClusteredRetriever
#
# # === Пути === #
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
# OUTPUT_CSV = os.path.join(DATA_DIR, "search_results_comparison.csv")
# PLOT_SIM_PATH = os.path.join(DATA_DIR, "similarity_comparison_train.png")
# PLOT_TIME_PATH = os.path.join(DATA_DIR, "runtime_comparison_train.png")
#
# # === Модель === #
# model = ClusteredRetriever(
#     model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
#     full_embeddings_path=os.path.join(DATA_DIR, "train_embeddings.npy"),
#     full_questions_path=os.path.join(DATA_DIR, "train_questions.csv"),
#     answer_map_path=os.path.join(DATA_DIR, "answer_map.csv"),
#     cluster_embeddings_path=os.path.join(DATA_DIR, "representative_train_embeddings.npy"),
#     cluster_questions_path=os.path.join(DATA_DIR, "representative_train_questions.csv")
# )
#
# # === Вопросы (ручной ввод или загрузка test set) === #
# queries = [
#     "What is The Mother 's Name on ' How I Met Your Mother ' ?",
#     "What does DNA stand for?",
#     "Who wrote Harry Potter?",
#     "When did World War II end?",
#     "How tall is Mount Everest?"
# ]
#
# results = []
# timings = []
#
# for query in queries:
#     print(f"\n🔍 Вопрос: {query}")
#     for mode in ["clustered", "full"]:
#         start = time.time()
#         top_k = model.retrieve(query, top_k=1, mode=mode)
#         elapsed = time.time() - start
#         timings.append({"query": query, "mode": mode, "time_sec": elapsed})
#
#         for res in top_k:
#             results.append({
#                 "query": query,
#                 "mode": mode,
#                 "rank": res["rank"],
#                 "question": res["question"],
#                 "answer": res["answer"],
#                 "similarity": res["score"]
#             })
#
# # === Сохраняем CSV === #
# df = pd.DataFrame(results)
# os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
# df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
# print(f"\n📄 Результаты сохранены в: {OUTPUT_CSV}")
#
# # === Визуализация: Сходство === #
# plt.figure(figsize=(10, 5))
# sns.barplot(data=df[df["rank"] == 1], x="query", y="similarity", hue="mode", palette="pastel")
# plt.title("📈 Сходство топ-1 ответа: Full vs Clustered")
# plt.ylabel("Cosine Similarity")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig(PLOT_SIM_PATH, dpi=200)
# print(f"✅ График сходства сохранён в: {PLOT_SIM_PATH}")
#
# # === Визуализация: Время === #
# df_time = pd.DataFrame(timings)
# plt.figure(figsize=(10, 5))
# sns.barplot(data=df_time, x="query", y="time_sec", hue="mode", palette="Set2")
# plt.title("⏱️ Время выполнения поиска: Full vs Clustered")
# plt.ylabel("Seconds")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig(PLOT_TIME_PATH, dpi=200)
# print(f"✅ График времени сохранён в: {PLOT_TIME_PATH}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from clustered_retriever import ClusteredRetriever
import os



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_CSV = os.path.join(DATA_DIR, "search_results_comparison_custom_encoder.csv")
PLOT_SIM_PATH = os.path.join(DATA_DIR, "similarity_comparison_train_custom_encoder.png")
PLOT_TIME_PATH = os.path.join(DATA_DIR, "runtime_comparison_train_custom_encoder.png")
PLOT_AVG_LATENCY_PATH = os.path.join(DATA_DIR, "avg_runtime_small_eval_custom_encoder.png")
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "custom_encoder_kaggle")

model = ClusteredRetriever(
    model_name=LOCAL_MODEL_PATH,
    full_embeddings_path=os.path.join(DATA_DIR, "train_embeddings_custom_encoder.npy"),
    full_questions_path=os.path.join(DATA_DIR, "train_questions_custom_encoder.csv"),
    answer_map_path=os.path.join(DATA_DIR, "answer_map_custom_encoder.csv"),
    cluster_embeddings_path=os.path.join(DATA_DIR, "representative_train_embeddings_custom_encoder.npy"),
    cluster_questions_path=os.path.join(DATA_DIR, "representative_train_questions_custom_encoder.csv")
)


queries = [
    "What is The Mother 's Name on ' How I Met Your Mother ' ?",
    "What does DNA stand for?",
    "Who wrote Harry Potter?",
    "When did World War II end?"
]

results = []
timings = []

for query in queries:
    print(f"\nВопрос: {query}")
    for mode in ["clustered", "full"]:
        start = time.time()
        top_k = model.retrieve(query, top_k=1, mode=mode)
        elapsed = time.time() - start
        timings.append({"query": query, "mode": mode, "time_sec": elapsed})

        for res in top_k:
            results.append({
                "query": query,
                "mode": mode,
                "rank": res["rank"],
                "question": res["question"],
                "answer": res["answer"],
                "similarity": res["score"]
            })


df = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nРезультаты сохранены в: {OUTPUT_CSV}")


plt.figure(figsize=(10, 5))
sns.barplot(data=df[df["rank"] == 1], x="query", y="similarity", hue="mode", palette="pastel")
plt.title("Сходство топ-1 ответа: Full vs Clustered (Custom Encoder)")
plt.ylabel("Cosine Similarity")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOT_SIM_PATH, dpi=200)
print(f"График сходства сохранён в: {PLOT_SIM_PATH}")

# === Визуализация: Время === #
df_time = pd.DataFrame(timings)
plt.figure(figsize=(10, 5))
sns.barplot(data=df_time, x="query", y="time_sec", hue="mode", palette="Set2")
plt.title("Время выполнения поиска: Full vs Clustered (Custom Encoder)")
plt.ylabel("Seconds")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOT_TIME_PATH, dpi=200)
print(f"График времени сохранён в: {PLOT_TIME_PATH}")


avg_times = df_time.groupby("mode")["time_sec"].mean().reset_index()

plt.figure(figsize=(6, 4))
sns.barplot(data=avg_times, x="mode", y="time_sec", hue="mode", palette="muted", legend=False)
plt.title("Average Search Time in Small-Scale Evaluation")
plt.ylabel("Average Time per Query (sec.)")
plt.xlabel("Search Mode")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(PLOT_AVG_LATENCY_PATH, dpi=200)
plt.show()
print(f"График среднего времени поиска сохранён в: {PLOT_AVG_LATENCY_PATH}")
