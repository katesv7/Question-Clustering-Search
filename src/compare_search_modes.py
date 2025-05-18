import os
import pandas as pd
from evaluation_utils import evaluate_search
from clustered_retriever import ClusteredRetriever
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
QUESTIONS_PATH = os.path.join(DATA_DIR, "test_questions.csv")
PLOT_PATH = os.path.join(DATA_DIR, "search_modes_comparison_plot.png")
ANSWER_MAP_PATH = os.path.join(DATA_DIR, "answer_map.csv")
FULL_EMB = os.path.join(DATA_DIR, "train_embeddings.npy")
CLUST_EMB = os.path.join(DATA_DIR, "representative_train_embeddings.npy")
FULL_Q = os.path.join(DATA_DIR, "train_questions.csv")
CLUST_Q = os.path.join(DATA_DIR, "representative_train_questions.csv")
df_test = pd.read_csv(QUESTIONS_PATH)
test_questions = df_test["question"].dropna().tolist()
df_map = pd.read_csv(ANSWER_MAP_PATH)
answer_map = dict(zip(df_map["question"].str.strip().str.lower(), df_map["answer"].str.strip().str.lower()))


model = ClusteredRetriever(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    full_embeddings_path=FULL_EMB,
    full_questions_path=FULL_Q,
    answer_map_path=ANSWER_MAP_PATH,
    cluster_embeddings_path=CLUST_EMB,
    cluster_questions_path=CLUST_Q
)

results = []

for mode in ["full", "clustered"]:
    print(f"\nОцениваем режим: {mode}")
    search_fn = lambda q, top_k: model.retrieve(q, top_k=10, mode=mode)
    summary = evaluate_search(
        queries=test_questions,
        search_fn=search_fn,
        top_k=10,
        true_answer_map=answer_map,
        return_details=False
    )
    summary["mode"] = mode
    results.append(summary)

df = pd.DataFrame(results)
cols = ["mode", "precision@1", "recall@10", "mrr@10", "ndcg@10", "avg_latency_sec", "count"]
df = df[cols]

CSV_PATH = os.path.join(DATA_DIR, "search_modes_comparison.csv")
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
print(f"\nТаблица сохранена: {CSV_PATH}")
print(df.to_markdown(index=False))



df_plot = df.melt(id_vars="mode",
                  value_vars=["precision@1", "recall@10", "mrr@10", "ndcg@10"],
                  var_name="metric", value_name="value")

plt.figure(figsize=(9, 5))
sns.barplot(data=df_plot, x="metric", y="value", hue="mode", palette="pastel")
plt.title("Сравнение поиска: Full vs Clustered (Top-10)")
plt.ylim(0, 1)
plt.ylabel("Значение метрики")
plt.xlabel("Метрика")
plt.legend(title="Режим")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
print(f"График сравнения сохранён: {PLOT_PATH}")
