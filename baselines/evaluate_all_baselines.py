import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from baselines.tfidf_baseline import TFIDFBaseline
from baselines.bm25_baseline import BM25Baseline
from baselines.semantic_baseline import SemanticBaseline
from clustered_retriever import ClusteredRetriever
from src.evaluation_utils import evaluate_search
from src.representatives import select_representatives

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(PLOT_DIR, exist_ok=True)

print("Объединяем train и test эмбеддинги и вопросы")
train_emb = np.load(os.path.join(DATA_DIR, "train_embeddings.npy"))
test_emb = np.load(os.path.join(DATA_DIR, "test_embeddings.npy"))
all_emb = np.vstack([train_emb, test_emb])
np.save(os.path.join(DATA_DIR, "question_embeddings.npy"), all_emb)

df_train = pd.read_csv(os.path.join(DATA_DIR, "train_questions.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "test_questions.csv"))
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all.to_csv(os.path.join(DATA_DIR, "question_texts.csv"), index=False, encoding="utf-8-sig")
print("Готово: question_embeddings.npy и question_texts.csv созданы.")

print("Генерируем representative embeddings и вопросы (split='combined')")
select_representatives(split="combined", center_method="mean", top_k=1)
print("Репрезентативные центры кластеров созданы.")

CSV_PATH = os.path.join(DATA_DIR, "questions_answers_clean.csv")
ANSWERS_PATH = os.path.join(DATA_DIR, "answer_map.csv")
TRUE_MAP_PATH = os.path.join(DATA_DIR, "true_answer_map.csv")
QUESTION_CSV = os.path.join(DATA_DIR, "question_texts.csv")

true_answer_df = pd.read_csv(TRUE_MAP_PATH)
true_answer_map = {
    q.strip().lower(): a.strip().lower()
    for q, a in zip(true_answer_df["question"], true_answer_df["answer"])
}

queries = list(true_answer_map.keys()) 
results_summary = []
details_list = []

print("TF-IDF:")
tfidf_model = TFIDFBaseline(CSV_PATH, ANSWERS_PATH)
summary, details = evaluate_search(queries, tfidf_model.search, true_answer_map=true_answer_map, return_details=True)
summary["method"] = "TF-IDF"
details["method"] = "TF-IDF"
results_summary.append(summary)
details_list.append(details)

print("BM25:")
bm25_model = BM25Baseline(CSV_PATH, ANSWERS_PATH)
summary, details = evaluate_search(queries, bm25_model.retrieve, true_answer_map=true_answer_map, return_details=True)
summary["method"] = "BM25"
details["method"] = "BM25"
results_summary.append(summary)
details_list.append(details)

print("Semantic:")
semantic_model = SemanticBaseline(
    questions_path=QUESTION_CSV,
    answer_map_path=ANSWERS_PATH,
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
summary, details = evaluate_search(queries, semantic_model.retrieve, true_answer_map=true_answer_map, return_details=True)
summary["method"] = "Semantic"
details["method"] = "Semantic"
results_summary.append(summary)
details_list.append(details)

print("Clustered:")
clustered_model = ClusteredRetriever(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    full_embeddings_path=os.path.join(DATA_DIR, "question_embeddings.npy"),
    full_questions_path=QUESTION_CSV,
    answer_map_path=ANSWERS_PATH,
    cluster_embeddings_path=os.path.join(DATA_DIR, "representative_combined_embeddings.npy"),
    cluster_questions_path=os.path.join(DATA_DIR, "representative_combined_questions.csv")
)
summary, details = evaluate_search(
    queries,
    lambda q, top_k: clustered_model.retrieve(q, top_k=top_k, mode="clustered"),
    true_answer_map=true_answer_map,
    return_details=True
)
summary["method"] = "Clustered"
details["method"] = "Clustered"
results_summary.append(summary)
details_list.append(details)

df_results = pd.concat(details_list, ignore_index=True)
df_summary = pd.DataFrame(results_summary)
df_results.to_csv(os.path.join(DATA_DIR, "baseline_comparison_results.csv"), index=False, encoding="utf-8-sig")
df_summary.to_csv(os.path.join(DATA_DIR, "baseline_comparison_summary.csv"), index=False, encoding="utf-8-sig")
print("Сохранено: baseline_comparison_results.csv и summary.csv")

plt.figure(figsize=(8, 4))
sns.barplot(data=df_summary, x="method", y="precision@1", palette="Set2")
plt.title("Precision@1 по методам")
plt.ylabel("Precision@1")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "precision_at_1_comparison.png"), dpi=200)
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(data=df_summary, x="method", y="avg_latency_sec", palette="Set3")
plt.title("Средняя задержка ответа")
plt.ylabel("Latency (sec.)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "latency_comparison.png"), dpi=200)
plt.show()

query_set = set([q.strip().lower() for q in queries])
answer_keys = set(true_answer_map.keys())

intersection = query_set.intersection(answer_keys)
print(f"Пересечения: {len(intersection)} / {len(queries)}")

if intersection:
    print("Найдены дублирующие вопросы в queries и true_answer_map:")
    for i, q in enumerate(list(intersection)[:10]):
        print(f"  {i+1}. {q}")
else:
    print("Совпадений нет.")
