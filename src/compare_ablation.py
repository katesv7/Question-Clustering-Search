import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation_utils import evaluate_search
from clustered_retriever import ClusteredRetriever
from baselines.tfidf_baseline import TFIDFBaseline
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_QUESTIONS_PATH = os.path.join(DATA_DIR, "test_questions.csv")
ANSWER_MAP_PATH = os.path.join(DATA_DIR, "answer_map.csv")
TRAIN_Q_PATH = os.path.join(DATA_DIR, "train_questions.csv")
TRAIN_EMB_PATH = os.path.join(DATA_DIR, "train_embeddings.npy")
REPR_Q_PATH = os.path.join(DATA_DIR, "representative_train_questions.csv")
REPR_EMB_PATH = os.path.join(DATA_DIR, "representative_train_embeddings.npy")


df_test = pd.read_csv(TEST_QUESTIONS_PATH)
test_questions = df_test["question"].dropna().tolist()
df_map = pd.read_csv(ANSWER_MAP_PATH)
answer_map = dict(zip(df_map["question"].str.strip().str.lower(), df_map["answer"].str.strip().str.lower()))


final_model = ClusteredRetriever(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    full_embeddings_path=TRAIN_EMB_PATH,
    full_questions_path=TRAIN_Q_PATH,
    answer_map_path=ANSWER_MAP_PATH,
    cluster_embeddings_path=REPR_EMB_PATH,
    cluster_questions_path=REPR_Q_PATH
)


experiments = {
    "Final": lambda: final_model.retrieve,
    "No_Clustering": lambda: lambda q, top_k=10: final_model.retrieve(q, top_k=top_k, mode="full"),
    "TF-IDF": lambda: TFIDFBaseline(TRAIN_Q_PATH, ANSWER_MAP_PATH).retrieve,
    "Weak_Encoder": lambda: ClusteredRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        full_embeddings_path=TRAIN_EMB_PATH,
        full_questions_path=TRAIN_Q_PATH,
        answer_map_path=ANSWER_MAP_PATH,
        cluster_embeddings_path=REPR_EMB_PATH,
        cluster_questions_path=REPR_Q_PATH
    ).retrieve
}

def create_mean_center_model():
    centers_path = os.path.join(DATA_DIR, "cluster_centers.npy")
    dummy_questions_path = os.path.join(DATA_DIR, "cluster_center_names.txt")

    if not os.path.exists(dummy_questions_path):
        centers = np.load(centers_path)
        dummy_names = [f"center_{i}" for i in range(len(centers))]
        pd.DataFrame({"question": dummy_names}).to_csv(dummy_questions_path, index=False)

    return ClusteredRetriever(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        full_embeddings_path=TRAIN_EMB_PATH,
        full_questions_path=TRAIN_Q_PATH,
        answer_map_path=ANSWER_MAP_PATH,
        cluster_embeddings_path=centers_path,
        cluster_questions_path=dummy_questions_path
    ).retrieve

experiments["Mean_Center"] = create_mean_center_model


def create_random_center_model():
    df_train = pd.read_csv(TRAIN_Q_PATH)
    random_questions = df_train["question"].sample(n=100, random_state=42).tolist()

    random_txt = os.path.join(DATA_DIR, "random_centers.csv")
    random_emb = os.path.join(DATA_DIR, "random_centers.npy")

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    embeddings = model.encode(random_questions, normalize_embeddings=True)
    np.save(random_emb, embeddings)

    pd.DataFrame({"question": random_questions}).to_csv(random_txt, index=False)

    return ClusteredRetriever(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        full_embeddings_path=TRAIN_EMB_PATH,
        full_questions_path=TRAIN_Q_PATH,
        answer_map_path=ANSWER_MAP_PATH,
        cluster_embeddings_path=random_emb,
        cluster_questions_path=random_txt
    ).retrieve

experiments["Random_Centers"] = create_random_center_model
results = []

for name, fn in experiments.items():
    print(f"\nОцениваем вариант: {name}")
    search_fn = fn()
    summary = evaluate_search(
        queries=test_questions,
        search_fn=search_fn,
        top_k=10,
        true_answer_map=answer_map,
        return_details=False
    )
    summary["variant"] = name
    results.append(summary)

df = pd.DataFrame(results)
final_row = df[df["variant"] == "Final"].iloc[0]
deltas = []

for _, row in df.iterrows():
    if row["variant"] == "Final":
        continue
    delta = {
        "variant": f"Δ vs Final ({row['variant']})",
        "precision@1": row["precision@1"] - final_row["precision@1"],
        "recall@10": row["recall@10"] - final_row["recall@10"],
        "mrr@10": row["mrr@10"] - final_row["mrr@10"],
        "ndcg@10": row["ndcg@10"] - final_row["ndcg@10"],
        "avg_latency_sec": row["avg_latency_sec"] - final_row["avg_latency_sec"],
        "count": row["count"]
    }
    deltas.append(delta)

df_delta = pd.DataFrame(deltas)
df_all = pd.concat([df, df_delta], ignore_index=True)
csv_path = os.path.join(DATA_DIR, "ablation_results.csv")
df_all.to_csv(csv_path, index=False)
print(f"CSV сохранён: {csv_path}")
df_delta_only = df_all[df_all["variant"].str.startswith("Δ vs Final")].copy()
delta_csv_path = os.path.join(DATA_DIR, "ablation_deltas_only.csv")
df_delta_only.to_csv(delta_csv_path, index=False)
print(f"CSV с дельтами сохранён: {delta_csv_path}")

df = pd.DataFrame(results)
csv_path = os.path.join(DATA_DIR, "ablation_results.csv")
df_all.to_csv(csv_path, index=False)
print(f"CSV сохранён: {csv_path}")


plt.figure(figsize=(10, 5))
df_melted = df.melt(id_vars="variant", value_vars=["precision@1", "recall@10", "mrr@10", "ndcg@10"], var_name="metric")
sns.barplot(data=df_melted, x="metric", y="value", hue="variant", palette="Set2")
plt.title("Абляционное сравнение компонентов пайплайна")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(DATA_DIR, "ablation_results_plot.png")
plt.savefig(plot_path, dpi=200)
print(f"График сохранён: {plot_path}")


plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="variant", y="avg_latency_sec", palette="pastel")
plt.title("Среднее время отклика (latency) по вариантам")
plt.ylabel("Среднее время (сек)")
plt.xlabel("Вариант")
plt.grid(True)
plt.tight_layout()
latency_path = os.path.join(DATA_DIR, "ablation_latency_plot.png")
plt.savefig(latency_path, dpi=200)
print(f"График latency сохранён: {latency_path}")
