import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from utils import clean_text
from collections import defaultdict

ENCODERS = {
    "e5-base": "intfloat/multilingual-e5-base",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-m3": "BAAI/bge-m3"
}


df = pd.read_csv("data/processed/questions_answers_clean.csv")
df = df.drop_duplicates()


counts = df.groupby("answer")["question"].count()
valid_answers = counts[counts >= 2].index
df_filtered = df[df["answer"].isin(valid_answers)]

print(f"Используем {df_filtered.shape[0]} вопросов с {len(valid_answers)} ответами")


answer2questions = defaultdict(list)
for _, row in df_filtered.iterrows():
    answer2questions[row["answer"]].append(clean_text(row["question"]))


results = []

for label, model_name in ENCODERS.items():
    print(f"\nТестируем: {label}")
    model = SentenceTransformer(model_name)

    all_questions = []
    answer_labels = []

    for i, (answer, qlist) in enumerate(answer2questions.items()):
        for q in qlist:
            all_questions.append(q)
            answer_labels.append(i)

    t0 = time.time()
    embeddings = model.encode(all_questions, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    t1 = time.time()

    sim_within = []
    sim_between = []

    for i, (answer, qlist) in enumerate(answer2questions.items()):
        if len(qlist) < 2:
            continue
        embs = model.encode(qlist, normalize_embeddings=True)
        sims = cosine_similarity(embs)
        upper_tri = sims[np.triu_indices(len(embs), k=1)]
        sim_within.extend(upper_tri)


    rng = np.random.default_rng(42)
    for _ in range(1000):
        i, j = rng.integers(0, len(embeddings), 2)
        if answer_labels[i] != answer_labels[j]:
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            sim_between.append(sim)


    sil = silhouette_score(embeddings, answer_labels)

    results.append({
        "encoder": label,
        "avg_within_similarity": np.mean(sim_within),
        "avg_between_similarity": np.mean(sim_between),
        "silhouette_score": sil,
        "encoding_time_sec": t1 - t0
    })


df_results = pd.DataFrame(results).set_index("encoder")
df_results = df_results.round(4)
print("\nСравнение энкодеров:")
print(df_results)

df_results.to_csv("experiments/encoder_similarity_comparison.csv")
