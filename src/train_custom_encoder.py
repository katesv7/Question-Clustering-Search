
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
