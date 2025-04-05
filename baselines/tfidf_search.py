import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/processed/questions_answers_clean.csv"
df = pd.read_csv(DATA_PATH)

rows = []
for _, row in df.iterrows():
    questions = row["question"].split("; ")
    for q in questions:
        rows.append((q, row["answer"]))

qa_df = pd.DataFrame(rows, columns=["question", "answer"])
print(f"TF-IDF baseline: {len(qa_df)} пар")

vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(qa_df["question"])


def tfidf_search(query, top_k=5):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for i in top_idx:
        results.append({
            "question": qa_df.iloc[i]["question"],
            "answer": qa_df.iloc[i]["answer"],
            "score": float(sims[i])
        })
    return results


if __name__ == "__main__":
    while True:
        query = input("\nВведите вопрос (или 'exit'): ").strip()
        if query.lower() == "exit":
            break
        results = tfidf_search(query)
        print(f"\nРезультаты (TF-IDF):")
        for r in results:
            print(f"— {r['question']}")
            print(f"  {r['answer']}")
            print(f"  {r['score']:.4f}\n")
