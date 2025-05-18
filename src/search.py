import os
from clustered_retriever import ClusteredRetriever


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


retriever = ClusteredRetriever(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    full_embeddings_path=os.path.join(DATA_DIR, "train_embeddings.npy"),
    full_questions_path=os.path.join(DATA_DIR, "train_questions.csv"),
    answer_map_path=os.path.join(DATA_DIR, "answer_map.csv"),
    cluster_embeddings_path=os.path.join(DATA_DIR, "representative_train_embeddings.npy"),
    cluster_questions_path=os.path.join(DATA_DIR, "representative_train_questions.csv")
)

print("Поиск готов. Введите вопрос (или 'exit').")

while True:
    query = input("\nВопрос: ").strip()
    if query.lower() == "exit":
        break


    full_results = retriever.retrieve(query, top_k=1, mode="full")[0]
    clustered_results = retriever.retrieve(query, top_k=1, mode="clustered")[0]
    print("\n=== Сравнение FULL vs CLUSTERED ===\n")

    print("FULL:")
    print(f"   Вопрос: {full_results['question']}")
    print(f"   Ответ:  {full_results['answer']}")
    print(f"   Сходство: {full_results['score']:.4f}")

    print("\nCLUSTERED:")
    print(f"   Вопрос: {clustered_results['question']}")
    print(f"   Ответ:  {clustered_results['answer']}")
    print(f"   Сходство: {clustered_results['score']:.4f}")

    delta = full_results['score'] - clustered_results['score']
    same_answer = full_results['answer'].strip().lower() == clustered_results['answer'].strip().lower()

    print(f"\nРазница в similarity (full - clustered): {delta:+.4f}")
    print(f"Совпадение ответов: {'Да' if same_answer else 'Нет'}")
