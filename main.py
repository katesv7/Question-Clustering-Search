from clustered_retriever import ClusteredRetriever

model = ClusteredRetriever(
    model_name="intfloat/multilingual-e5-base",
    full_embeddings_path="data/processed/question_embeddings.npy",
    full_questions_path="data/processed/question_texts.txt",
    answer_map_path="data/processed/answer_map.csv",
    cluster_embeddings_path="data/processed/representative_embeddings.npy",
    cluster_questions_path="data/processed/representative_questions.txt"
)

print("ClusteredRetriever запущен!")
while True:
    query = input("\nВведите вопрос (или 'exit'): ").strip()
    if query.lower() == "exit":
        break

    mode = input("Режим поиска ('full' или 'clustered') [clustered]: ").strip()
    mode = mode if mode in ["full", "clustered"] else "clustered"

    results = model.retrieve(query, top_k=5, mode=mode)

    print(f"\nТоп-5 результатов ({mode}):\n")
    for r in results:
        print(f"{r['rank']}. {r['question']}")
        print(f"   {r['answer']}")
        print(f"   Сходство: {r['similarity']:.4f}\n")
