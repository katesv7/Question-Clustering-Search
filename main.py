from src import eda, augmentation, embeddings, clustering, representatives, search_simulation


RAW_PATH = "data/raw/small_nq_new.jsonl"
CLEANED_PATH = "data/processed/questions_answers_clean.csv"
AUGMENTED_PATH = "data/processed/questions_answers_augmented.csv"
EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
TEXTS_PATH = "data/processed/question_texts.txt"
ANSWER_MAP_PATH = "data/processed/answer_map.csv"
CLUSTERED_PATH = "data/processed/clustered_questions.csv"
REPS_PATH = "data/processed/representative_questions.csv"
REP_EMBEDDINGS_PATH = "data/processed/representative_embeddings.npy"
REP_TEXTS_PATH = "data/processed/representative_questions.txt"
MODEL_NAME = "intfloat/multilingual-e5-base"


def step_eda():
    eda.run(
        input_path=RAW_PATH,
        output_path=CLEANED_PATH,
        max_rows=None
    )


def step_augmentation():
    augmentation.run(
        input_path=CLEANED_PATH,
        output_path=AUGMENTED_PATH,
        only_multi=True,
        top_k=3
    )


def step_embeddings():
    embeddings.run(
        input_path=AUGMENTED_PATH,
        emb_out_path=EMBEDDINGS_PATH,
        txt_out_path=TEXTS_PATH,
        map_out_path=ANSWER_MAP_PATH,
        model_name=MODEL_NAME
    )


def step_clustering():
    clustering.run_on_local(
        embeddings_path=EMBEDDINGS_PATH,
        questions_path=TEXTS_PATH,
        output_path=CLUSTERED_PATH,
        method="dbscan",
        eps=0.2,
        min_samples=5,
        xi=0.05,
        min_cluster_size=5
    )


def step_representatives():
    representatives.run_on_local(
        cluster_file=CLUSTERED_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        output_file=REPS_PATH,
        rep_embeddings_out=REP_EMBEDDINGS_PATH,
        rep_text_out=REP_TEXTS_PATH,
        center_method="mean",
        top_k=1,
        plot=True
    )


def step_simulation():
    search_simulation.run_on_local(
        embeddings_path=EMBEDDINGS_PATH,
        cluster_file=CLUSTERED_PATH,
        reps_file=REPS_PATH,
        top_k=5
    )


if __name__ == "__main__":
    print("Запускаем все шаги пайплайна\n")

    step_eda()
    step_augmentation()
    step_embeddings()
    step_clustering()
    step_representatives()
    step_simulation()

    print("\nВсе шаги успешно выполнены.")
