import numpy as np
from clustered_retriever import ClusteredRetriever
from utils import clean_text

MODEL_NAME = "intfloat/multilingual-e5-base"
FULL_EMB = "data/processed/question_embeddings.npy"
FULL_QS = "data/processed/question_texts.txt"
ANSWERS = "data/processed/answer_map.csv"
CLUSTER_EMB = "data/processed/representative_embeddings.npy"
CLUSTER_QS = "data/processed/representative_questions.txt"

def test_clean_text():
    text = "  <b>Hello</b>  \n World!  "
    cleaned = clean_text(text)
    assert cleaned == "hello world!", f"Ожидали 'hello world!', получили: {cleaned}"
    print("test_clean_text passed")


def test_model_load():
    model = ClusteredRetriever(
        model_name=MODEL_NAME,
        full_embeddings_path=FULL_EMB,
        full_questions_path=FULL_QS,
        answer_map_path=ANSWERS
    )
    assert hasattr(model, "retrieve"), "У модели нет метода retrieve()"
    print("test_model_load passed")


def test_encode_shape():
    model = ClusteredRetriever(
        model_name=MODEL_NAME,
        full_embeddings_path=FULL_EMB,
        full_questions_path=FULL_QS,
        answer_map_path=ANSWERS
    )
    vec = model.encode("что такое ESG?")
    assert vec.shape == (1, 768), f"Неверная размерность эмбеддинга: {vec.shape}"
    print("test_encode_shape passed")


def test_retrieve_full():
    model = ClusteredRetriever(
        model_name=MODEL_NAME,
        full_embeddings_path=FULL_EMB,
        full_questions_path=FULL_QS,
        answer_map_path=ANSWERS
    )
    results = model.retrieve("что такое устойчивое развитие?", top_k=3, mode="full")
    assert len(results) == 3, "Неверное количество результатов"
    assert "question" in results[0] and "answer" in results[0], "Отсутствуют ключи"
    print("test_retrieve_full passed")


def test_retrieve_clustered():
    model = ClusteredRetriever(
        model_name=MODEL_NAME,
        full_embeddings_path=FULL_EMB,
        full_questions_path=FULL_QS,
        answer_map_path=ANSWERS,
        cluster_embeddings_path=CLUSTER_EMB,
        cluster_questions_path=CLUSTER_QS
    )
    results = model.retrieve("кто изобрёл лампочку?", top_k=3, mode="clustered")
    assert len(results) == 3, "Неверное количество результатов"
    print("test_retrieve_clustered passed")


def test_clustered_fallback_error():
    model = ClusteredRetriever(
        model_name=MODEL_NAME,
        full_embeddings_path=FULL_EMB,
        full_questions_path=FULL_QS,
        answer_map_path=ANSWERS
    )
    try:
        model.retrieve("любой текст", mode="clustered")
    except ValueError as e:
        assert "кластерные эмбеддинги" in str(e).lower()
        print("test_clustered_fallback_error passed")
        return
    assert False, "Ожидалась ошибка, но она не произошла"


if __name__ == "__main__":
    print("Запуск unit-тестов...\n")
    test_clean_text()
    test_model_load()
    test_encode_shape()
    test_retrieve_full()
    test_retrieve_clustered()
    test_clustered_fallback_error()
    print("\nВсе тесты прошли успешно!")
