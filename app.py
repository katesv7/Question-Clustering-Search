import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import load_embeddings, load_questions, load_model, clean_text

EMBEDDINGS_FULL = "data/processed/question_embeddings.npy"
QUESTIONS_FULL = "data/processed/question_texts.txt"
ANSWERS = "data/processed/answer_map.csv"

EMBEDDINGS_CLUSTERED = "data/processed/representative_embeddings.npy"
QUESTIONS_CLUSTERED = "data/processed/representative_questions.txt"

MODEL_NAME = "intfloat/multilingual-e5-base"

st.set_page_config(page_title="Clustered Question Search", layout="centered")
st.title("🔍 Поиск по кластеризованным вопросам")
st.markdown("Введите вопрос, и модель найдёт похожие формулировки и ответы.")

query = st.text_input("Ваш вопрос", placeholder="Например: кто изобрёл лампочку?")
mode = st.radio("Режим поиска", ["clustered (быстрый)", "full (точный)"])
k = st.slider("Сколько результатов показать?", 1, 10, 3)

if st.button("Найти") and query:
    st.write("Идёт поиск...")

    model = load_model(MODEL_NAME)
    query_vec = model.encode([clean_text(query)], normalize_embeddings=True)

    if "clustered" in mode:
        embeddings = load_embeddings(EMBEDDINGS_CLUSTERED)
        questions = load_questions(QUESTIONS_CLUSTERED)
    else:
        embeddings = load_embeddings(EMBEDDINGS_FULL)
        questions = load_questions(QUESTIONS_FULL)

    df_answers = pd.read_csv(ANSWERS)

    sims = cosine_similarity(query_vec, embeddings)[0]
    top_idx = np.argsort(sims)[-k:][::-1]

    st.success(f"Найдено {k} совпадений")
    for rank, idx in enumerate(top_idx, 1):
        q = questions[idx]
        row = df_answers[df_answers["question"] == q]
        a = row["answer"].iloc[0] if not row.empty else "ответ не найден"

        st.markdown(f"""
        **{rank}. {q}**  
        _{a}_  
        Сходство: `{sims[idx]:.4f}`
        """)
