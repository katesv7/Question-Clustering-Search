# Question-Clustering-Search
# 🧠 Question Clustering for Symmetric Semantic Search

This project explores the task of clustering semantically similar **natural language questions** to optimize **symmetric question-to-question search**. The goal is to speed up search and reduce computation while maintaining high retrieval quality.

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python3-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Framework-SentenceTransformers-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Clustering-DBSCAN-green?style=flat-square"/>
</p>

---

## 💡 Problem Statement

Symmetric search (e.g., "Which question is similar to mine?") is useful in:
- FAQ systems
- Customer support bots
- Educational platforms
- Duplicate detection (e.g., Stack Overflow)

However, pairwise comparison over all questions is computationally expensive.

---

## 🚀 Solution Overview

I propose a clustering-based retrieval system:

1. **Semantic Encoding**  
   Use pretrained transformers (`e5-base`, `BGE`, `RuBERT`) to encode questions.

2. **Clustering**  
   Cluster questions using **DBSCAN** or **OPTICS** to group similar ones.

3. **Cluster Representatives**  
   Select a central question for each cluster as its representative.

4. **Fast Search**  
   For a new query:
   - Find the nearest **cluster center**
   - Search only **within that cluster** — reducing computations up to **10–50×**

---

## 📈 Evaluation Metrics

I evaluate the performance of the clustered retrieval system compared to traditional symmetric search using the following metrics:

| **Metric**            | **Description**                                                                 |
|-----------------------|---------------------------------------------------------------------------------|
| `cosine_similarity`   | Measures semantic closeness between the query and retrieved questions          |
| `score_diff`          | Difference in similarity between full pairwise search and clustered search     |
| `Recall@k`            | (Optional) Checks if the full-search top-k results contain the clustered result |
| `Execution Time`      | Average time taken per query for full vs. clustered retrieval                  |

**Highlights:**
- 📊 Average `score_diff` between full vs clustered search: **≈ 0.07**
- ⚡ Speedup: **10× to 50× faster** retrieval using clusters
- 🧠 Semantic quality is preserved despite reduced computations

