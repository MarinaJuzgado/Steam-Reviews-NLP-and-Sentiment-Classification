# Steam-Reviews-NLP-and-Sentiment-Classification

This project applies **Natural Language Processing and Machine Learning** to a dataset of Steam game reviews, covering the full pipeline from text preprocessing and vectorization to sentiment classification and recommendation systems. The dataset was entirely built by the group using a custom web scraping script.

The project was developed for the **Machine Learning Applications** course at Universidad Carlos III de Madrid (UC3M).

---

## Project Objective

The goal is to transform raw Steam review text into structured numerical representations and use them to solve two downstream tasks:

- **Task 1 — NLP and Text Vectorization:** Build a preprocessing pipeline and compare four families of document representations: classical (BoW/TF-IDF), embedding-based (Word2Vec), topic modeling (LDA), and contextual embeddings (BERT).
- **Task 2.1 — Sentiment Classification:** Predict whether a review is positive or negative using the vectorizations from Task 1, with feature extraction/selection analysis.
- **Task 2.3 — Recommender Systems:** Build a content-based recommendation system using document vectors, and compare it with collaborative filtering approaches via the Surprise library.

---

## Dataset

The dataset was built entirely by the group using a **self-developed Python web scraping script** that collected publicly available user reviews from the Steam platform (`store.steampowered.com`), owned by Valve Corporation. It contains **12,826 reviews** and 12 variables including review text, sentiment label (`voted_up`), language, game title, and playtime.

No third-party datasets or pre-collected corpora were used.

> **Note:** The raw dataset is not included in this repository as it was collected from a third-party platform. To reproduce the data collection step, run the scraping script provided in the notebook.

---

## Methodology

### Task 1 — Preprocessing and Vectorization

**Preprocessing pipeline:**
- HTML removal (BeautifulSoup), URL removal, contraction expansion, non-ASCII filtering, lowercasing.
- Linguistic processing with **spaCy** (`en_core_web_md`): POS filtering (NOUN, PROPN, VERB, ADJ), stopword removal, and lemmatization.

**Vectorization strategies:**
- **BoW / TF-IDF:** Gensim dictionary with vocabulary filtering (min/max frequency thresholds), resulting in a 76% vocabulary reduction to 4,376 terms.
- **Word2Vec:** Custom Skip-gram model (200 dimensions) trained on the corpus to capture gaming-specific semantics ("boss" → "fight", "tough"; "bug" → "glitch"). Validated with t-SNE visualization.
- **LDA Topic Modeling:** tomotopy library, k=10 topics (justified by Cv coherence and Jaccard stability sweep over k∈{5,10,15,20,25,50}), with N-gram detection (Gensim) and rm_top=20 filtering. Visualized with pyLDAvis.
- **BERT:** `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional L2-normalized sentence embeddings. Validated via pairwise cosine similarity analysis and PCA projections.

### Task 2.1 — Sentiment Classification

- **Problem:** Binary classification (positive / negative) on a strongly imbalanced corpus (~84% positive / 16% negative, ratio 5.4:1).
- **Protocol:** 5-fold Stratified Cross-Validation + GridSearchCV for hyperparameter tuning + held-out 20% test split.
- **Feature extraction:** TruncatedSVD (LSA) on TF-IDF, SelectKBest (mutual information) on TF-IDF, PCA on BERT.
- **Classifiers:** Logistic Regression, calibrated Linear SVM, Random Forest — across all 4 vectorizations.
- **Best model:** BERT + Linear SVM (C=0.1), F1-macro = 0.75, ROC-AUC = 0.917 on the test set.

### Task 2.3 — Recommender Systems

- **Content-based:** Each game represented as the centroid of its review vectors; recommendations ranked by cosine similarity. Evaluated with leave-one-out Hit-Rate@5. Best: BERT (Hit-Rate@5 = 0.3350).
- **Collaborative filtering (Surprise):** BaselineOnly, KNNBasic/WithMeans/WithZScore, SVD (biased ALS), NMF — tuned with 3-fold GridSearchCV. Best RMSE: BaselineOnly (RMSE = 1.2703), with SVD as the strongest latent-factor model.

---

## Results Summary

| Vectorization | Best F1-macro | Best ROC-AUC | Hit-Rate@5 |
|---|---|---|---|
| BERT | 0.7642 | 0.9174 | 0.3350 |
| TF-IDF | 0.7534 | 0.8935 | — |
| BoW | 0.7497 | 0.8566 | — |
| TF-IDF + SVD(200) | 0.7042 | 0.8718 | 0.3307 |
| Word2Vec | 0.6781 | 0.8504 | 0.3295 |
| LDA Topics | 0.6047 | 0.7729 | 0.2908 |

BERT consistently leads across both tasks, though the margin over TF-IDF with a linear classifier is narrow (~0.01 F1), confirming that Steam sentiment is largely keyword-driven.

---

## Conclusion

BERT contextual embeddings achieve the best performance across classification and content-based recommendation, but TF-IDF with a linear classifier remains a strong, interpretable, and computationally cheaper alternative. Dimensionality reduction (SelectKBest at k=3,000; PCA at 100 components) recovers 97–99% of full-baseline performance while reducing feature count significantly. The recommended production architecture is a hybrid system: BERT content-based recommendations for cold-start scenarios, switching to SVD collaborative filtering once sufficient interaction history is available.

---

## Repository Contents

- `scraper.py`: Python script used to collect the Steam reviews dataset via web scraping (`store.steampowered.com`).
- `ML_Applications_Project.ipynb`: Full annotated Jupyter notebook with preprocessing pipeline, all vectorizations, classification experiments, and recommender system implementations.
- `report.pdf`: Written project report with full methodology, results, and discussion.

---

## Requirements

- Python 3.x
- spaCy (`en_core_web_md`)
- Gensim, tomotopy, pyLDAvis
- sentence-transformers (`all-MiniLM-L6-v2`)
- scikit-learn, Surprise
- BeautifulSoup, contractions
- NumPy, Pandas, Matplotlib

---

## Authors

- Carla Aranda Sánchez — [@100523031](https://github.com/100523031)
- David Gao
- Marina Juzgado Gómez-Menor
- Iván López Anca — [@ilopezanca](https://github.com/ilopezanca)
