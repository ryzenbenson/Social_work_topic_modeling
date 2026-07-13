# Topic Modeling and LLM-Based Interpretation Pipeline

This repository presents a structured pipeline for topic modeling on oncology social work notes using **BERTopic**, followed by automated topic labeling with large language models (LLMs). If you use or modify this approach, please cite our work using:

**Ryzen Benson, Swetha Rajkumar, Clodagh Kenny, Michelle Zhao, Ji-Hyun Chang, Theodore Scheel, Lauren Boreta, Julian C Hong, Computational identification of salient cancer care topics and themes in oncology social work notes, JNCI Cancer Spectrum, Volume 10, Issue 4, August 2026, pkag033, https://doi.org/10.1093/jncics/pkag033**

For more details on BERTopic, see the official repository:
https://github.com/MaartenGr/BERTopic

---

## Overview

The workflow is divided into three sequential stages:

1. **Text Embedding and Coherence-Based Model Selection**
2. **Final Topic Model Training and Export**
3. **LLM-Based Topic Labeling and Interpretation**

---

## 1. Embedding and Coherence Optimization

Text data is loaded from a parquet file and converted into sentence embeddings using:

* `sentence-transformers/all-MiniLM-L6-v2`

Dimensionality reduction is performed via:

* Principal Component Analysis (PCA, 97% variance retained)
* L2 normalization

BERTopic is then applied with **KMeans clustering**, and a grid search is conducted over different cluster sizes (`k`). Model performance is evaluated using **C_V coherence**, enabling selection of an optimal number of topics.

**Outputs:**

* `reduced_embeddings.npy`
* Coherence plot (C_V vs. number of clusters)
* Selected optimal `k`

---

## 2. Final Topic Model Training

Using the optimal number of clusters identified in Stage 1, a final BERTopic model is trained with:

* KMeans clustering (replacing default HDBSCAN)
* CountVectorizer (unigrams and bigrams)
* Class-based TF-IDF weighting
* KeyBERT and Maximal Marginal Relevance (MMR) representations

**Outputs:**

* Serialized BERTopic model
* `topic_model_raw.csv` (document-to-topic assignments)
* `topic_model_representative.csv` (topic summaries and representative documents)

---

## 3. LLM-Based Topic Labeling

Each topic is post-processed using LLMs to generate:

* A concise **topic label**
* A short **interpretive description (1–3 sentences)**

Models used:

* **GPT-5 (Azure OpenAI)**
* **LLaMA 3.1 (via Ollama)**
* **Qwen (via Ollama)**

Inputs per topic include:

* Representative sentences
* KeyBERT keywords
* MMR keywords

A constraint mechanism ensures that generated labels remain distinct across topics.

**Outputs:**

* Excel files containing labeled topics for each model

---

## Requirements

Key dependencies include:

* `bertopic`
* `sentence-transformers`
* `scikit-learn`
* `gensim`
* `ollama` (for local LLM inference)
* `openai` (Azure OpenAI client)

---

## Usage

Execute the scripts in the following order:

1. **Coherence_experiments.ipynb:** Generate embeddings and determine optimal number of topics
2. **Final_topic_model.ipynb:** Train final BERTopic model and export results
3. **BERTopic_LLM_interpretation:** Generate LLM-based topic labels and descriptions

---

## Notes

* Update all file paths (`.parquet`, `.csv`, `.xlsx`) as appropriate for your environment
* Configure Azure OpenAI credentials via a `.env` file
* Ensure required Ollama models are installed locally prior to execution

---
