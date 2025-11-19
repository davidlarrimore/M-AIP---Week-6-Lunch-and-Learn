🎯 Overall Goal

Update the Sentiment Analysis & Topic Modeling page so that:
	1.	It uses only traditional NLP models (no transformers, no LLMs).
	2.	All training and inference happen in the Streamlit app runtime (nothing offline).
	3.	The UI clearly contrasts:
	•	“Realtime NLP (traditional)” – fast, local, classical models
	•	“LLM” – slower, external, generative (on your LLM page)

VADER and LDA stay as baselines, but we add higher-accuracy classical models that are still fast enough to train and run live.

⸻

1. Sentiment Analysis – Realtime Traditional Models

1.1 Keep existing VADER sentiment

No change required:
	•	Keep analyze_sentiment_vader(...) and any existing batch logic.
	•	This is your “lexicon-based baseline”, very fast and fully traditional.

⸻

1.2 Add an in-app trained ML sentiment model (TF-IDF + Linear Classifier)

Key idea:
Train a simple supervised model inside the Streamlit app, using the same dataset the user loads (or a built-in demo dataset). Cache it so training happens only once per session, but it’s still clearly “live, in runtime.”

A. Create labels on the fly
Assumption: your demo dataset has a rating-like field (e.g., star_rating or similar).
	1.	When the dataset is loaded in the app (you already do this), create binary labels:

def build_sentiment_labels_from_df(df):
    # Example rule – adjust to your schema:
    # rating >= 4 → positive (1), rating <= 2 → negative (0), drop neutrals
    labeled = df.copy()
    labeled = labeled[labeled["rating"].isin([1, 2, 4, 5])]  # drop 3-star neutrals

    labeled["label"] = labeled["rating"].apply(
        lambda r: 1 if r >= 4 else 0
    )

    texts = labeled["review"].tolist()
    labels = labeled["label"].tolist()
    return texts, labels


	2.	If ratings don’t exist, devs can:
	•	Use an included labeled dataset, or
	•	Add a small example CSV with a label column.

⸻

B. Train the model in runtime (and cache it)
	1.	Add imports:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# or: from sklearn.svm import LinearSVC


	2.	Create a cached training function:

import hashlib
import time
import streamlit as st

@st.cache_resource
def train_ml_sentiment_model(texts, labels):
    # Optional: sample to keep training snappy
    max_samples = 2000
    if len(texts) > max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=20000
    )
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000)
    # or LinearSVC(), if you don’t need probabilities
    clf.fit(X, labels)

    return vectorizer, clf


	3.	When the dataset is loaded:

texts, labels = build_sentiment_labels_from_df(df)
vectorizer, clf = train_ml_sentiment_model(texts, labels)

	•	This training runs once per app session (thanks to @st.cache_resource), but it’s clearly happening inside the Streamlit runtime.
	•	Training on a subset (e.g., 2k samples) will be very fast on a MacBook.

⸻

C. Add a per-review ML sentiment function

def analyze_sentiment_ml(review_text: str, vectorizer, clf) -> dict:
    start_time = time.time()

    X = vectorizer.transform([review_text])
    pred = clf.predict(X)[0]

    if hasattr(clf, "predict_proba"):
        confidence = float(clf.predict_proba(X)[0].max())
    else:
        confidence = 1.0  # or derive from decision_function if desired

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "sentiment": "positive" if pred == 1 else "negative",
        "confidence": confidence,
        "elapsed_ms": elapsed_ms,
    }

For batch sentiment (if you show it), just loop over reviews inside the app using this function—no external batch job.

⸻

D. UI: let users switch models and see speed
On the Sentiment page:

sentiment_model_choice = st.radio(
    "Sentiment model (traditional NLP)",
    ["VADER (lexicon-based)", "TF-IDF + Linear Model (trained live)"]
)

Then, when analyzing a review:

if sentiment_model_choice == "VADER (lexicon-based)":
    result = analyze_sentiment_vader(review_text)
else:
    result = analyze_sentiment_ml(review_text, vectorizer, clf)

st.metric("Sentiment", result["sentiment"])
st.metric("Confidence", f"{result['confidence']:.2f}")
st.metric("Latency (ms)", f"{result['elapsed_ms']:.1f}")

This makes the “realtime NLP vs LLM” story very clear: both models run live; both are fast; later you compare to the LLM page.

⸻

2. Topic Modeling – Realtime Classical Methods

All topic modeling work stays inside Streamlit, too.

2.1 Keep LDA as the baseline
	•	Keep your existing extract_topics_lda(...) implementation.
	•	This is your classical probabilistic topic model baseline.

⸻

2.2 Add NMF (matrix factorization) as an alternative topic model

Goal: Show that a different traditional method (NMF + TF-IDF) can be trained and run in realtime on the same data.
	1.	Add import:

from sklearn.decomposition import NMF


	2.	Implement extract_topics_nmf(reviews, num_topics, words_per_topic):

def extract_topics_nmf(reviews, num_topics: int, words_per_topic: int = 5) -> dict:
    start_time = time.time()

    docs = [r["review"] for r in reviews]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )
    X = vectorizer.fit_transform(docs)

    nmf = NMF(n_components=num_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic_vec in enumerate(H):
        top_indices = topic_vec.argsort()[:-words_per_topic - 1:-1]
        words = [terms[i] for i in top_indices]
        weights = [float(topic_vec[i]) for i in top_indices]

        topics.append({
            "id": topic_idx,
            "words": words,
            "weights": weights,
        })

    elapsed_ms = (time.time() - start_time) * 1000
    return {
        "topics": topics,
        "elapsed_ms": elapsed_ms,
    }



	•	This trains NMF on the fly on the current dataset.
	•	You can optionally cache the result with @st.cache_resource keyed on dataset hash + num_topics.

⸻

2.3 UI: switch between LDA and NMF and show timing

topic_model_choice = st.radio(
    "Topic modeling method (traditional NLP)",
    ["LDA (probabilistic)", "NMF (matrix factorization)"]
)

if topic_model_choice == "LDA (probabilistic)":
    topic_result = extract_topics_lda(reviews, num_topics, words_per_topic)
else:
    topic_result = extract_topics_nmf(reviews, num_topics, words_per_topic)

st.metric("Topic modeling latency (ms)", f"{topic_result['elapsed_ms']:.1f}")
# Then render topics as you already do

Again: all computation is inside the Streamlit runtime, on demand.

⸻

3. Single-Review “Topics” / Aspects – Better Phrases, Still Realtime

3.1 Replace raw word frequency with RAKE/YAKE for a single review

Goal: For one selected review, we want fast, on-the-fly keyphrase extraction that’s still traditional (no neural models).
	1.	Add dependency and import (example with RAKE):

# pip install rake-nltk
from rake_nltk import Rake


	2.	Update extract_topics_from_single_review(review_text):

def extract_topics_from_single_review(review_text: str) -> dict:
    start_time = time.time()

    rake = Rake()
    rake.extract_keywords_from_text(review_text)
    ranked_phrases = rake.get_ranked_phrases()  # already sorted by importance

    top_phrases = ranked_phrases[:5]

    # Existing logic:
    # - map phrases to aspects (price, quality, shipping, etc.)
    # - run VADER on sentences/phrases per aspect
    aspect_sentiments = compute_aspect_sentiments(review_text, top_phrases)

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "top_phrases": top_phrases,
        "aspect_sentiments": aspect_sentiments,
        "elapsed_ms": elapsed_ms,
    }


	3.	Show this in the UI for a single selected review, including latency.

All of this is instant on a MacBook and very clearly “realtime NLP.”

⸻

4. How This Supports the “Realtime NLP vs LLM” Story

With these changes, devs will enable you to demonstrate:
	•	Traditional Realtime NLP
	•	VADER and TF-IDF + Linear model: trained and run inside the app, sub-second latency.
	•	LDA and NMF topics: trained live on the current dataset, with visible latency metrics.
	•	RAKE/YAKE per-review phrases: realtime extraction and aspect sentiment.
	•	LLM Side (on your other page)
	•	Same text, but:
	•	Sent through an LLM endpoint.
	•	Higher latency and different cost profile.
	•	More “intelligent” generation, but clearly slower / more resource-intensive.

The dev work is mostly:
	•	Adding one runtime-trained sentiment model.
	•	Adding one NMF topic model.
	•	Swapping single-review frequency counts for RAKE.
	•	Adding a few radio buttons + latency metrics.