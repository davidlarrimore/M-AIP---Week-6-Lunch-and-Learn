"""Sentiment Analysis and Topic Modeling on Amazon Reviews using Traditional NLP."""

import html
import re
import time
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datasets import load_dataset
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)


download_nltk_data()


@st.cache_data
def load_amazon_reviews(num_samples: int = 50, seed: Optional[int] = None) -> List[Dict]:
    """
    Load real Amazon reviews from Hugging Face dataset.
    Uses SetFit/amazon_reviews_multi_en which contains English reviews.
    """
    # Load the dataset (validation split for diverse reviews)
    dataset = load_dataset('SetFit/amazon_reviews_multi_en', split='validation')

    # Convert to pandas for easier manipulation
    df = pd.DataFrame(dataset)

    random_state = seed if seed is not None else None
    final_df = df.sample(n=min(num_samples, len(df)), random_state=random_state)

    # Convert to our format
    reviews = []
    for idx, row in final_df.iterrows():
        # Extract fields from the dataset
        review_text = row.get('text', row.get('review_body', ''))
        rating = int(row.get('label', 0)) + 1  # Convert 0-4 to 1-5

        # Get product title if available
        product = row.get('review_title', row.get('product_title', f'Product #{idx + 1}'))
        if not product or len(product.strip()) == 0:
            product = f'Product #{idx + 1}'

        # Only include reviews with actual text
        if review_text and len(review_text.strip()) > 10:
            reviews.append({
                'id': idx + 1,
                'product': product[:100],  # Limit product name length
                'rating': rating,
                'review': review_text[:500],  # Limit review length for display
            })

    # Return up to num_samples reviews
    return reviews[:num_samples]


@st.cache_resource
def get_vader_analyzer():
    """Initialize and cache VADER sentiment analyzer."""
    return SentimentIntensityAnalyzer()


def analyze_sentiment_vader(review_text: str) -> Dict[str, Any]:
    """
    Use VADER sentiment analysis - a rule-based model optimized for social media text.
    VADER is extremely fast (microseconds) compared to LLMs (seconds).
    """
    start_time = time.time()

    analyzer = get_vader_analyzer()
    scores = analyzer.polarity_scores(review_text)

    # Determine overall sentiment
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Extract key phrases by finding highly polar sentences
    sentences = re.split(r'[.!?]+', review_text)
    key_phrases = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sent_scores = analyzer.polarity_scores(sentence)
            if abs(sent_scores['compound']) > 0.3:  # Significant sentiment
                # Take first few words as key phrase
                words = sentence.split()[:6]
                if len(words) > 0:
                    key_phrases.append(' '.join(words))

    key_phrases = key_phrases[:3]  # Limit to top 3

    # Calculate confidence based on compound score magnitude
    confidence = min(abs(compound), 1.0)

    # Generate reasoning
    if sentiment == "positive":
        reasoning = f"High positive sentiment detected (compound: {compound:.3f}). Positive indicators: {scores['pos']:.2f}, Negative: {scores['neg']:.2f}"
    elif sentiment == "negative":
        reasoning = f"High negative sentiment detected (compound: {compound:.3f}). Negative indicators: {scores['neg']:.2f}, Positive: {scores['pos']:.2f}"
    else:
        reasoning = f"Neutral or mixed sentiment (compound: {compound:.3f}). Balanced positive ({scores['pos']:.2f}) and negative ({scores['neg']:.2f}) language"

    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_phrases": key_phrases,
        "scores": scores,
        "processing_time_ms": elapsed_time
    }


def preprocess_for_topics(text: str) -> List[str]:
    """Preprocess text for topic modeling."""
    # Tokenize
    tokens = word_tokenize(text.lower())

    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [
        word for word in tokens
        if word.isalnum() and word not in stop_words and len(word) > 3
    ]

    return tokens


@st.cache_data
def extract_topics_lda(reviews: List[Dict], num_topics: int = 5, words_per_topic: int = 5) -> Dict[str, Any]:
    """
    Use Latent Dirichlet Allocation (LDA) for topic modeling.
    LDA is a traditional statistical model that's much faster than LLM-based approaches.
    """
    start_time = time.time()

    # Preprocess all reviews
    processed_docs = [preprocess_for_topics(r['review']) for r in reviews]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto'
    )

    # Extract topics
    topics = []
    for idx in range(num_topics):
        topic_words = lda_model.show_topic(idx, topn=words_per_topic)
        topics.append({
            'id': idx,
            'words': [word for word, _ in topic_words],
            'weights': [float(weight) for _, weight in topic_words]
        })

    elapsed_time = (time.time() - start_time) * 1000

    return {
        'topics': topics,
        'model': lda_model,
        'dictionary': dictionary,
        'corpus': corpus,
        'processing_time_ms': elapsed_time
    }


def extract_topics_from_single_review(review_text: str) -> Dict[str, Any]:
    """
    Extract topics from a single review using keyword extraction.
    Uses TF-IDF to find the most important words.
    """
    start_time = time.time()

    # Preprocess
    tokens = preprocess_for_topics(review_text)

    # Simple frequency-based topics for single review
    word_freq = Counter(tokens)
    top_words = word_freq.most_common(5)

    # Map to common aspect categories
    aspect_keywords = {
        'quality': ['quality', 'excellent', 'premium', 'perfect', 'great', 'amazing', 'incredible', 'good', 'solid', 'wonderful', 'superb', 'outstanding'],
        'price': ['price', 'expensive', 'cheap', 'worth', 'value', 'money', 'cost', 'affordable', 'overpriced'],
        'comfort': ['comfortable', 'comfort', 'ergonomic', 'soft', 'cozy'],
        'durability': ['durable', 'sturdy', 'strong', 'broke', 'broken', 'stopped', 'working', 'lasted', 'reliable'],
        'design': ['design', 'looks', 'style', 'appearance', 'beautiful', 'gorgeous', 'attractive', 'ugly'],
        'performance': ['performance', 'works', 'working', 'responsive', 'fast', 'slow', 'effective'],
        'support': ['support', 'service', 'help', 'customer', 'warranty'],
        'shipping': ['shipping', 'delivery', 'arrived', 'package', 'packaging'],
        'size': ['size', 'large', 'small', 'fits', 'fitting', 'dimensions'],
    }

    # Identify aspects mentioned
    aspects = {}
    for aspect, keywords in aspect_keywords.items():
        for word, _ in top_words:
            if word in keywords:
                # Determine sentiment of this aspect using VADER on sentences containing the word
                analyzer = get_vader_analyzer()
                sentences_with_word = [s for s in re.split(r'[.!?]+', review_text) if word in s.lower()]
                if sentences_with_word:
                    aspect_sentiment = analyzer.polarity_scores(' '.join(sentences_with_word))['compound']
                    if aspect_sentiment >= 0.05:
                        aspects[aspect] = 'positive'
                    elif aspect_sentiment <= -0.05:
                        aspects[aspect] = 'negative'
                    else:
                        aspects[aspect] = 'neutral'
                    break

    # Generate summary
    topics_list = [word for word, _ in top_words]
    summary = f"Review focuses on: {', '.join(topics_list[:3])}"

    elapsed_time = (time.time() - start_time) * 1000

    return {
        "topics": topics_list,
        "aspects": aspects,
        "summary": summary,
        "processing_time_ms": elapsed_time
    }


def get_sentiment_color(sentiment: str) -> str:
    """Return color based on sentiment."""
    colors = {
        "positive": "#10b981",  # green
        "negative": "#ef4444",  # red
        "neutral": "#f59e0b",   # yellow
    }
    return colors.get(sentiment.lower(), "#6b7280")


def get_sentiment_emoji(sentiment: str) -> str:
    """Return emoji based on sentiment."""
    emojis = {
        "positive": "😊",
        "negative": "😞",
        "neutral": "😐",
    }
    return emojis.get(sentiment.lower(), "🤔")


def build_review_card_html(review: Dict[str, Any], sentiment_result: Dict[str, Any], topic_result: Dict[str, Any]) -> str:
    """Return HTML for a single review card without leading indentation."""
    sentiment = sentiment_result["sentiment"]
    sentiment_color = get_sentiment_color(sentiment)
    sentiment_emoji = get_sentiment_emoji(sentiment)
    stars = "⭐" * review["rating"] + "☆" * (5 - review["rating"])

    aspects = topic_result.get("aspects", {})
    topics = topic_result.get("topics", [])[:3]

    lines = [
        "<div class='review-card'>",
        f"<div class='review-header' style='border-left-color: {sentiment_color};'>",
        "<div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;'>",
        "<div style='flex: 1;'>",
        f"<strong style='color: #1f2937; font-size: 1rem;'>{html.escape(review['product'])}</strong>",
        "</div>",
        f"<div style='font-size: 1rem;'>{stars}</div>",
        "</div>",
        "<div style='color: #4b5563; line-height: 1.5; font-size: 0.95rem; font-style: italic; margin-bottom: 0.8rem;'>",
        f"\"{html.escape(review['review'])}\"",
        "</div>",
        "<div style='margin-bottom: 0.8rem;'>",
        f"<span class='sentiment-badge' style='background: {sentiment_color};'>",
        f"{sentiment_emoji} {sentiment.upper()}",
        "</span>",
        f"<span class='performance-badge'>⚡ {sentiment_result['processing_time_ms']:.1f}ms</span>",
        "</div>",
    ]

    if topics:
        lines.append("<div style='margin-top: 0.3rem;'>")
        for topic in topics:
            lines.append(f"<span class='topic-chip'>🏷️ {html.escape(topic)}</span>")
        lines.append("</div>")

    if aspects:
        lines.append("<div style='margin-top: 0.5rem;'>")
        for aspect, aspect_sentiment in aspects.items():
            aspect_color = get_sentiment_color(aspect_sentiment)
            lines.append(
                f"<span class='aspect-badge' style='background: {aspect_color}; color: white;'>{html.escape(aspect.title())}</span>"
            )
        lines.append("</div>")

    lines.append(build_analysis_details_html(review, sentiment_result, topic_result))

    lines.extend(
        [
            "</div>",
            "</div>",
            "</div>",
        ]
    )

    return "\n".join(lines)


def build_analysis_details_html(review: Dict[str, Any], sentiment_result: Dict[str, Any], topic_result: Dict[str, Any]) -> str:
    """Return expandable HTML with sentiment/topic details."""
    scores = sentiment_result.get('scores', {})
    key_phrases = sentiment_result.get('key_phrases') or []
    confidence_pct = sentiment_result.get('confidence', 0) * 100
    reasoning = html.escape(sentiment_result.get('reasoning', ''))

    key_phrase_items = "".join(f"<li>{html.escape(phrase)}</li>" for phrase in key_phrases)
    key_phrases_block = (
        f"<div><strong>Key Phrases:</strong><ul class='analysis-list'>{key_phrase_items}</ul></div>"
        if key_phrase_items else ""
    )

    aspects = topic_result.get("aspects", {})
    aspect_lines = []
    if aspects:
        for aspect, aspect_sentiment in aspects.items():
            aspect_color = get_sentiment_color(aspect_sentiment)
            aspect_emoji = get_sentiment_emoji(aspect_sentiment)
            aspect_lines.append(
                f"<div class='aspect-detail' style='border-left-color: {aspect_color};'>"
                f"{html.escape(aspect.title())}: {aspect_emoji} {aspect_sentiment.title()}"
                "</div>"
            )
    else:
        aspect_lines.append("<p style='color: #6b7280;'>No specific product aspects detected.</p>")
    aspect_items = "".join(aspect_lines)

    topic_summary = html.escape(topic_result.get("summary", ""))
    topic_summary_block = f"<p><strong>Summary:</strong> {topic_summary}</p>" if topic_summary else ""

    processing_ms = topic_result.get("processing_time_ms", 0.0)

    sentiment_details = "".join([
        "<div class='analysis-column'>",
        "<h4>Sentiment Details</h4>",
        f"<p><strong>Confidence:</strong> {confidence_pct:.1f}%</p>",
        f"<p><strong>Compound Score:</strong> {scores.get('compound', 0):.3f}</p>",
        "<ul class='analysis-list'>",
        f"<li>Positive: {scores.get('pos', 0):.2f}</li>",
        f"<li>Neutral: {scores.get('neu', 0):.2f}</li>",
        f"<li>Negative: {scores.get('neg', 0):.2f}</li>",
        "</ul>",
        f"<div class='analysis-note'>{reasoning}</div>",
        key_phrases_block,
        "</div>",
    ])

    topic_details = "".join([
        "<div class='analysis-column'>",
        "<h4>Topic Analysis</h4>",
        f"<p><strong>Processing Time:</strong> {processing_ms:.2f}ms</p>",
        f"<div><strong>Aspect-Based Sentiment:</strong>{aspect_items}</div>",
        topic_summary_block,
        "</div>",
    ])

    return "".join([
        "<details class='analysis-details'>",
        "<summary>📊 View Full Analysis</summary>",
        "<div class='analysis-columns'>",
        sentiment_details,
        topic_details,
        "</div>",
        "</details>",
    ])


@st.cache_data
def calculate_aggregate_stats(reviews: List[Dict]) -> Dict[str, Any]:
    """Calculate aggregate statistics from all reviews."""
    total = len(reviews)
    avg_rating = sum(r["rating"] for r in reviews) / total

    rating_distribution = Counter(r["rating"] for r in reviews)

    return {
        "total_reviews": total,
        "average_rating": avg_rating,
        "rating_distribution": dict(rating_distribution),
    }


def extract_keywords_tfidf(reviews: List[Dict], top_n: int = 15) -> List[Tuple[str, float]]:
    """Extract important keywords using TF-IDF."""
    texts = [r["review"] for r in reviews]

    # Remove common stop words and use TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=top_n,
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=1,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Get average TF-IDF score for each term
        avg_scores = tfidf_matrix.mean(axis=0).A1
        keywords = [(feature_names[i], avg_scores[i]) for i in range(len(feature_names))]
        keywords.sort(key=lambda x: x[1], reverse=True)

        return keywords
    except Exception:
        return []


@st.cache_data
def batch_analyze_sentiments(reviews: List[Dict]) -> Dict[int, Dict[str, Any]]:
    """Batch analyze all reviews to demonstrate speed."""
    start_time = time.time()
    results = {}

    for review in reviews:
        results[review['id']] = analyze_sentiment_vader(review['review'])

    total_time = (time.time() - start_time) * 1000
    avg_time = total_time / len(reviews)

    return {
        'results': results,
        'total_time_ms': total_time,
        'avg_time_ms': avg_time
    }


def sentiment_analysis_page() -> None:
    """Render the sentiment analysis and topic modeling page."""

    # Custom CSS for presentation-friendly layout
    st.markdown(
        """
        <style>
        .review-card {
            background: white;
            padding: 0;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: all 0.3s ease;
        }
        .review-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        .review-header {
            padding: 1.2rem;
            border-left: 5px solid;
        }
        .review-content {
            padding: 0 1.2rem 1.2rem 1.2rem;
        }
        .sentiment-badge {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            color: white;
        }
        .topic-chip {
            display: inline-block;
            background: #e0e7ff;
            color: #4338ca;
            padding: 0.25rem 0.6rem;
            border-radius: 5px;
            margin: 0.2rem;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .performance-badge {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .reviews-container {
            max-height: 600px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }
        .reviews-container::-webkit-scrollbar {
            width: 8px;
        }
        .reviews-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        .reviews-container::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }
        .reviews-container::-webkit-scrollbar-thumb:hover {
            background: #4f46e5;
        }
        .aspect-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 0.1rem;
        }
        .analysis-details {
            background: #f9fafb;
            border-radius: 8px;
            margin-top: 0.8rem;
            border: 1px solid #e5e7eb;
        }
        .analysis-details summary {
            cursor: pointer;
            padding: 0.8rem 1rem;
            font-weight: 600;
            color: #1f2937;
            list-style: none;
            position: relative;
        }
        .analysis-details summary::after {
            content: "▾";
            position: absolute;
            right: 1rem;
            transition: transform 0.2s ease;
        }
        .analysis-details summary::-webkit-details-marker {
            display: none;
        }
        .analysis-details[open] summary {
            border-bottom: 1px solid #e5e7eb;
            background: #eef2ff;
        }
        .analysis-details[open] summary::after {
            transform: rotate(180deg);
        }
        .analysis-columns {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            padding: 1rem;
        }
        .analysis-column {
            flex: 1;
            min-width: 240px;
            background: white;
            border-radius: 6px;
            padding: 0.8rem;
            border: 1px solid #e5e7eb;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.02);
        }
        .analysis-column h4 {
            margin-top: 0;
            margin-bottom: 0.4rem;
            font-size: 1rem;
            color: #111827;
        }
        .analysis-list {
            margin: 0.3rem 0 0 1rem;
            padding: 0;
        }
        .analysis-note {
            background: #eef2ff;
            padding: 0.6rem;
            border-radius: 6px;
            margin-top: 0.6rem;
            border-left: 4px solid #6366f1;
            color: #3730a3;
            font-size: 0.9rem;
        }
        .aspect-detail {
            padding: 0.35rem 0.5rem;
            margin: 0.2rem 0;
            background: #f3f4f6;
            border-left: 4px solid #10b981;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #1f2937;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown("# 💭 Sentiment Analysis & Topic Modeling")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; margin-bottom: 2rem;'>
            <p style='margin: 0; color: #4b5563; font-size: 1.05rem;'>
                <strong>What you'll learn:</strong> How traditional NLP models can analyze sentiment and extract topics
                <em>without expensive LLM API calls</em> - delivering results in <strong>milliseconds instead of seconds</strong>.
            </p>
            <p style='margin: 0.8rem 0 0 0; color: #4b5563;'>
                <strong>Key advantage:</strong> Traditional models like VADER (sentiment) and LDA (topics) are
                <strong>100-1000x faster</strong> than LLMs, cost nothing to run, and work offline. Perfect for
                high-volume analysis or real-time applications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Performance comparison callout
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 1.2rem; border-radius: 10px; margin-bottom: 1.5rem; border-left: 4px solid #10b981;'>
            <strong style='color: #065f46; font-size: 1.1rem;'>⚡ Traditional NLP vs. LLM Performance</strong>
            <div style='margin-top: 0.5rem; color: #065f46;'>
                <div>• <strong>VADER Sentiment Analysis:</strong> ~1-5ms per review vs. ~500-2000ms with GPT-4</div>
                <div>• <strong>LDA Topic Modeling:</strong> ~100-500ms for entire dataset vs. ~10-30 seconds with LLMs</div>
                <div>• <strong>Cost:</strong> $0 (runs locally) vs. $0.01-0.10 per review with API calls</div>
                <div>• <strong>Scalability:</strong> Process millions of reviews per hour on a single machine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "sentiment_results" not in st.session_state:
        st.session_state["sentiment_results"] = {}
    if "topic_results" not in st.session_state:
        st.session_state["topic_results"] = {}
    if "reviews_loaded" not in st.session_state:
        st.session_state["reviews_loaded"] = False
    if "sample_reviews" not in st.session_state:
        st.session_state["sample_reviews"] = []
    if "expanded_reviews" not in st.session_state:
        st.session_state["expanded_reviews"] = set()

    # Section 1: Sample Data Display
    st.markdown("## 📋 Section 1: Real Amazon Review Data with Integrated Analysis")
    st.markdown("Loading real customer reviews from the **SetFit/amazon_reviews_multi_en** dataset on Hugging Face.")

    # Load data button
    if not st.session_state["reviews_loaded"]:
        col1, col2 = st.columns([3, 1])
        with col1:
            num_reviews = st.slider("Number of reviews to load:", min_value=10, max_value=100, value=30, step=10)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Load Reviews from Dataset", type="primary", use_container_width=True):
                with st.spinner("Loading real Amazon reviews from Hugging Face..."):
                    try:
                        random_seed = random.randint(0, 2**32 - 1)
                        reviews = load_amazon_reviews(num_samples=num_reviews, seed=random_seed)
                        st.session_state["sample_reviews"] = reviews
                        st.session_state["reviews_loaded"] = True
                        st.success(f"✅ Loaded {len(reviews)} real Amazon reviews!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading dataset: {e}")
                        st.info("Make sure you have internet connection to download from Hugging Face.")
    else:
        reviews = st.session_state["sample_reviews"]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ {len(reviews)} real Amazon reviews loaded from Hugging Face")
        with col2:
            if st.button("🔄 Reload Different Reviews", type="secondary", use_container_width=True):
                st.session_state["reviews_loaded"] = False
                st.session_state["sample_reviews"] = []
                st.session_state["sentiment_results"] = {}
                st.session_state["topic_results"] = {}
                st.session_state["expanded_reviews"] = set()
                st.rerun()

    if st.session_state["reviews_loaded"] and st.session_state["sample_reviews"]:
        reviews = st.session_state["sample_reviews"]

        # Precompute analysis for all reviews (fast, enables dashboard metrics)
        for review in reviews:
            review_id = review['id']
            if review_id not in st.session_state["sentiment_results"]:
                st.session_state["sentiment_results"][review_id] = analyze_sentiment_vader(review['review'])
            if review_id not in st.session_state["topic_results"]:
                st.session_state["topic_results"][review_id] = extract_topics_from_single_review(review['review'])

        # Cost & performance savings dashboard
        st.markdown("### 💰 Cost & Time Savings Dashboard")
        total_reviews = len(reviews)
        nlp_total_time_ms = sum(
            st.session_state["sentiment_results"][r['id']]["processing_time_ms"]
            for r in reviews
        )
        avg_nlp_time_ms = nlp_total_time_ms / total_reviews if total_reviews else 0

        # Estimated LLM costs/times (assumptions based on GPT-4 style API)
        llm_time_per_review_ms = 1500  # 1.5s average latency
        llm_cost_per_review = 0.05     # $0.05 per review on average
        nlp_cost_per_review = 0.0

        total_llm_time_ms = total_reviews * llm_time_per_review_ms
        total_llm_cost = total_reviews * llm_cost_per_review
        total_nlp_cost = total_reviews * nlp_cost_per_review

        time_saved_ms = total_llm_time_ms - nlp_total_time_ms
        cost_saved = total_llm_cost - total_nlp_cost

        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #ecfccb 0%, #d9f99d 100%);
            padding: 1.2rem; border-radius: 12px; border-left: 4px solid #65a30d; margin: 1rem 0;'>
                <div style='display: flex; flex-wrap: wrap; gap: 1.5rem;'>
                    <div style='flex: 1; min-width: 220px;'>
                        <div style='font-size: 0.85rem; color: #4d7c0f;'>Average per review</div>
                        <div style='font-size: 1.6rem; font-weight: 700; color: #1a2e05;'>{avg_nlp_time_ms:.2f} ms</div>
                        <div style='color: #4d7c0f;'>Traditional NLP processing time</div>
                    </div>
                    <div style='flex: 1; min-width: 220px;'>
                        <div style='font-size: 0.85rem; color: #4d7c0f;'>Estimated LLM time</div>
                        <div style='font-size: 1.6rem; font-weight: 700; color: #1a2e05;'>{llm_time_per_review_ms / 1000:.2f} sec</div>
                        <div style='color: #4d7c0f;'>Per-review GPT-style API latency</div>
                    </div>
                    <div style='flex: 1; min-width: 220px;'>
                        <div style='font-size: 0.85rem; color: #4d7c0f;'>Estimated cost avoided</div>
                        <div style='font-size: 1.6rem; font-weight: 700; color: #1a2e05;'>${cost_saved:,.2f}</div>
                        <div style='color: #4d7c0f;'>vs. ${total_llm_cost:,.2f} in LLM fees</div>
                    </div>
                </div>
                <div style='margin-top: 0.8rem; font-size: 0.95rem; color: #365314;'>
                    ✅ Saved ~{time_saved_ms/1000:.2f} seconds of processing time for {total_reviews} reviews by running VADER + keyword models locally.
                </div>
                <div style='margin-top: 0.3rem; font-size: 0.8rem; color: #4d7c0f;'>
                    Assumptions: VADER average = {avg_nlp_time_ms:.2f}ms, LLM average = {llm_time_per_review_ms/1000:.1f}s, LLM cost = ${llm_cost_per_review:.02f}/review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display reviews with integrated analysis
        st.markdown("### 📝 Reviews (Click to expand full analysis)")

        st.markdown("<div class='reviews-container'>", unsafe_allow_html=True)

        for idx, review in enumerate(reviews):
            review_id = review['id']

            sentiment_result = st.session_state["sentiment_results"][review_id]
            topic_result = st.session_state["topic_results"][review_id]

            review_card_html = build_review_card_html(review, sentiment_result, topic_result)

            st.markdown(review_card_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Section 2: Aggregate Dashboards
        st.markdown("## 📊 Section 2: Aggregate Analysis Dashboard")
        st.markdown("Patterns and insights across all reviews in the dataset")

        # Calculate aggregate stats
        stats = calculate_aggregate_stats(reviews)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", stats["total_reviews"])
        with col2:
            st.metric("Average Rating", f"{stats['average_rating']:.2f} ⭐")
        with col3:
            positive_count = sum(1 for r in reviews if r["rating"] >= 4)
            st.metric("Positive Reviews", f"{positive_count} ({positive_count/stats['total_reviews']*100:.0f}%)")
        with col4:
            negative_count = sum(1 for r in reviews if r["rating"] <= 2)
            st.metric("Negative Reviews", f"{negative_count} ({negative_count/stats['total_reviews']*100:.0f}%)")

        st.markdown("---")

        # Visualizations
        st.markdown("### Rating Distribution")
        rating_df = pd.DataFrame([
            {"Rating": f"{rating} Stars", "Count": count}
            for rating, count in sorted(stats["rating_distribution"].items(), reverse=True)
        ])
        fig1 = px.bar(
            rating_df,
            x="Rating",
            y="Count",
            color="Count",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
        )
        fig1.update_layout(
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Topic modeling on entire dataset
        st.markdown("### 🔍 Topic Modeling (LDA) - Entire Dataset")

        if st.button("🎯 Extract Topics from All Reviews", type="secondary"):
            with st.spinner("Running LDA topic modeling..."):
                lda_results = extract_topics_lda(reviews, num_topics=5, words_per_topic=6)
                st.session_state['lda_results'] = lda_results

        if 'lda_results' in st.session_state:
            lda_results = st.session_state['lda_results']
            proc_time = lda_results['processing_time_ms']

            st.markdown(f"**LDA Analysis completed in {proc_time:.0f}ms** ⚡")

            # Display topics
            for topic in lda_results['topics']:
                topic_id = topic['id']
                words = topic['words']
                weights = topic['weights']

                st.markdown(f"**Topic {topic_id + 1}:** {', '.join(words[:5])}")

                # Create horizontal bar for word weights
                topic_df = pd.DataFrame({
                    'Word': words,
                    'Weight': weights
                })

                fig = px.bar(
                    topic_df,
                    x='Weight',
                    y='Word',
                    orientation='h',
                    color='Weight',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    showlegend=False,
                    height=200,
                    margin=dict(t=10, b=10, l=10, r=10),
                    yaxis={'categoryorder': 'total ascending'},
                )
                st.plotly_chart(fig, use_container_width=True)

        # Keyword extraction
        st.markdown("### 🔑 Most Important Keywords (TF-IDF)")
        keywords = extract_keywords_tfidf(reviews, top_n=15)

        if keywords:
            keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Importance"])
            fig3 = px.bar(
                keyword_df,
                x="Importance",
                y="Keyword",
                orientation='h',
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig3.update_layout(
                showlegend=False,
                height=400,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis={'categoryorder': 'total ascending'},
            )
            st.plotly_chart(fig3, use_container_width=True)

    # Information expander
    with st.expander("ℹ️ How Traditional NLP Works"):
        st.markdown(
            """
            ### VADER Sentiment Analysis

            **What it is:** Valence Aware Dictionary and sEntiment Reasoner - a rule-based model specifically tuned for social media text.

            **How it works:**
            - Uses a lexicon of words with pre-assigned sentiment scores
            - Accounts for capitalization, punctuation, degree modifiers ("very good" vs "good")
            - Handles negations, contrasts, and emoticons
            - No neural network or API calls - pure algorithmic analysis

            **Advantages:**
            - **Speed:** Processes text in 1-5 milliseconds
            - **Cost:** Completely free, runs locally
            - **Reliability:** Deterministic results, no hallucinations
            - **Privacy:** No data leaves your infrastructure
            - **Accuracy:** ~80-85% on product reviews (vs ~85-90% for fine-tuned LLMs)

            ---

            ### Aspect-Based Sentiment

            **What it is:** Identifying specific product aspects (quality, price, comfort) and their sentiment.

            **How it works:**
            - Extracts key terms from each review
            - Maps terms to common product aspects
            - Analyzes sentiment of sentences mentioning each aspect
            - Provides granular understanding beyond overall sentiment

            **Business Value:**
            - Understand what customers like/dislike specifically
            - Guide product improvements
            - Track aspect sentiment over time

            ---

            ### LDA Topic Modeling

            **What it is:** Latent Dirichlet Allocation - a probabilistic model for discovering abstract topics in documents.

            **How it works:**
            - Treats documents as mixtures of topics
            - Topics are distributions over words
            - Uses statistical inference to discover topic structures
            - No training data needed - unsupervised learning

            **Advantages:**
            - **Speed:** Processes entire datasets in milliseconds to seconds
            - **Interpretability:** Clear word-to-topic mappings
            - **No training required:** Works out of the box
            - **Scalability:** Handles millions of documents
            """
        )

    # Presentation tips
    with st.expander("🎯 Presentation Tips for Lunch & Learn"):
        st.markdown(
            """
            ### Suggested Flow:

            1. **Start with the value prop** - Emphasize the speed/cost advantage in the header
               - "This runs 100-1000x faster than GPT-4"
               - "Processes reviews for FREE on your laptop"

            2. **Load the data** - Click "Load Reviews from Dataset"
               - Explain this is real data from Hugging Face
               - Show 30 reviews load instantly

            3. **Show the cost savings dashboard**
               - Point out instant processing time (single-digit ms)
               - Highlight the avoided LLM latency and per-review API cost
               - Reinforce the value for high-volume workloads

            4. **Scroll through reviews** - Show integrated analysis
               - Point out color-coded sentiment (green/yellow/red borders)
               - Show topic chips and aspect badges
               - Demonstrate the visual overview

            5. **Expand a positive review** - Click expander
               - Show detailed VADER scores
               - Explain compound score and confidence
               - Show aspect-based sentiment breakdown

            6. **Expand a negative review** - Show contrast
               - Point out negative aspects identified
               - Show how VADER caught the negativity

            7. **Expand a mixed review** - Show nuance
               - Demonstrate aspect-based analysis catching both positives and negatives
               - E.g., "Good quality" but "Poor shipping"

            8. **Scroll to dashboards** - Show aggregate insights
               - Rating distribution
               - LDA topics
               - TF-IDF keywords

            ### Key Talking Points:

            - **"Instant visual feedback"** - Color-coded borders show sentiment at a glance
            - **"Detailed on demand"** - Expand for full analysis
            - **"Aspect-based insights"** - Not just positive/negative, but what specifically
            - **"Production-ready performance"** - Real-time analysis suitable for user-facing apps
            - **"Cost-effective scale"** - Process millions for $0
            - **"Privacy-friendly"** - All analysis happens locally

            ### Interactive Elements:

            - Show how quickly you can scan 30 reviews visually
            - Compare the integrated approach to clicking through individual reviews
            - Demonstrate the accordion pattern for progressive disclosure
            - Highlight the performance metrics showing sub-millisecond processing
            """
        )
