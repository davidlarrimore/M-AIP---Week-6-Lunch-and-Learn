"""Sentiment Analysis and Topic Modeling on Amazon Reviews using Traditional NLP."""

import html
import re
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

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


# Category mapping for Amazon product categories
CATEGORY_MAPPING = {
    'book': 'Books',
    'dvd': 'Movies & TV',
    'electronics': 'Electronics',
    'kitchen': 'Kitchen & Home',
    'kitchen_&_housewares': 'Kitchen & Home',
    'apparel': 'Clothing',
    'camera': 'Electronics',
    'health_&_personal_care': 'Health & Personal Care',
    'music': 'Music',
    'software': 'Software',
    'sports_&_outdoors': 'Sports & Outdoors',
    'toys': 'Toys & Games',
    'video': 'Movies & TV',
    'wireless': 'Electronics',
    'office_product': 'Office Products',
}


@st.cache_data
def load_amazon_reviews(num_samples: int = 50, seed: int = 42) -> List[Dict]:
    """
    Load real Amazon reviews from Hugging Face dataset.
    Uses SetFit/amazon_reviews_multi_en which contains English reviews.
    """
    # Load the dataset (validation split for diverse reviews)
    dataset = load_dataset('SetFit/amazon_reviews_multi_en', split='validation')

    # Convert to pandas for easier manipulation
    df = pd.DataFrame(dataset)

    # Sample diverse reviews across different ratings
    sampled_reviews = []

    # Try to get diverse ratings (1-5 stars)
    for rating in [1, 2, 3, 4, 5]:
        rating_reviews = df[df['label'] == rating - 1]  # Labels are 0-4 in this dataset
        if len(rating_reviews) > 0:
            # Sample a few from each rating category
            sample_size = min(num_samples // 5, len(rating_reviews))
            samples = rating_reviews.sample(n=sample_size, random_state=seed + rating)
            sampled_reviews.append(samples)

    # Combine all samples
    if sampled_reviews:
        final_df = pd.concat(sampled_reviews, ignore_index=True)
    else:
        # Fallback: just sample randomly
        final_df = df.sample(n=min(num_samples, len(df)), random_state=seed)

    # Convert to our format
    reviews = []
    for idx, row in final_df.iterrows():
        # Extract fields from the dataset
        review_text = row.get('text', row.get('review_body', ''))
        rating = int(row.get('label', 0)) + 1  # Convert 0-4 to 1-5

        # Try to get product category
        category = row.get('product_category', 'unknown')
        if isinstance(category, str):
            category = CATEGORY_MAPPING.get(category.lower(), category.title())
        else:
            category = 'General'

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
                'category': category
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


@st.cache_data
def calculate_aggregate_stats(reviews: List[Dict]) -> Dict[str, Any]:
    """Calculate aggregate statistics from all reviews."""
    total = len(reviews)
    avg_rating = sum(r["rating"] for r in reviews) / total

    rating_distribution = Counter(r["rating"] for r in reviews)
    category_distribution = Counter(r["category"] for r in reviews)

    return {
        "total_reviews": total,
        "average_rating": avg_rating,
        "rating_distribution": dict(rating_distribution),
        "category_distribution": dict(category_distribution),
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
        .section-container {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .review-card {
            background: #f9fafb;
            padding: 1.2rem;
            border-radius: 10px;
            margin: 0.8rem 0;
            border-left: 4px solid #667eea;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .review-card:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }
        .review-card-selected {
            background: #eef2ff;
            border-left: 4px solid #4f46e5;
        }
        .sentiment-badge {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .topic-chip {
            display: inline-block;
            background: #e0e7ff;
            color: #4338ca;
            padding: 0.3rem 0.7rem;
            border-radius: 6px;
            margin: 0.2rem;
            font-size: 0.85rem;
        }
        .performance-badge {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        .reviews-container {
            max-height: 500px;
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
    if "selected_review_id" not in st.session_state:
        st.session_state["selected_review_id"] = None
    if "sentiment_results" not in st.session_state:
        st.session_state["sentiment_results"] = {}
    if "topic_results" not in st.session_state:
        st.session_state["topic_results"] = {}
    if "reviews_loaded" not in st.session_state:
        st.session_state["reviews_loaded"] = False
    if "sample_reviews" not in st.session_state:
        st.session_state["sample_reviews"] = []

    # Section 1: Sample Data Display
    st.markdown("## 📋 Section 1: Real Amazon Review Data")
    st.markdown("Loading real customer reviews from the **SetFit/amazon_reviews_multi_en** dataset on Hugging Face.")

    # Load data button
    if not st.session_state["reviews_loaded"]:
        col1, col2 = st.columns([3, 1])
        with col1:
            num_reviews = st.slider("Number of reviews to load:", min_value=10, max_value=100, value=50, step=10)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Load Reviews from Dataset", type="primary", use_container_width=True):
                with st.spinner("Loading real Amazon reviews from Hugging Face..."):
                    try:
                        reviews = load_amazon_reviews(num_samples=num_reviews)
                        st.session_state["sample_reviews"] = reviews
                        st.session_state["reviews_loaded"] = True
                        st.success(f"✅ Loaded {len(reviews)} real Amazon reviews!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading dataset: {e}")
                        st.info("Make sure you have internet connection to download from Hugging Face.")
    else:
        reviews = st.session_state["sample_reviews"]
        st.success(f"✅ {len(reviews)} real Amazon reviews loaded from Hugging Face")

        if st.button("🔄 Reload Different Reviews", type="secondary"):
            st.session_state["reviews_loaded"] = False
            st.session_state["sample_reviews"] = []
            st.session_state["selected_review_id"] = None
            st.session_state["sentiment_results"] = {}
            st.session_state["topic_results"] = {}
            st.rerun()

    if st.session_state["reviews_loaded"] and st.session_state["sample_reviews"]:
        reviews = st.session_state["sample_reviews"]

        # Display reviews in a scrollable container
        st.markdown("### Click any review to analyze:")

        st.markdown("<div class='reviews-container'>", unsafe_allow_html=True)

        # Create columns for better layout
        for idx, review in enumerate(reviews):
            is_selected = st.session_state["selected_review_id"] == review["id"]
            card_class = "review-card review-card-selected" if is_selected else "review-card"

            # Star rating visualization
            stars = "⭐" * review["rating"] + "☆" * (5 - review["rating"])

            review_html = f"""
            <div class='{card_class}' id='review-{review["id"]}'>
                <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;'>
                    <div>
                        <strong style='color: #1f2937; font-size: 1rem;'>{html.escape(review["product"])}</strong>
                        <span style='color: #6b7280; margin-left: 0.5rem;'>({review["category"]})</span>
                    </div>
                    <div style='font-size: 1.1rem;'>{stars}</div>
                </div>
                <div style='color: #4b5563; line-height: 1.5;'>"{html.escape(review["review"])}"</div>
            </div>
            """
            st.markdown(review_html, unsafe_allow_html=True)

            # Create button for clicking
            if st.button(f"Select Review #{review['id']}", key=f"select_{review['id']}", type="secondary"):
                st.session_state["selected_review_id"] = review["id"]
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Section 2: Real-time Analysis
        if st.session_state["selected_review_id"] is not None:
            selected_review = next(r for r in reviews if r["id"] == st.session_state["selected_review_id"])

            st.markdown("## 🔍 Section 2: Real-time Analysis")
            st.markdown(f"### Analyzing: {selected_review['product']}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 😊 Sentiment Analysis (VADER)")

                review_id = selected_review["id"]
                if review_id not in st.session_state["sentiment_results"]:
                    sentiment_result = analyze_sentiment_vader(selected_review["review"])
                    st.session_state["sentiment_results"][review_id] = sentiment_result
                else:
                    sentiment_result = st.session_state["sentiment_results"][review_id]

                sentiment = sentiment_result["sentiment"]
                confidence = sentiment_result["confidence"]
                reasoning = sentiment_result["reasoning"]
                key_phrases = sentiment_result.get("key_phrases", [])
                proc_time = sentiment_result.get("processing_time_ms", 0)

                sentiment_color = get_sentiment_color(sentiment)
                sentiment_emoji = get_sentiment_emoji(sentiment)

                st.markdown(
                    f"""
                    <div style='background: {sentiment_color}; color: white; padding: 1.5rem;
                    border-radius: 10px; text-align: center; margin-bottom: 1rem;'>
                        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>{sentiment_emoji}</div>
                        <div style='font-size: 1.5rem; font-weight: 700; text-transform: uppercase;'>
                            {sentiment}
                        </div>
                        <div style='font-size: 1rem; opacity: 0.9; margin-top: 0.5rem;'>
                            Confidence: {confidence:.1%}
                        </div>
                        <div style='font-size: 0.85rem; opacity: 0.9; margin-top: 0.3rem;'>
                            ⚡ Analyzed in {proc_time:.2f}ms
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**How VADER classified this:**")
                st.info(reasoning)

                if key_phrases:
                    st.markdown("**Key phrases identified:**")
                    phrases_html = " ".join([f"<span class='topic-chip'>{html.escape(phrase)}</span>" for phrase in key_phrases])
                    st.markdown(phrases_html, unsafe_allow_html=True)

            with col2:
                st.markdown("#### 🏷️ Topic Extraction (Keyword Analysis)")

                if review_id not in st.session_state["topic_results"]:
                    topic_result = extract_topics_from_single_review(selected_review["review"])
                    st.session_state["topic_results"][review_id] = topic_result
                else:
                    topic_result = st.session_state["topic_results"][review_id]

                topics = topic_result.get("topics", [])
                aspects = topic_result.get("aspects", {})
                summary = topic_result.get("summary", "")
                proc_time = topic_result.get("processing_time_ms", 0)

                st.markdown(f"**Identified Topics:** <span class='performance-badge'>⚡ {proc_time:.2f}ms</span>", unsafe_allow_html=True)

                if aspects:
                    for topic, aspect_sentiment in aspects.items():
                        aspect_color = get_sentiment_color(aspect_sentiment)
                        st.markdown(
                            f"""
                            <div style='background: white; padding: 0.8rem; border-radius: 8px;
                            margin: 0.5rem 0; border-left: 4px solid {aspect_color};'>
                                <strong style='color: #1f2937;'>{html.escape(topic.title())}</strong>
                                <span style='float: right; color: {aspect_color}; font-weight: 600;'>
                                    {aspect_sentiment.title()}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if topics:
                    st.markdown("**Key terms:**")
                    topics_html = " ".join([f"<span class='topic-chip'>{html.escape(topic)}</span>" for topic in topics[:5]])
                    st.markdown(topics_html, unsafe_allow_html=True)

                st.markdown("**Summary:**")
                st.success(summary)

            st.divider()
        else:
            st.info("👆 Click any review above to see real-time sentiment analysis and topic extraction")
            st.divider()

        # Section 3: Aggregate Dashboards
        st.markdown("## 📊 Section 3: Aggregate Analysis Dashboard")

        # Batch process all reviews to show speed
        st.markdown("### ⚡ Batch Processing Performance")

        if st.button("🚀 Analyze All Reviews (Batch Processing)", type="primary"):
            with st.spinner("Processing all reviews with traditional NLP..."):
                batch_results = batch_analyze_sentiments(reviews)
                st.session_state['batch_results'] = batch_results

        if 'batch_results' in st.session_state:
            batch_results = st.session_state['batch_results']
            total_time = batch_results['total_time_ms']
            avg_time = batch_results['avg_time_ms']

            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                padding: 1.2rem; border-radius: 10px; margin-bottom: 1.5rem;'>
                    <strong style='color: #1e40af; font-size: 1.1rem;'>⚡ Performance Results</strong>
                    <div style='margin-top: 0.5rem; color: #1e40af;'>
                        <div>• <strong>Total Processing Time:</strong> {total_time:.2f}ms for {len(reviews)} reviews</div>
                        <div>• <strong>Average Time per Review:</strong> {avg_time:.2f}ms</div>
                        <div>• <strong>Throughput:</strong> ~{int(1000/avg_time * 60):,} reviews per minute</div>
                        <div>• <strong>LLM Comparison:</strong> Same task would take ~{len(reviews) * 1000 / 1000:.1f}-{len(reviews) * 2000 / 1000:.1f} seconds with GPT-4</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
        col1, col2 = st.columns(2)

        with col1:
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

        with col2:
            st.markdown("### Reviews by Category")
            category_df = pd.DataFrame([
                {"Category": cat, "Count": count}
                for cat, count in stats["category_distribution"].items()
            ])
            fig2 = px.pie(
                category_df,
                values="Count",
                names="Category",
                hole=0.4,
            )
            fig2.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

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

            **When to use:**
            - High-volume batch processing
            - Real-time applications
            - Cost-sensitive scenarios
            - Offline/air-gapped environments

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

            **When to use:**
            - Exploring unknown datasets
            - Finding themes in large document collections
            - Customer feedback analysis
            - Content categorization

            ---

            ### TF-IDF Keyword Extraction

            **What it is:** Term Frequency - Inverse Document Frequency

            **How it works:**
            - Measures how important a word is to a document in a collection
            - Common words are downweighted, distinctive words highlighted
            - Pure statistical method, no machine learning

            **Advantages:**
            - Extremely fast
            - No model training
            - Interpretable results
            - Works well with small datasets

            ---

            ### When to Use Traditional NLP vs. LLMs

            **Use Traditional NLP when:**
            - ✅ You need millisecond response times
            - ✅ Processing millions of documents
            - ✅ Working with limited budgets
            - ✅ Offline/on-premise requirements
            - ✅ Good-enough accuracy is acceptable
            - ✅ You want deterministic, explainable results

            **Use LLMs when:**
            - 🎯 You need nuanced understanding
            - 🎯 Handling complex, context-dependent text
            - 🎯 Accuracy is critical
            - 🎯 Working with creative or ambiguous content
            - 🎯 Need multi-step reasoning
            - 🎯 Processing volume is low enough to justify cost

            **Best approach:** Use traditional NLP for initial filtering/categorization, then apply LLMs only to edge cases or high-value items.
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
               - Show the variety of products and ratings

            3. **Show Section 1** - Display the sample data
               - Scroll through reviews to show diversity

            4. **Click a positive review (5 stars)** - Demonstrate real-time analysis
               - Point out the processing time (1-5ms)
               - Explain VADER's approach (rule-based, lexicon)
               - Show the confidence and reasoning

            5. **Click a negative review (1-2 stars)** - Show the contrast
               - Emphasize how fast it processes negative sentiment
               - Compare accuracy to what an LLM would find

            6. **Click a mixed review (3 stars)** - Show nuanced analysis
               - 3-star reviews often have both positive and negative aspects
               - Show how VADER handles balanced sentiment

            7. **Batch processing demo** - Click "Analyze All Reviews"
               - Show the total time for all reviews
               - Calculate throughput (thousands per minute)
               - Compare to LLM cost/time

            8. **Topic modeling** - Click "Extract Topics"
               - Show LDA results
               - Explain unsupervised discovery
               - Point out speed (100-500ms for entire dataset)

            9. **Scroll to dashboards** - Show aggregate insights

            ### Key Talking Points:

            - **"Traditional NLP still has a place in 2024"** - Not everything needs GPT-4
            - **"Speed matters"** - Real-time applications, user-facing features
            - **"Cost matters"** - At scale, LLM costs add up quickly
            - **"Hybrid approach"** - Use traditional NLP for 95% of cases, LLMs for edge cases
            - **"Privacy matters"** - Traditional NLP runs locally, no data sent to APIs
            - **"Reliability matters"** - No API downtime, rate limits, or hallucinations
            - **"Real data"** - These are actual Amazon customer reviews, not synthetic

            ### Questions to Address:

            **Q: Is traditional NLP less accurate?**
            A: For sentiment analysis on reviews, VADER achieves ~80-85% accuracy vs ~85-90% for LLMs. The 5-10% accuracy trade-off often isn't worth 1000x slower processing.

            **Q: When should we use LLMs instead?**
            A: For complex reasoning, creative tasks, or when you need to extract structured information that requires understanding context and nuance.

            **Q: Can we combine both?**
            A: Absolutely! Use VADER to filter reviews, then send only uncertain cases (compound score near 0) to an LLM for final classification.

            **Q: Where does this data come from?**
            A: Real Amazon product reviews from the SetFit/amazon_reviews_multi_en dataset on Hugging Face, containing thousands of verified customer reviews.
            """
        )
