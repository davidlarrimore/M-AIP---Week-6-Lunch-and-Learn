import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import embed_text, cosine_similarity


DEFAULT_COMPARE = """A fast fox leaps over a sleepy dog.
The sun is shining on the meadow.
Astronomy books sit on the shelf."""


def describe_similarity(score: float) -> str:
    """Provide a short explanation based on similarity thresholds."""
    if score >= 0.9:
        return "🟢 Very High Similarity - Nearly identical meanings"
    if score >= 0.75:
        return "🟡 High Similarity - Related concepts and topics"
    if score >= 0.5:
        return "🟠 Moderate Similarity - Some common themes"
    return "🔴 Low Similarity - Distant or different topics"


def get_similarity_color(score: float) -> str:
    """Return a color based on similarity score."""
    if score >= 0.9:
        return "#10b981"  # green
    if score >= 0.75:
        return "#f59e0b"  # yellow
    if score >= 0.5:
        return "#f97316"  # orange
    return "#ef4444"  # red


def embeddings_page() -> None:
    """Show embeddings and cosine similarity in an interactive chart."""
    st.markdown("# 🎯 Embedding Similarity Explorer")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; margin-bottom: 1.5rem;'>
            <p style='margin: 0; color: #4b5563; font-size: 1.05rem;'>
                Generate vector representations and compare them visually. Explore how embeddings
                capture semantic meaning and measure similarity between different texts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input section in columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📌 Anchor Sentence")
        base_text = st.text_area(
            "Base text for comparison",
            value="A curious cat investigates a sunbeam.",
            height=120,
            label_visibility="collapsed",
            help="This is the reference text that all others will be compared against",
        )

    with col2:
        st.markdown("### 📋 Comparison Sentences")
        comparison_input = st.text_area(
            "Enter sentences to compare (one per line)",
            value=DEFAULT_COMPARE,
            height=120,
            label_visibility="collapsed",
            help="Enter multiple sentences, one per line",
        )

    if "embedding_results" not in st.session_state:
        st.session_state["embedding_results"] = []

    # Center-aligned button
    col_button = st.columns([1, 2, 1])
    with col_button[1]:
        if st.button("🚀 Generate Embeddings & Compare", use_container_width=True, type="primary"):
            candidates = [
                line.strip()
                for line in comparison_input.splitlines()
                if line.strip()
            ]
            if not base_text.strip() or not candidates:
                st.warning("⚠️ Provide both an anchor sentence and at least one comparison sentence")
            else:
                with st.spinner("🔄 Generating embeddings and calculating similarities..."):
                    base_embedding = embed_text(base_text)
                    results = []
                    for candidate in candidates:
                        candidate_embedding = embed_text(candidate)
                        score = cosine_similarity(base_embedding, candidate_embedding)
                        results.append(
                            {
                                "text": candidate,
                                "score": round(score, 3),
                            }
                        )
                    st.session_state["embedding_results"] = results

    st.markdown("---")

    if st.session_state["embedding_results"]:
        results = st.session_state["embedding_results"]
        st.markdown("### 📊 Similarity Results")

        # Create enhanced plotly chart
        labels = [entry["text"][:50] + "..." if len(entry["text"]) > 50 else entry["text"] for entry in results]
        scores = [entry["score"] for entry in results]
        colors = [get_similarity_color(score) for score in scores]

        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=scores,
                text=[f"{score:.3f}" for score in scores],
                textposition="outside",
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(0,0,0,0.2)', width=1)
                ),
                hovertemplate="<b>%{x}</b><br>Similarity: %{y:.3f}<extra></extra>",
            )
        ])

        fig.update_layout(
            yaxis=dict(
                range=[0, 1],
                title="Cosine Similarity Score",
                gridcolor='rgba(0,0,0,0.1)',
            ),
            xaxis=dict(
                title="Comparison Sentences",
                tickangle=-45,
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(t=20, b=100),
            font=dict(size=12),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Results cards
        st.markdown("### 🏆 Detailed Results")

        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        for idx, entry in enumerate(sorted_results):
            score = entry["score"]
            color = get_similarity_color(score)

            st.markdown(
                f"""
                <div style='background: white; padding: 1.2rem; border-radius: 10px;
                margin: 0.8rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                border-left: 4px solid {color};'>
                    <div style='display: flex; justify-content: space-between; align-items: start;'>
                        <div style='flex: 1;'>
                            <div style='color: #6b7280; font-size: 0.85rem; margin-bottom: 0.3rem;'>
                                Rank #{idx + 1}
                            </div>
                            <div style='color: #1f2937; font-size: 1rem; margin-bottom: 0.5rem;'>
                                "{entry["text"]}"
                            </div>
                            <div style='color: {color}; font-weight: 600;'>
                                {describe_similarity(score)}
                            </div>
                        </div>
                        <div style='background: {color}; color: white; padding: 0.8rem 1.2rem;
                        border-radius: 8px; font-size: 1.5rem; font-weight: 700; min-width: 80px;
                        text-align: center;'>
                            {score:.3f}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Information expander
        with st.expander("ℹ️ Understanding Cosine Similarity"):
            st.markdown(
                """
                **What is Cosine Similarity?**
                - A measure of similarity between two vectors (0 to 1 scale)
                - 1.0 = Identical meaning/direction
                - 0.0 = Completely different

                **How Embeddings Work:**
                - Text is converted to high-dimensional vectors
                - Similar meanings cluster together in vector space
                - Distance between vectors indicates semantic similarity

                **Interpretation Guide:**
                - **0.9 - 1.0**: Nearly identical meaning
                - **0.75 - 0.9**: Strong semantic relationship
                - **0.5 - 0.75**: Moderate topical overlap
                - **< 0.5**: Weak or no relationship
                """
            )
    else:
        st.info("💡 Click the button above to generate embeddings and see similarity comparisons")
