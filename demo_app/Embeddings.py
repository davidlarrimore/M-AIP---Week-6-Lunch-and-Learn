import streamlit as st
import plotly.express as px

from utils import embed_text, cosine_similarity


DEFAULT_COMPARE = """A fast fox leaps over a sleepy dog.
The sun is shining on the meadow.
Astronomy books sit on the shelf."""


def describe_similarity(score: float) -> str:
    """Provide a short explanation based on similarity thresholds."""
    if score >= 0.9:
        return "Green → Very similar meaning."
    if score >= 0.75:
        return "Yellow → Related concepts."
    return "Red → Distant or different topic."


def embeddings_page() -> None:
    """Show embeddings and cosine similarity in an interactive chart."""
    st.header("Embedding Similarity Explorer")
    st.markdown(
        """
        Generate vector representations and compare them visually. Use the chart
        to explain why higher cosine scores imply closer meaning.
        """
    )
    base_text = st.text_area(
        "Anchor sentence", value="A curious cat investigates a sunbeam.", height=140
    )
    comparison_input = st.text_area(
        "Comparison sentences (one per line)",
        value=DEFAULT_COMPARE,
        height=140,
    )

    if "embedding_results" not in st.session_state:
        st.session_state["embedding_results"] = []

    if st.button("Generate embeddings and compare"):
        candidates = [
            line.strip()
            for line in comparison_input.splitlines()
            if line.strip()
        ]
        if not base_text.strip() or not candidates:
            st.warning("Provide both an anchor sentence and at least one comparison.")
        else:
            with st.spinner("Embedding text..."):
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

    if st.session_state["embedding_results"]:
        results = st.session_state["embedding_results"]
        labels = [entry["text"] for entry in results]
        scores = [entry["score"] for entry in results]
        colors = scores
        fig = px.bar(
            x=labels,
            y=scores,
            text=[f"{score:.2f}" for score in scores],
            range_y=[0, 1],
            color=colors,
            color_continuous_scale=px.colors.sequential.Tealrose,
            labels={"x": "Candidate sentence", "y": "Cosine similarity"},
        )
        fig.update_layout(transition_duration=400)
        st.plotly_chart(fig, use_container_width=True)

        best = max(results, key=lambda entry: entry["score"])
        st.metric("Most similar", best["text"], delta=f"{best['score']:.3f}")
        st.markdown(describe_similarity(best["score"]))
    else:
        st.info("Press the button above to fetch embeddings and see similarity bars.")
