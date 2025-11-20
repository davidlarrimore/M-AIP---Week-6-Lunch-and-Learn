"""Interactive lab for evaluating translations with BLEU."""

import streamlit as st
import plotly.express as px
import pandas as pd

from utils import compute_bleu


def bleu_page() -> None:
    """Render the BLEU evaluation playground."""
    st.markdown("# 🧮 BLEU Evaluation Lab")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #6366f1; margin-bottom: 1.2rem;'>
            <p style='margin: 0; color: #4338ca; font-size: 1.05rem;'>
                BLEU (Bilingual Evaluation Understudy) scores how close a model translation is to a human reference by
                comparing overlapping n-grams. Higher scores mean the candidate matches the reference phrasing more closely.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "BLEU focuses on precision of n-grams, so two fluent translations with different wording can both be good "
        "even if one scores lower. Use multiple references when possible and read the n-gram breakdown below."
    )

    col_ref, col_hyp = st.columns(2)
    with col_ref:
        st.subheader("Reference translation(s)")
        reference_text = st.text_area(
            "Each reference on a new line",
            value=(
                "The teammates protested when the referee overturned the decision after consulting with the sideline judge.\n"
                "The players objected when the lead official reversed the call after speaking with the assistants."
            ),
            help="Add one or more human references. Separate multiple references with line breaks.",
            height=140,
        )
    with col_hyp:
        st.subheader("Candidate translation")
        candidate_text = st.text_area(
            "Model output to evaluate",
            value="The teammates protested when the lead reviewer challenged the decision and reversed the outcome after consulting with advisors about the approach.",
            help="Paste the model-generated translation you want to score.",
            height=140,
        )

    col_controls, _ = st.columns([1, 3])
    with col_controls:
        max_n = st.slider("Highest n-gram order", min_value=2, max_value=4, value=4, step=1)

    references = [line.strip() for line in reference_text.splitlines() if line.strip()]
    if not references:
        st.warning("Add at least one reference translation to compute BLEU.")
        return

    if not candidate_text.strip():
        st.info("Enter a candidate translation to see BLEU scores.")
        return

    bleu_score, precisions = compute_bleu(references, candidate_text, max_n=max_n)
    bleu_percent = bleu_score * 100

    metric_col, chart_col = st.columns([1, 2])
    with metric_col:
        st.metric("BLEU score", f"{bleu_percent:.2f}", help="0 = no n-gram overlap, 100 = perfect match")
        st.caption("Includes brevity penalty and smoothing for short sentences.")

    with chart_col:
        df = pd.DataFrame(
            {
                "n-gram": [f"{n}-gram" for n in range(1, max_n + 1)],
                "precision": precisions,
            }
        )
        fig = px.bar(
            df,
            x="n-gram",
            y="precision",
            color="precision",
            color_continuous_scale="Blues",
            range_y=[0, 1],
            title="Modified n-gram precision (higher is better)",
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### How to read this")
    st.markdown(
        """
        - **1-gram vs. 4-gram:** High 1-gram precision means good word choice; higher-order n-grams capture fluency and phrasing.
        - **Multiple references:** Add more references to reward valid rephrasings—BLEU improves when any reference matches.
        - **Limitations:** BLEU measures surface overlap; it can under-score valid paraphrases and does not check adequacy alone.
        """
    )


if __name__ == "__main__":
    bleu_page()
