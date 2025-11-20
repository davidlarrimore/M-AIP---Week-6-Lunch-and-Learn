"""Guided BLEU walkthrough for side-by-side translations and scoring."""

from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from utils import compute_bleu


EXAMPLE_SENTENCES: List[Dict[str, str]] = [
    {
        "id": "cat-mat",
        "label": "The cat sits on the mat.",
        "source": "El gato se sienta en la estera.",
        "reference": "The cat sits on the mat.",
        "traditional": "The cat is sitting on mat.",
        "llm": "The cat sits on the mat.",
        "context": "Shortest option: literal phrasing, almost no ambiguity.",
    },
    {
        "id": "weather",
        "label": "The weather today is beautiful, perfect for a walk.",
        "source": "El clima hoy es hermoso, perfecto para dar un paseo.",
        "reference": "The weather today is beautiful, perfect for a walk.",
        "traditional": "The weather is very nice today, perfect to take a walk.",
        "llm": "The weather today is beautiful, perfect for a walk.",
        "context": "Adds a purpose clause; checks how models keep both ideas together.",
    },
    {
        "id": "quiet-door",
        "label": "She opened the door quietly so she wouldn’t wake the baby.",
        "source": "Ella abrió la puerta en silencio para no despertar al bebé.",
        "reference": "She opened the door quietly so she would not wake the baby.",
        "traditional": "She quietly opened the door so the baby would not wake up.",
        "llm": "She opened the door quietly so she wouldn’t wake the baby.",
        "context": "Medium length with causal clause; tests reordering and pronoun handling.",
    },
    {
        "id": "budget-review",
        "label": "Longer: budget impact and legal review before a vote",
        "source": (
            "El comité pospuso la votación hasta recibir un informe completo sobre el impacto presupuestario "
            "y las implicaciones legales del acuerdo propuesto."
        ),
        "reference": (
            "The committee postponed the vote until it received a full report on the budget impact "
            "and the legal implications of the proposed agreement."
        ),
        "traditional": (
            "The committee delayed the vote until getting a complete report about the budget impact "
            "and legal consequences of the proposed deal."
        ),
        "llm": (
            "The committee postponed the vote until it got a complete report on the budget impact "
            "and the legal implications of the proposed agreement."
        ),
        "context": "Complex, multi-clause sentence that stresses longer n-gram continuity and brevity penalties.",
    },
    {
        "id": "airport-disruption",
        "label": "Longest: airport disruption, rebooking, and overnight vouchers",
        "source": (
            "Después de horas de retrasos por tormentas, la aerolínea reubicó a los pasajeros en vuelos alternativos "
            "y ofreció vales de hotel para quienes no podían salir hasta la mañana siguiente."
        ),
        "reference": (
            "After hours of storm delays, the airline rebooked passengers on alternate flights and offered hotel vouchers "
            "to those who could not depart until the next morning."
        ),
        "traditional": (
            "After hours of storm delays, the airline moved passengers to alternate flights and gave hotel coupons "
            "to those who could not leave until next morning."
        ),
        "llm": (
            "After hours of weather delays, the airline rebooked passengers on alternate flights and provided hotel vouchers "
            "for those who could not depart until the next morning."
        ),
        "context": "Longer sentence with paraphrased clauses; shows how BLEU rewards phrase-level overlap.",
    },
]

COLOR_BANDS: List[Tuple[int, int, str, str]] = [
    (0, 30, "🔴", "Needs work"),
    (30, 70, "🟡", "Okay"),
    (70, 101, "🟢", "Strong match"),
]


def _reset_progress() -> None:
    """Reset translation and scoring state when the example changes."""
    st.session_state["translations_ready"] = False
    st.session_state["scores_ready"] = False
    st.session_state["show_highlights"] = False


def _badge_for_score(value: float) -> str:
    """Return colored badge text for a BLEU percentage."""
    for low, high, icon, label in COLOR_BANDS:
        if low <= value < high:
            return f"{icon} {label}"
    return ""


def _highlight_candidate(reference: str, candidate: str) -> str:
    """Produce HTML markup showing token overlap strength against the reference."""
    ref_tokens = [tok.strip(".,!?").lower() for tok in reference.split()]
    spans: List[str] = []
    for token in candidate.split():
        cleaned = token.strip(".,!?").lower()
        best_match = max((SequenceMatcher(None, cleaned, ref).ratio() for ref in ref_tokens), default=0.0)
        if best_match >= 0.88:
            color, text_color = "#dcfce7", "#166534"
        elif best_match >= 0.65:
            color, text_color = "#fef9c3", "#92400e"
        else:
            color, text_color = "#fee2e2", "#991b1b"
        spans.append(
            f"<span style='background:{color}; color:{text_color}; padding:2px 6px; border-radius:6px; margin-right:4px;'>"
            f"{token}</span>"
        )
    return " ".join(spans)


def _compute_scores(reference: str, candidates: Dict[str, str]) -> List[Dict[str, str]]:
    """Calculate BLEU for each candidate and attach friendly interpretations."""
    results: List[Dict[str, str]] = []
    for model, text in candidates.items():
        bleu_score, _ = compute_bleu([reference], text, max_n=4)
        percent = bleu_score * 100
        results.append(
            {
                "Model": model,
                "BLEU Score": f"{percent:.1f}",
                "Signal": _badge_for_score(percent),
                "Friendly Explanation": (
                    "Very close to the reference translation."
                    if percent >= 70
                    else "Some overlap, but wording differs a lot."
                    if percent <= 30
                    else "Decent overlap with some wording differences."
                ),
            }
        )
    return results


def bleu_page() -> None:
    """Render the guided BLEU evaluation walkthrough."""
    st.markdown("# 🧮 BLEU Evaluation Lab")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f0f7ff 0%, #e8f0fe 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #1d4ed8; margin-bottom: 1rem;'>
            <p style='margin: 0; color: #1e3a8a; font-size: 1.05rem;'>
                BLEU (Bilingual Evaluation Understudy) is a precision-first metric: it counts how many n-grams in a
                model translation overlap with a human reference and applies a brevity penalty if the hypothesis is too short.
                Higher scores mean closer phrasing matches; multiple references help reward valid paraphrases.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #e8f0fe 0%, #e0f2fe 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #1d4ed8; margin-bottom: 1.2rem;'>
            <p style='margin: 0; color: #1e3a8a; font-size: 1.05rem;'>
                Walk through BLEU like a learning journey: see a source sentence, compare different model translations,
                then watch the scores tell the story of reference vs. hypothesis vs. quality. Longer examples below
                highlight how phrase choices and brevity penalties matter.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "selected_example" not in st.session_state:
        st.session_state["selected_example"] = EXAMPLE_SENTENCES[0]["id"]
    if "translations_ready" not in st.session_state:
        _reset_progress()

    st.markdown("### 1️⃣ Set the Stage: Source, Reference, and Scenario")
    st.caption(
        "We are going to first select a sentence that will be used for translation. Pick from short to long options to see how complexity affects BLEU."
    )

    option_labels = {example["label"]: example for example in EXAMPLE_SENTENCES}
    selected_label = st.selectbox(
        "Pick a short sentence to explore",
        options=list(option_labels.keys()),
        index=0,
        key="example_select",
        on_change=_reset_progress,
    )
    example = option_labels[selected_label]
    st.session_state["selected_example"] = example["id"]

    col_source, col_reference = st.columns(2)
    with col_source:
        st.markdown("**Original (Spanish source)**")
        st.info(example["source"])
    with col_reference:
        st.markdown("**Reference translation (English)**")
        st.success(example["reference"])
    st.caption(example.get("context", ""))

    st.markdown(
        """
        **Progression of examples:** short literal → adds purpose clause → adds causal nuance → long budget/legal review → longest travel disruption.
        Expect BLEU to reward longer shared phrases and penalize drops or rephrasings that break n-grams.
        """
    )

    st.divider()
    st.markdown("### 2️⃣ Test Multiple Translation Models")
    st.markdown(
        "Generate side-by-side outputs from a traditional baseline vs. a modern LLM."
    )
    st.info(
        "🧊 **Traditional baseline:** Phrase-based / statistical feel—more literal, may drop small words.\n"
        "🤖 **LLM (ChatGPT-style):** Neural output similar to GPT-4o mini—more fluent and closer to the reference.\n"
        "These are fixed exemplar outputs to keep the BLEU comparison deterministic."
    )

    if st.button("🚀 Generate translations", key="generate_translations", use_container_width=True):
        st.session_state["translations_ready"] = True
        st.session_state["scores_ready"] = False

    if st.session_state["translations_ready"]:
        translations = {
            "🧊 Traditional NLP": example["traditional"],
            "🤖 LLM (ChatGPT-style)": example["llm"],
        }

        st.markdown("#### Model outputs")
        trans_df = pd.DataFrame({"Model": translations.keys(), "Translation": translations.values()})
        st.dataframe(trans_df, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("### 3️⃣ Run BLEU and Compare Scores")
        st.markdown("See the numeric scores with a friendly interpretation and color coding.")

        if st.button("📊 Calculate BLEU scores", key="calc_bleu", use_container_width=True):
            st.session_state["scores_ready"] = True

        if st.session_state.get("scores_ready"):
            score_rows = _compute_scores(example["reference"], translations)
            score_df = pd.DataFrame(score_rows)
            st.dataframe(score_df, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("### 4️⃣ Highlight Differences (optional)")
        st.markdown("Spot the n-grams that overlap with the reference. Green = match, yellow = partial, red = missing.")
        if st.button("✨ Highlight differences", key="highlight_differences", use_container_width=True):
            st.session_state["show_highlights"] = True
        if st.session_state.get("show_highlights"):
            st.markdown("**Reference:**")
            st.markdown(example["reference"])
            st.markdown("**Traditional NLP output:**", help="Shorter, more literal phrasing can lose n-gram overlap.")
            st.markdown(
                _highlight_candidate(example["reference"], translations["🧊 Traditional NLP"]),
                unsafe_allow_html=True,
            )
            st.markdown("**LLM output:**", help="Tends to stay close to the reference phrasing.")
            st.markdown(
                _highlight_candidate(example["reference"], translations["🤖 LLM (ChatGPT-style)"]),
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### 🧠 Deep dive (optional)")
    with st.expander("How BLEU works + models in this lab"):
        st.markdown(
            """
            - **BLEU mechanics:** Counts matching n-grams between hypothesis and reference, averages precisions, and applies a brevity penalty (shorter outputs get penalized). We use smoothing to avoid zeros on short sentences.
            - **Why multiple references help:** Any reference can satisfy an n-gram match, so paraphrases get credit instead of being punished.
            - **Traditional baseline:** Represents phrase-based / statistical MT tendencies—literal word choices and occasional dropped words.
            - **LLM output:** Mirrors modern neural MT/LLM behavior with more fluent phrasing (akin to [OpenAI GPT-4o mini](https://platform.openai.com/docs/models/gpt-4o-mini)).
            - **Models used elsewhere in this demo:** [OpenAI GPT-4o mini](https://platform.openai.com/docs/models/gpt-4o-mini) for neural translations and [Helsinki-NLP/opus-mt-mul-en](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en) as a lightweight local baseline.
            - **Reading the colors:** Green spans align well with the reference, yellow are partial overlaps, and red are gaps—these are the same overlaps BLEU is counting.
            """
        )


if __name__ == "__main__":
    bleu_page()
