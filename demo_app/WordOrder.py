import streamlit as st

from utils import translate_text


EXAMPLES = [
    {
        "title": "Normal word order",
        "sentence": "The student read the book before the lecture started.",
        "explanation": "Simple syntax keeps translation predictable.",
    },
    {
        "title": "Reordered clauses",
        "sentence": "Before the lecture started, the student read the book.",
        "explanation": "Transformers adapt, but they rely on positional encodings to track clause order.",
    },
    {
        "title": "Scrambled phrases",
        "sentence": "Lecture started student the book read before.",
        "explanation": "Drift increases as positions become ambiguous; older RNNs struggled more with this.",
    },
]


def render_examples(target_lang: str) -> list[dict]:
    """Translate each predefined sentence into the selected language."""
    results = []
    for example in EXAMPLES:
        translation = translate_text(example["sentence"], target_lang)
        results.append(
            {
                **example,
                "translation": translation,
            }
        )
    return results


def word_order_page() -> None:
    """Demonstrate how transformer translations shift as word order changes."""
    st.header("Word Order & Transformer Behavior Demo")
    st.markdown(
        """
        Show why positional encodings matter: the same tokens can yield different meanings when reordered.
        Explain why attention mechanisms make these models more robust than legacy RNNs.
        """
    )
    target_lang = st.selectbox(
        "Target language for all examples",
        ["Spanish", "French", "German"],
        index=0,
    )

    if "order_translations" not in st.session_state:
        st.session_state["order_translations"] = []

    if st.button("Translate the sample sentences"):
        with st.spinner("Requesting translations..."):
            st.session_state["order_translations"] = render_examples(target_lang)

    if st.session_state["order_translations"]:
        for entry in st.session_state["order_translations"]:
            with st.expander(entry["title"]):
                st.markdown(f"**Original:** {entry['sentence']}")
                st.markdown(f"**Translation:** {entry['translation']}")
                st.caption(entry["explanation"])
    else:
        st.info("Click the button to see how each change in word order moves the translation.")

    st.divider()
    st.markdown(
        """
        - **Positional encoding:** Transformers learn where each token lives in the sentence.
        - **Attention:** Models compare every token to every other token so order is remembered.
        - **Drift:** Highly scrambled inputs can still confuse attention, showing why structure matters.
        """
    )
