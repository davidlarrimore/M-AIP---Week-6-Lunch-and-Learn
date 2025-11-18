import streamlit as st

from utils import tokenize_text


def tokenization_page() -> None:
    """Provide an interactive subword tokenization explorer."""
    st.header("Tokenization Playground")
    st.markdown(
        """
        Watch how GPT-style tokenizers break text into byte-pair tokens. Enter any
        sentence and see both the human-readable token fragments and their numeric IDs.
        """
    )
    sample_text = "The quick brown fox jumps over the lazy dog."
    text = st.text_area("Play with a sentence", value=sample_text, height=160)
    tokens, readable = tokenize_text(text)
    st.metric("Token count", len(tokens))
    if not tokens:
        st.info("Start typing to unlock subword tokens and IDs.")
        return

    st.subheader("Token chips")
    columns = st.columns(6)
    for idx, (token_id, token_text) in enumerate(zip(tokens, readable)):
        column = columns[idx % len(columns)]
        column.markdown(
            f"<span style='display:block; border-radius:12px; "
            f"padding:6px 10px; background:#f2f2ff; margin-bottom:4px;'>"
            f"<strong>#{idx + 1}</strong> `{token_text}`<br>"
            f"<small>ID {token_id}</small></span>",
            unsafe_allow_html=True,
        )

    st.caption("Subword tokens can be shared across words; mixing casing or punctuation changes the ID stream.")
