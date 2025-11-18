import streamlit as st

from utils import translate_text


def ensure_session_keys() -> None:
    """Initialize translation results storage."""
    for key in ("plain_translation", "context_translation"):
        if key not in st.session_state:
            st.session_state[key] = ""


def translation_page() -> None:
    """Guide users through translating text with and without extra context."""
    st.header("Translation Sandbox")
    st.markdown(
        """
        Step through translating a sentence before and after providing extra context.
        This highlights how transformers resolve ambiguity as more information arrives.
        """
    )
    ensure_session_keys()
    default_sentence = "I saw her duck while she was cleaning."
    sentence = st.text_area("Step 1 — Base sentence", value=default_sentence, height=140)
    language = st.selectbox(
        "Target language",
        ["Spanish", "French", "German", "Italian", "Portuguese"],
        index=0,
    )
    context = st.text_area(
        "Step 2 — Add supporting context or clarifications",
        value="This is a biology lab instruction about observing a duck.",
        height=120,
    )

    st.caption("First translate blindly, then add context to disambiguate the action.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Translate (Step 1)", key="translate_base"):
            st.session_state["plain_translation"] = translate_text(
                sentence, language
            )
    with col2:
        if st.button("Translate with context (Step 2)", key="translate_context"):
            st.session_state["context_translation"] = translate_text(
                sentence, language, context
            )

    if st.session_state["plain_translation"]:
        st.subheader("Step 1 result")
        st.write(st.session_state["plain_translation"])
        st.caption("Translation produced without additional context.")

    if st.session_state["context_translation"]:
        st.subheader("Step 2 result")
        st.write(st.session_state["context_translation"])
        st.caption("Translation produced after adding clarifying context.")
