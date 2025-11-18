import streamlit as st

from Tokenization import tokenization_page
from Embeddings import embeddings_page
from Translation import translation_page
from WordOrder import word_order_page

PAGE_NAVIGATION = {
    "Home / Overview": "Home / Overview",
    "Tokenization Playground": "Tokenization Playground",
    "Embedding Similarity Explorer": "Embedding Similarity Explorer",
    "Translation Sandbox": "Translation Sandbox",
    "Word Order & Transformer Demo": "Word Order & Transformer Demo",
}


def show_home() -> None:
    """Render the landing overview with objectives and instructions."""
    st.title("Multi-Page NLP & Machine Translation Demo")
    st.markdown(
        """
        This lightweight Streamlit app demystifies how GPT-style models tokenize,
        embed, and translate language. Each page is a guided step in the NLP workflow:
        """
    )
    st.markdown(
        """
        - **Home / Overview:** Learn the teaching goals and controls.
        - **Tokenization Playground:** See how text becomes tokens.
        - **Embedding Similarity Explorer:** Compare vector distances.
        - **Translation Sandbox:** Translate with and without context.
        - **Word Order & Transformer Demo:** Observe translation drift.
        """
    )
    st.divider()

    st.subheader("Objectives for presenters")
    st.markdown(
        """
        1. Keep the interface friendly and non-technical.
        2. Use big, clear controls that show each step of the pipeline.
        3. Avoid surprises by keeping answers concise and deterministic.
        """
    )
    st.subheader("Getting ready to demo")
    st.markdown(
        """
        - Install the required packages: `streamlit`, `openai`, `numpy`, `tiktoken`, `plotly`.
        - Set your OpenAI key: `export OPENAI_API_KEY="your-key"`.
        - Run the app from the `demo_app` directory:
          `streamlit run Home.py`
        - Use the sidebar to jump between the scenarios in the order presented.
        """
    )
    st.success("Once the API key is set, every page will show real-time responses.")


def main() -> None:
    """Route to the selected page."""
    st.set_page_config(page_title="NLP & MT Demo", layout="wide")
    nav_options = list(PAGE_NAVIGATION.keys())
    selection = st.sidebar.radio("Navigate the demo", nav_options, index=0)
    if selection == "Home / Overview":
        show_home()
    elif selection == "Tokenization Playground":
        tokenization_page()
    elif selection == "Embedding Similarity Explorer":
        embeddings_page()
    elif selection == "Translation Sandbox":
        translation_page()
    elif selection == "Word Order & Transformer Demo":
        word_order_page()


if __name__ == "__main__":
    main()
