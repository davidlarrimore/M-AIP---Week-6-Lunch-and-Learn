import streamlit as st

from utils import translate_text


EXAMPLES = [
    {
        "title": "✅ Normal Word Order",
        "icon": "✅",
        "sentence": "The student read the book before the lecture started.",
        "explanation": "Simple syntax keeps translation predictable and accurate.",
        "difficulty": "Easy",
        "color": "#10b981",
    },
    {
        "title": "⚠️ Reordered Clauses",
        "icon": "⚠️",
        "sentence": "Before the lecture started, the student read the book.",
        "explanation": "Transformers adapt using positional encodings to track clause order, but subtle shifts may occur.",
        "difficulty": "Medium",
        "color": "#f59e0b",
    },
    {
        "title": "❌ Scrambled Phrases",
        "icon": "❌",
        "sentence": "Lecture started student the book read before.",
        "explanation": "Translation drift increases significantly as positions become ambiguous. Legacy RNNs struggled even more with this.",
        "difficulty": "Hard",
        "color": "#ef4444",
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
    st.markdown("# 🔄 Word Order & Transformer Demo")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; margin-bottom: 1.5rem;'>
            <p style='margin: 0; color: #4b5563; font-size: 1.05rem;'>
                Discover why positional encodings matter in transformer models. See how the same
                tokens yield different meanings when reordered, and understand why attention
                mechanisms make transformers more robust than legacy RNNs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Configuration section
    col1, col2 = st.columns([2, 1])

    with col1:
        target_lang = st.selectbox(
            "🌍 Select Target Language",
            ["Spanish", "French", "German", "Italian", "Portuguese"],
            index=0,
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if "order_translations" not in st.session_state:
            st.session_state["order_translations"] = []

        if st.button("🚀 Translate All Examples", use_container_width=True, type="primary"):
            with st.spinner("🔄 Translating all examples..."):
                st.session_state["order_translations"] = render_examples(target_lang)

    st.markdown("---")

    # Results section
    if st.session_state["order_translations"]:
        st.markdown("### 📊 Translation Results")

        # Create tabs for each example
        tab1, tab2, tab3 = st.tabs([
            "✅ Normal Order",
            "⚠️ Reordered",
            "❌ Scrambled"
        ])

        tabs = [tab1, tab2, tab3]

        for idx, (tab, entry) in enumerate(zip(tabs, st.session_state["order_translations"])):
            with tab:
                # Example card
                st.markdown(
                    f"""
                    <div style='background: white; padding: 1.5rem; border-radius: 12px;
                    border-left: 5px solid {entry["color"]}; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    margin-bottom: 1.5rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                            <h3 style='margin: 0; color: #1f2937;'>{entry["title"]}</h3>
                            <span style='background: {entry["color"]}; color: white; padding: 0.4rem 1rem;
                            border-radius: 20px; font-size: 0.85rem; font-weight: 600;'>
                                {entry["difficulty"]}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Original and translation in columns
                col_orig, col_trans = st.columns(2)

                with col_orig:
                    st.markdown("#### 📝 Original (English)")
                    st.markdown(
                        f"""
                        <div style='background: #f9fafb; padding: 1.2rem; border-radius: 8px;
                        border: 2px solid #e5e7eb; min-height: 100px;'>
                            <p style='font-size: 1.05rem; color: #1f2937; margin: 0; line-height: 1.6;'>
                                "{entry["sentence"]}"
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_trans:
                    st.markdown(f"#### 🌐 Translation ({target_lang})")
                    st.markdown(
                        f"""
                        <div style='background: {entry["color"]}15; padding: 1.2rem; border-radius: 8px;
                        border: 2px solid {entry["color"]}; min-height: 100px;'>
                            <p style='font-size: 1.05rem; color: #1f2937; margin: 0; line-height: 1.6;'>
                                "{entry["translation"]}"
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Explanation
                st.markdown("#### 💡 Key Insight")
                st.info(entry["explanation"])

        # Summary metrics
        st.markdown("---")
        st.markdown("### 📈 Complexity Analysis")

        metric_cols = st.columns(3)

        for idx, (col, entry) in enumerate(zip(metric_cols, st.session_state["order_translations"])):
            with col:
                st.markdown(
                    f"""
                    <div style='background: white; padding: 1.2rem; border-radius: 10px;
                    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    border-top: 4px solid {entry["color"]};'>
                        <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{entry["icon"]}</div>
                        <div style='color: #6b7280; font-size: 0.85rem; margin-bottom: 0.3rem;'>
                            {entry["difficulty"]} Complexity
                        </div>
                        <div style='color: #1f2937; font-weight: 600;'>
                            {len(entry["sentence"].split())} words
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        st.info("💡 Click the 'Translate All Examples' button above to see how word order affects translation quality")

    # Educational section
    st.markdown("---")
    st.markdown("### 🧠 Understanding Transformers & Word Order")

    col_edu1, col_edu2, col_edu3 = st.columns(3)

    with col_edu1:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem; border-radius: 12px; color: white; height: 100%;'>
                <h4 style='color: white; margin-top: 0;'>🎯 Positional Encoding</h4>
                <p style='font-size: 0.95rem; opacity: 0.95; margin-bottom: 0;'>
                    Transformers learn where each token lives in the sentence,
                    allowing them to understand sequence order despite parallel processing.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_edu2:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            padding: 1.5rem; border-radius: 12px; color: white; height: 100%;'>
                <h4 style='color: white; margin-top: 0;'>👁️ Attention Mechanism</h4>
                <p style='font-size: 0.95rem; opacity: 0.95; margin-bottom: 0;'>
                    Models compare every token to every other token,
                    ensuring word order and relationships are preserved in translation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_edu3:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 1.5rem; border-radius: 12px; color: white; height: 100%;'>
                <h4 style='color: white; margin-top: 0;'>⚡ Translation Drift</h4>
                <p style='font-size: 0.95rem; opacity: 0.95; margin-bottom: 0;'>
                    Highly scrambled inputs can still confuse attention mechanisms,
                    demonstrating why proper sentence structure matters.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
