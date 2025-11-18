import streamlit as st

from utils import translate_text


def ensure_session_keys() -> None:
    """Initialize translation results storage."""
    for key in ("plain_translation", "context_translation"):
        if key not in st.session_state:
            st.session_state[key] = ""


def translation_page() -> None:
    """Guide users through translating text with and without extra context."""
    st.markdown("# 🌐 Translation Sandbox")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; margin-bottom: 1.5rem;'>
            <p style='margin: 0; color: #4b5563; font-size: 1.05rem;'>
                Discover how context transforms translation quality. Compare translations
                before and after providing additional context to see how transformers
                resolve ambiguity and improve accuracy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ensure_session_keys()

    # Input section
    st.markdown("### ⚙️ Configuration")

    col1, col2 = st.columns([3, 1])

    with col1:
        default_sentence = "I saw her duck while she was cleaning."
        sentence = st.text_area(
            "📝 Source Sentence",
            value=default_sentence,
            height=100,
            help="Enter the text you want to translate",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        language = st.selectbox(
            "🌍 Target Language",
            ["Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Chinese"],
            index=0,
        )

    st.markdown("### 🎯 Context (Optional)")
    context = st.text_area(
        "Add context to improve translation accuracy",
        value="This is a biology lab instruction about observing a duck.",
        height=80,
        help="Provide additional context to help disambiguate the translation",
    )

    st.markdown("---")

    # Action buttons
    st.markdown("### 🚀 Actions")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

    with col_btn1:
        if st.button("1️⃣ Translate Without Context", use_container_width=True, type="secondary"):
            with st.spinner("Translating..."):
                st.session_state["plain_translation"] = translate_text(
                    sentence, language
                )

    with col_btn2:
        if st.button("2️⃣ Translate With Context", use_container_width=True, type="primary"):
            with st.spinner("Translating with context..."):
                st.session_state["context_translation"] = translate_text(
                    sentence, language, context
                )

    with col_btn3:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state["plain_translation"] = ""
            st.session_state["context_translation"] = ""
            st.rerun()

    # Results section - side by side comparison
    if st.session_state["plain_translation"] or st.session_state["context_translation"]:
        st.markdown("---")
        st.markdown("### 📊 Translation Comparison")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown(
                """
                <div style='text-align: center; padding: 0.8rem; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                color: white; border-radius: 10px 10px 0 0; font-weight: 600;'>
                    1️⃣ Without Context
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state["plain_translation"]:
                st.markdown(
                    f"""
                    <div style='background: white; padding: 1.5rem; border-radius: 0 0 10px 10px;
                    border: 2px solid #f59e0b; min-height: 150px; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);'>
                        <div style='font-size: 1.1rem; color: #1f2937; line-height: 1.6;'>
                            "{st.session_state["plain_translation"]}"
                        </div>
                        <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
                        color: #6b7280; font-size: 0.9rem;'>
                            ℹ️ Translation without additional context
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style='background: #f9fafb; padding: 1.5rem; border-radius: 0 0 10px 10px;
                    border: 2px dashed #d1d5db; min-height: 150px; display: flex;
                    align-items: center; justify-content: center;'>
                        <div style='color: #9ca3af; text-align: center;'>
                            Click "Translate Without Context" to see results
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with result_col2:
            st.markdown(
                """
                <div style='text-align: center; padding: 0.8rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; border-radius: 10px 10px 0 0; font-weight: 600;'>
                    2️⃣ With Context
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state["context_translation"]:
                st.markdown(
                    f"""
                    <div style='background: white; padding: 1.5rem; border-radius: 0 0 10px 10px;
                    border: 2px solid #667eea; min-height: 150px; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);'>
                        <div style='font-size: 1.1rem; color: #1f2937; line-height: 1.6;'>
                            "{st.session_state["context_translation"]}"
                        </div>
                        <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
                        color: #6b7280; font-size: 0.9rem;'>
                            ✅ Translation with clarifying context
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style='background: #f9fafb; padding: 1.5rem; border-radius: 0 0 10px 10px;
                    border: 2px dashed #d1d5db; min-height: 150px; display: flex;
                    align-items: center; justify-content: center;'>
                        <div style='color: #9ca3af; text-align: center;'>
                            Click "Translate With Context" to see results
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Show difference if both translations exist
        if st.session_state["plain_translation"] and st.session_state["context_translation"]:
            if st.session_state["plain_translation"] != st.session_state["context_translation"]:
                st.success("✨ **Key Insight:** Notice how the translation changed with added context! This demonstrates how transformers use additional information to resolve ambiguity.")
            else:
                st.info("💡 **Observation:** The translations are the same. Try a more ambiguous sentence or different context!")

    # Educational info
    with st.expander("ℹ️ Why Context Matters in Translation"):
        st.markdown(
            """
            **Understanding Context in Machine Translation:**

            - **Ambiguity Resolution**: Many words have multiple meanings (e.g., "duck" as a noun vs. verb)
            - **Transformer Attention**: Models use context to determine the most likely meaning
            - **Context Window**: Additional information helps the model make better decisions

            **Best Practices:**
            - Provide domain-specific context when translating technical content
            - Include relevant background information for ambiguous phrases
            - Specify the intended audience or use case

            **Example Ambiguous Phrases:**
            - "The bass was too loud" (fish or music?)
            - "I'll meet you at the bank" (river or financial institution?)
            - "Time flies like an arrow" (literal or metaphorical?)
            """
        )
