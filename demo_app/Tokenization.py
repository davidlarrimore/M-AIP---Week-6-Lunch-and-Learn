import html

import streamlit as st
import streamlit.components.v1 as components

from utils import tokenize_text


def tokenization_page() -> None:
    """Provide an interactive subword tokenization explorer."""
    # Page header
    st.markdown("# 🔤 Tokenization Playground")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; margin-bottom: 1.5rem;'>
            <p style='margin: 0; color: #4b5563; font-size: 1.05rem;'>
                Watch how GPT-style tokenizers break text into byte-pair tokens.
                Enter any sentence and see both the human-readable token fragments and their numeric IDs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        sample_text = "The quick brown fox jumps over the lazy dog."
        text = st.text_area(
            "📝 Enter your text",
            value=sample_text,
            height=120,
            help="Type or paste any text to see how it's tokenized",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        tokens, readable = tokenize_text(text)

        # Enhanced metric display
        st.markdown(
            f"""
            <div style='background: white; padding: 1.5rem; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; margin-top: 0.5rem;'>
                <div style='color: #9ca3af; font-size: 0.85rem; text-transform: uppercase;
                letter-spacing: 0.5px; margin-bottom: 0.5rem;'>Token Count</div>
                <div style='color: #667eea; font-size: 2.5rem; font-weight: 700;'>{len(tokens)}</div>
                <div style='color: #6b7280; font-size: 0.9rem; margin-top: 0.5rem;'>
                    {len(text)} characters
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not tokens:
        st.info("💡 Start typing to see subword tokens and IDs appear below")
        return

    st.markdown("---")
    st.markdown("### 🏷️ Token Visualization")

    # Build the styled token chips
    tokens_html = ""
    for idx, (token_id, token_text) in enumerate(zip(tokens, readable)):
        # Escape special characters for HTML to keep labels visible
        safe_token = html.escape(token_text)
        tokens_html += f"""
        <span class="token-chip">
            <span class="token-number">#{idx + 1}</span>
        <span class="token-text">{safe_token}</span>
        <span class="token-id">ID: {token_id}</span>
        </span>
        """

    style_and_tokens = f"""
    <style>
    .token-wrapper {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        line-height: 2.4;
    }}
    .token-chip {{
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.45rem 0.9rem;
        border-radius: 18px;
        font-family: 'Monaco', 'Courier New', monospace;
        font-size: 0.9rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .token-chip:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(102, 126, 234, 0.4);
    }}
    .token-number {{
        background: rgba(255, 255, 255, 0.2);
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.75rem;
        margin-right: 0.35rem;
    }}
    .token-text {{
        font-weight: 600;
    }}
    .token-id {{
        font-size: 0.75rem;
        opacity: 0.85;
        margin-left: 0.5rem;
    }}
    </style>
    <div class="token-wrapper">
        {tokens_html}
    </div>
    """
    components.html(
        style_and_tokens,
        height=min(520, max(180, 90 + len(tokens) * 36)),
        scrolling=True,
    )

    # Information section
    st.markdown("---")
    with st.expander("ℹ️ Understanding Tokenization"):
        st.markdown(
            """
            **Key Concepts:**
            - **Subword Tokens**: Words are broken into smaller pieces that can be reused
            - **Token IDs**: Each token has a unique numeric identifier
            - **Case Sensitivity**: Capitalization affects tokenization
            - **Punctuation**: Special characters often become separate tokens

            **Why This Matters:**
            - Token count affects model cost and context limits
            - Understanding tokens helps optimize prompts
            - Different languages may tokenize differently
            """
        )
