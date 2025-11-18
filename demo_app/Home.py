import streamlit as st

from Tokenization import tokenization_page
from Embeddings import embeddings_page
from Translation import translation_page
from WordOrder import word_order_page
from TransformerDemo import transformer_page

PAGE_NAVIGATION = {
    "🏠 Home": ("Home / Overview", "🏠"),
    "🔤 Tokenization": ("Tokenization Playground", "🔤"),
    "🎯 Embeddings": ("Embedding Similarity Explorer", "🎯"),
    "🌐 Translation": ("Translation Sandbox", "🌐"),
    "🔄 Word Order": ("Word Order & Transformer Demo", "🔄"),
    "⚡ Transformer Lab": ("Transformer Insight Studio", "⚡"),
}


def apply_custom_css() -> None:
    """Apply custom CSS for modern, responsive design."""
    st.markdown(
        """
        <style>
        /* Main container styling */
        .main {
            padding: 2rem 1rem;
        }

        /* Custom card styling */
        .custom-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            color: white;
        }

        .custom-card h1 {
            color: white;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }

        .custom-card p {
            font-size: 1.1rem;
            opacity: 0.95;
        }

        /* Feature card styling */
        .feature-card {
            background: white;
            border: 2px solid #f0f2f6;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
            border-color: #667eea;
        }

        .feature-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .feature-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }

        .feature-desc {
            color: #6b7280;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Info box styling */
        .info-box {
            background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }

        .info-box h3 {
            color: #667eea;
            margin-bottom: 0.8rem;
            font-size: 1.3rem;
        }

        /* Metric styling */
        .custom-metric {
            background: white;
            padding: 1.2rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        /* Button enhancements */
        .stButton>button {
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }

        [data-testid="stSidebar"] .stRadio > label {
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
        }

        /* Radio button styling */
        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.5rem;
        }

        [data-testid="stSidebar"] .stRadio label {
            background: rgba(255, 255, 255, 0.1);
            padding: 0.8rem 1rem;
            border-radius: 8px;
            transition: all 0.3s ease;
            color: white;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .custom-card h1 {
                font-size: 1.8rem;
            }

            .custom-card p {
                font-size: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_home() -> None:
    """Render the landing overview with objectives and instructions."""
    # Hero section
    st.markdown(
        """
        <div class="custom-card">
            <h1>🤖 NLP & Machine Translation Demo</h1>
            <p>Explore how GPT-style models tokenize, embed, and translate language through interactive demonstrations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ✨ Interactive Features")

    # Feature cards in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔤</div>
                <div class="feature-title">Tokenization Playground</div>
                <div class="feature-desc">Watch text transform into byte-pair tokens in real-time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">Translation Sandbox</div>
                <div class="feature-desc">Compare translations with and without context</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Embedding Explorer</div>
                <div class="feature-desc">Visualize semantic similarity through vector comparisons</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔄</div>
                <div class="feature-title">Word Order Analysis</div>
                <div class="feature-desc">Observe how positional encoding affects translation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Two-column layout for objectives and setup
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(
            """
            <div class="info-box">
                <h3>🎯 Presenter Objectives</h3>
                <ul style="line-height: 1.8;">
                    <li>Maintain a friendly, non-technical interface</li>
                    <li>Use clear controls showing each pipeline step</li>
                    <li>Provide concise, deterministic responses</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div class="info-box">
                <h3>🚀 Quick Start Guide</h3>
                <ol style="line-height: 1.8;">
                    <li>Install packages: <code>streamlit openai numpy tiktoken plotly</code></li>
                    <li>Set your API key in <code>.env</code> file</li>
                    <li>Run: <code>streamlit run Home.py</code></li>
                    <li>Navigate using the sidebar</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.success("✅ Once the API key is configured, all pages will show real-time AI responses")


def main() -> None:
    """Route to the selected page."""
    st.set_page_config(
        page_title="NLP & MT Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    apply_custom_css()

    # Sidebar navigation with icons
    st.sidebar.markdown("## 🧭 Navigation")
    st.sidebar.markdown("---")

    nav_options = list(PAGE_NAVIGATION.keys())
    selection = st.sidebar.radio("", nav_options, index=0, label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding: 1rem; color: white;'>
            <p style='font-size: 0.9rem; opacity: 0.8;'>🤖 Powered by OpenAI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Route to pages
    if selection == "🏠 Home":
        show_home()
    elif selection == "🔤 Tokenization":
        tokenization_page()
    elif selection == "🎯 Embeddings":
        embeddings_page()
    elif selection == "🌐 Translation":
        translation_page()
    elif selection == "🔄 Word Order":
        word_order_page()
    elif selection == "⚡ Transformer Lab":
        transformer_page()


if __name__ == "__main__":
    main()
