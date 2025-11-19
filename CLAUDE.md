# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-page Streamlit application that demonstrates NLP and machine translation concepts to non-technical audiences. The app uses OpenAI-compatible APIs (including AWS Bedrock) for tokenization, embeddings, translation, sentiment analysis, and transformer demonstrations.

## Development Commands

### Starting the Application
```bash
./start.sh
```
This script:
- Creates/activates Python virtual environment (`.venv`)
- Installs dependencies from `requirements.txt`
- Exports variables from `.env`
- Launches Streamlit at `app/1_Home.py`

### Manual Development Workflow
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/1_Home.py
```

### Testing
```bash
pytest tests
```
Add new test cases under `tests/` for utility functions, especially in `utils.py`.

## Architecture

### Configuration Management
The app supports two deployment modes with unified configuration handling in `app/utils.py`:

1. **Local Development**: Uses `.env` file loaded via `python-dotenv`
2. **Streamlit Cloud**: Uses `st.secrets` (configured via Streamlit dashboard)

The `_env_value()` function in `utils.py` tries `st.secrets` first, then falls back to `os.environ`, enabling seamless deployment across environments.

### Page Organization
Streamlit pages are numbered to control sidebar ordering:
- `app/1_Home.py` - Main entry point with navigation and landing page
- `app/2_SentimentAnalysis.py` - Sentiment analysis and topic modeling with VADER and traditional NLP
- `app/3_Translation.py` - Translation sandbox with context awareness
- `app/4_Tokenization.py` - Tokenization playground
- `app/5_Embeddings.py` - Embedding similarity explorer
- `app/6_TransformerDemo.py` - Transformer insight studio

New pages should follow the pattern `app/N_PageName.py` where N is the next number in sequence.

### Dynamic Module Loading
`1_Home.py` uses `importlib.util.spec_from_file_location()` to dynamically import page functions from numbered files. Each page module exports a function (e.g., `sentiment_analysis_page()`, `translation_page()`) that renders that page's content.

### Shared Utilities (`app/utils.py`)
Core reusable functions:
- `get_openai_client()` - Cached OpenAI client initialization with configurable base URL
- `get_tokenizer()` - Returns GPT-4o-mini tokenizer (falls back to cl100k_base)
- `tokenize_text()` - Tokenizes text and returns both token IDs and readable strings
- `embed_text()` - Generates embeddings via OpenAI API
- `translate_text()` - Translation with optional context
- `translate_with_source_language()` - Translation specifying source language
- `cosine_similarity()` - Vector similarity calculation

All functions use `@st.cache_resource` or `@st.cache_data` for performance.

## Configuration

### Required Environment Variables
```bash
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://bedrock.us-east-1.amazonaws.com/openai  # or OpenAI endpoint
```

### Optional Model Overrides
```bash
OPENAI_EMBEDDING_MODEL=amazon.titan-embed-text
OPENAI_TRANSLATION_MODEL=amazon.titan-translate
OPENAI_GENERATION_MODEL=gpt-4o-mini
```

### Local Setup
1. Copy `.env.example` to `.env`
2. Populate with your credentials
3. Run `./start.sh`

### Streamlit Cloud Deployment
1. Set main file to `app/1_Home.py`
2. Add secrets in TOML format via Streamlit dashboard
3. `.env` and `secrets.toml` are gitignored - never commit credentials

## Code Style

- Python 3.9+ with 4-space indentation
- Type hints for public functions in `utils.py`
- Docstrings stating intent and I/O
- `snake_case` for functions/variables, `CamelCase` for classes
- Preserve numeric prefixes on page files for navigation ordering

## Sentiment Analysis Page Architecture

The `2_SentimentAnalysis.py` page uses traditional NLP techniques rather than LLMs:
- **VADER**: Rule-based sentiment analysis optimized for social media text
- **Idiom preprocessing**: Custom regex-based replacement of context-dependent phrases (e.g., "to die for" → "absolutely wonderful") before sentiment analysis
- **Topic modeling**: Uses both LDA (Latent Dirichlet Allocation) via gensim and NMF (Non-negative Matrix Factorization) via scikit-learn
- **Dataset**: Amazon product reviews loaded via HuggingFace `datasets` library
- **Keyword extraction**: RAKE (Rapid Automatic Keyword Extraction) for identifying key phrases

## Testing Strategy

- Write deterministic tests for data prep, tokenization, and scoring logic
- Mock external API calls using pytest fixtures
- Mirror filenames with `test_*.py` modules under `tests/`
- Interactive app serves as smoke test - verify all sidebar pages render without exceptions after changes

## Git Workflow

Commit messages follow imperative style matching repository history:
- "Improve sentiment model, dashboard, and UI scaling"
- "Add sentiment analysis and topic modeling page"

No prefixes like "feat:" or "fix:" - use descriptive imperative sentences (~72 chars).
