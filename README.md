# Multi-Page NLP & MT Streamlit Demo

This repository hosts a guided Streamlit app that walks a non-technical audience through tokenization, embeddings, translation, and transformer behavior using delegation to OpenAI-compatible APIs (including Bedrock endpoints).

## Getting Started (Local Development)

1. Copy `.env.example` to `.env` and populate:
   - `OPENAI_API_KEY`
   - `OPENAI_API_BASE` (e.g., Bedrock's OpenAI-compatible gateway)
   - Optional overrides: `OPENAI_EMBEDDING_MODEL` / `OPENAI_TRANSLATION_MODEL`
   - Optional `OPENAI_GENERATION_MODEL` (defaults to `gpt-4o-mini`)
2. Run `./start.sh` to create/activate the virtual environment, install dependencies, source the `.env`, and start Streamlit.
3. Navigate via the sidebar to explore each page (`Home`, `Sentiment Analysis`, `Translation`, `Tokenization`, `Embeddings`, `Transformer Lab`) with live calls to your configured models.

## Deploying to Streamlit Cloud

1. Push your code to GitHub (ensure `.env` and `secrets.toml` are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository
3. Set the main file path to: `app/1_Home.py`
4. In your app settings, click "Secrets" and add your configuration in TOML format:

```toml
OPENAI_API_KEY = "your-real-key"
OPENAI_API_BASE = "https://bedrock.us-east-1.amazonaws.com/openai"
OPENAI_EMBEDDING_MODEL = "amazon.titan-embed-text"
OPENAI_TRANSLATION_MODEL = "amazon.titan-translate"
OPENAI_GENERATION_MODEL = "gpt-4o-mini"
```

5. Deploy! The app will automatically use `st.secrets` when deployed and `.env` for local development.

## Structure

- `app/`: Streamlit entry points for each teaching page plus shared utilities
  - `1_Home.py`: Main entry point and navigation
  - `2_SentimentAnalysis.py`: Sentiment analysis and topic modeling
  - `3_Translation.py`: Translation sandbox with context
  - `4_Tokenization.py`: Tokenization playground
  - `5_Embeddings.py`: Embedding similarity explorer
  - `6_TransformerDemo.py`: Transformer insight studio
  - `utils.py`: Shared utilities (supports both st.secrets and .env)
- `.env.example`: Template for local development
- `app/.streamlit/secrets.toml.example`: Template for Streamlit secrets
- `requirements.txt`: Python dependencies
- `start.sh`: Bootstraps `.venv`, installs deps, exports `.env`, and runs the app

## Notes

- Secrets are managed via `st.secrets` (Streamlit Cloud) or `.env` (local development)
- Both `.env` and `secrets.toml` are ignored by `.gitignore` to prevent credential leaks
- Adjust the tokenizer, embedding, or translation models through the configuration to swap between vendors
