# Multi-Page NLP & MT Streamlit Demo

This repository hosts a guided Streamlit app that walks a non-technical audience through tokenization, embeddings, translation, and transformer behavior using delegation to OpenAI-compatible APIs (including Bedrock endpoints).

## Getting started

1. Copy `.env.example` to `.env` and populate:
   - `OPENAI_API_KEY`
   - `OPENAI_API_BASE` (e.g., Bedrock’s OpenAI-compatible gateway)
   - Optional overrides: `OPENAI_EMBEDDING_MODEL` / `OPENAI_TRANSLATION_MODEL`
   - Optional `OPENAI_GENERATION_MODEL` (defaults to `gpt-4o-mini`)
2. Run `./start.sh` to create/activate the virtual environment, install dependencies, source the `.env`, and start Streamlit.
3. Navigate via the sidebar to explore each page (`Tokenization`, `Embeddings`, `Translation`, `Word Order`, `Transformer Lab`) with live calls to your configured models.

## Structure

- `demo_app/`: Streamlit entry points for each teaching page plus shared utilities.
- `.env.example`: Template for connecting to an OpenAI/Bedrock endpoint.
- `requirements.txt`: Python dependencies.
- `start.sh`: Bootstraps `.venv`, installs deps, exports `.env`, and runs the app.

## Notes

- Populate `.env` with secure values; it is ignored by `.gitignore`.
- Adjust the tokenizer, embedding, or translation models through the environment variables to swap between vendors.
