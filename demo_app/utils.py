import os

import numpy as np
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from tiktoken import Encoding
from typing import List, Tuple


load_dotenv()

EMBEDDING_MODEL_ENV = "OPENAI_EMBEDDING_MODEL"
TRANSLATION_MODEL_ENV = "OPENAI_TRANSLATION_MODEL"
GENERATION_MODEL_ENV = "OPENAI_GENERATION_MODEL"
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text"
DEFAULT_TRANSLATION_MODEL = "amazon.titan-translate"
DEFAULT_GENERATION_MODEL = "gpt-4o-mini"


def _env_value(key: str, default: str = "", required: bool = False) -> str:
    """Fetch an environment variable with optional defaulting/validation."""
    value = os.environ.get(key, "").strip()
    if required and not value:
        raise RuntimeError(f"{key} must be set via .env or the environment.")
    return value or default


@st.cache_resource
def get_openai_client() -> OpenAI:
    """Reuse a single OpenAI client throughout the app to avoid reinitialization."""
    api_key = _env_value("OPENAI_API_KEY", required=True)
    client_kwargs = {"api_key": api_key}
    api_base = _env_value("OPENAI_API_BASE")
    if api_base:
        client_kwargs["base_url"] = api_base
    return OpenAI(**client_kwargs)


@st.cache_resource
def get_tokenizer() -> Encoding:
    """Prefer the GPT-4o tokenizer but fall back to cl100k_base if unavailable."""
    try:
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except (KeyError, ValueError):
        return tiktoken.get_encoding("cl100k_base")


@st.cache_data
def tokenize_text(text: str) -> Tuple[List[int], List[str]]:
    """Return token ids and decoded token strings for the provided text."""
    if not text:
        return [], []
    encoding = get_tokenizer()
    tokens = encoding.encode(text, disallowed_special=())
    readable_tokens = [encoding.decode([token]) for token in tokens]
    return tokens, readable_tokens


@st.cache_data
def embed_text(text: str) -> np.ndarray:
    """Request embeddings from OpenAI for the provided text."""
    cleaned = text.strip()
    if not cleaned:
        return np.array([])
    client = get_openai_client()
    embedding_model = _env_value(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)
    response = client.embeddings.create(
        model=embedding_model,
        input=cleaned,
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity while guarding against zero vectors."""
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    dot = float(np.dot(vec_a, vec_b))
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return dot / denom if denom else 0.0


def translate_text(text: str, target_lang: str, context: str = "") -> str:
    """Ask GPT-4o-mini to translate text, optionally including extra context."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    client = get_openai_client()
    translation_model = _env_value(TRANSLATION_MODEL_ENV, DEFAULT_TRANSLATION_MODEL)
    prompt = f"Translate the following sentence into {target_lang}. Only return the translation."
    if context.strip():
        prompt += f" Context: {context.strip()}"
    prompt += f"\nSentence: {cleaned}"
    response = client.chat.completions.create(
        model=translation_model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise translator for educational demos. Keep the answer short.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def get_generation_model() -> str:
    """Return the configured generation model name."""
    return _env_value(GENERATION_MODEL_ENV, DEFAULT_GENERATION_MODEL)
