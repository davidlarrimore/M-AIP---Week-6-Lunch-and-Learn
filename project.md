A Multi-Page Streamlit Demo App for Teaching NLP & Machine Translation

Executive Summary

This white paper describes the architecture, requirements, and implementation outline for a simple, interactive Streamlit application designed to educate a non-technical audience on key concepts in Natural Language Processing (NLP) and Machine Translation (MT).

The app demonstrates:
	•	Tokenization
	•	Embeddings & Vector Similarity
	•	Real-time Machine Translation
	•	Context Effects in Transformers

Each concept is presented as a separate page, with step-by-step UI controls so users can observe how models convert human language into numbers, vectors, and translated outputs.

⸻

🎯 Objectives
	1.	Deliver a simple, safe, stable demo environment suitable for live webinars.
	2.	Provide hands-on, visual, sequential demonstrations that show how NLP works.
	3.	Keep technical complexity hidden while allowing users to “peek” behind the curtain.
	4.	Ensure code is minimal, dependencies light, and API usage predictable.

⸻

🏗️ High-Level Architecture

Streamlit App
│
├── Home / Overview Page
│
├── Tokenization Playground
│     - Subword tokenization
│     - Token IDs & visual chips
│
├── Embedding Similarity Explorer
│     - Vector generation
│     - Cosine similarity
│
├── Translation Sandbox
│     - Basic translation
│     - Context dependency
│
└── Word Order & Transformer Behavior Demo
      - Reordering sentences
      - Observing translation drift

All NLP and MT logic is handled via:
	•	OpenAI Models (recommended for simplicity/stability)
	•	Or HuggingFace Transformers (optional alternative)

⸻

📦 Tools, Libraries & Models

Python Libraries

Library	Purpose
streamlit	UI framework
openai	Embeddings, translation, tokenization
numpy	Cosine similarity computation
tiktoken	Local GPT-style tokenization
plotly (optional)	Fancy visualization of vector similarity


⸻

Models Used

1. Tokenization
	•	tiktoken (local)
	•	Fast, no API calls
	•	Matches GPT tokenizer behavior

2. Embeddings
	•	text-embedding-3-small (OpenAI)
	•	1536-dimensional embedding
	•	Low cost, fast
	•	Great for semantic similarity demos

3. Translation
	•	gpt-4.1-mini or gpt-4o-mini
	•	Multilingual
	•	Fast enough for live demos
	•	Handles ambiguity & context well

4. Context/Transformer Behavior
	•	Same translation models above
	•	Demonstrates attention implicitly via improved contextual translation

⸻

⚙️ Installation Requirements

Python Version

Python 3.9+

Install Dependencies

pip install streamlit openai numpy tiktoken plotly

Environment Variables

Set the OpenAI-compatible Bedrock endpoint along with the model names you want to demo:

```
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://bedrock.us-east-1.amazonaws.com/openai"
export OPENAI_EMBEDDING_MODEL="amazon.titan-embed-text"
export OPENAI_TRANSLATION_MODEL="amazon.titan-translate"
```


⸻

🗂️ Application Structure

demo_app/
│
├── Home.py
├── Tokenization.py
├── Embeddings.py
├── Translation.py
└── WordOrder.py

Running:

streamlit run Home.py


⸻

📄 Page 1 — Home / Overview

Purpose

Introduce concepts in plain English:
	•	What is NLP?
	•	What is machine translation?
	•	Why are tokens, embeddings, and context important?

Core Elements
	•	Simple markdown explaining the demo.
	•	Navigation instructions.
	•	No API calls.

Implementation Notes

Provide a conceptual graphic (ASCII optional):

TEXT → TOKENS → VECTORS → TRANSFORMER → TRANSLATION


⸻

📄 Page 2 — Tokenization Playground

Goal

Show users how words are broken into subword tokens and then mapped to token IDs.

Requirements
	•	tiktoken for tokenizing.
	•	Visual color blocks for each token.
	•	Step-by-step UX.

User Experience Flow
	1.	User enters a word or short sentence.
	2.	They click “Tokenize”.
	3.	Show:
	•	Token pieces
	•	Token IDs
	•	Total number of tokens
	4.	Step 2 button (optional):
	•	“Show how this affects cost / model processing”

Key Code Concepts

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode(user_input)
pieces = enc.encode(user_input, allowed_special=set(), disallowed_special=())


⸻

📄 Page 3 — Embedding Similarity Explorer

Goal

Show users how models turn text into vectors, and how cosine similarity measures meaning.

Requirements
	•	OpenAI embeddings API
	•	Numpy for vector math
	•	Simple bar chart for similarity score

User Experience Flow
	1.	User enters Sentence A.
	2.	User enters Sentence B.
	3.	Click “Compute Meaning Similarity”.
	4.	App shows:
	•	A percent similarity
	•	A simple color-coded interpretation
	•	Optional: vector length / shape info
	5.	Optional step:
	•	“Show embedding vector” (collapsed by default)

Similarity Calculation

import numpy as np

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


⸻

📄 Page 4 — Translation Sandbox

Goal

Demonstrate MT performance, ambiguity, and context sensitivity.

Requirements
	•	GPT model capable of multilingual translation
	•	Step-by-step interface to show incremental improvement with added context

User Experience Flow

Step 1 — Basic Translation
	1.	User enters text.
	2.	Selects target language.
	3.	Click Translate.
	4.	Details shown:
	•	Translation
	•	Confidence explanation
	•	Whether sentence is ambiguous

Step 2 — Add Context
	5.	User enters additional context (“He was an astronomer”).
	6.	Click Re-translate with context.
	7.	Show difference side-by-side.

Example Prompt

Translate this sentence into Spanish. Only provide the translation.
Sentence: {text}


⸻

📄 Page 5 — Word Order & Transformer Behavior

Goal

Show how word order affects meaning and translation quality.

Requirements
	•	Same translation model
	•	Preloaded examples

User Experience Flow
	1.	User clicks a button:
	•	Example 1: Normal word order
	•	Example 2: Reordered clauses
	•	Example 3: Highly scrambled sentence
	2.	App displays:
	•	Original sentence
	•	Translation
	•	Short explanation why translation drifted

Educational Outcomes
	•	Positional encoding concepts
	•	Attention robustness and its limits
	•	Why transformers outperform older RNN/LSTM systems

⸻

🧪 Core Demo Code (Modular Snippets)

Below are reusable abstractions you’ll implement once and share across pages:

Embedding Function

from openai import OpenAI
client = OpenAI()

def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

Translation Function

def translate(text, target_lang):
    prompt = f"Translate this into {target_lang}. Only return the translation: {text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


⸻

🎨 UI/UX Principles for Non-Technical Audiences
	•	Use big buttons, clear labels.
	•	Hide complexity behind expandable sections.
	•	Use color-coded outputs:
	•	Green = similar meaning
	•	Yellow = moderate similarity
	•	Red = different
	•	Include one-sentence explanations under every result.
	•	Always show the workflow:
Step 1 → Step 2 → Step 3

⸻

📑 Deployment Options

Local

streamlit run Home.py

Cloud Options
	•	Streamlit Cloud
	•	GitHub Codespaces
	•	HuggingFace Spaces
	•	Docker container on a cloud VM

Use OpenAI API keys stored in environment variables only.

⸻

🔚 Conclusion

This multi-page Streamlit app provides a simple, visually engaging, and non-technical-friendly platform for demonstrating the key concepts behind modern NLP and machine translation.

By structuring each page as a guided, step-by-step scenario, you ensure users not only see the outputs, but actually understand how machine intelligence processes and transforms language.

⸻

If you’d like, I can now generate:

✅ Full working Streamlit app code (all pages)
✅ Matching slide deck for your webinar
✅ A one-page cheat sheet for attendees

Just tell me what you want next!
