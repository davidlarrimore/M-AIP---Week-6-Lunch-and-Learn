# Repository Guidelines

## Project Structure & Module Organization
- `app/` holds Streamlit entry points numbered `1_*` through `6_*` to control sidebar order; keep new instructional pages in this directory and suffix them with short, action-oriented titles (e.g., `7_NewTopic.py`).
- Shared helpers live in `app/utils.py`; favor extracting API clients or token logic here before reusing across pages.
- Configuration templates (`.env.example`, `app/.streamlit/secrets.toml.example`) must stay in version control, while real secrets remain local via `.env` or Streamlit `Secrets`.
- Automation resides in `start.sh`, and dependency pins live in `requirements.txt`. Avoid adding ad-hoc setup scripts elsewhere.

## Build, Test, and Development Commands
- `./start.sh`: creates/refreshes `.venv`, installs dependencies, exports `.env`, and runs `streamlit run app/1_Home.py` for the full guided tour.
- `source .venv/bin/activate && pip install -r requirements.txt`: use when iterating quickly without restarting the whole script.
- `streamlit run app/1_Home.py`: launches the UI with your currently active virtualenv and environment variables.
- `pytest tests`: run lightweight unit tests (add new cases under `tests/`) covering utilities such as tokenization helpers before relying on manual demos.

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indentation, type hints for public helpers, and docstrings that state intent and IO (especially in `utils.py`).
- Use `snake_case` for functions/variables, `CamelCase` for classes, and preserve numeric prefixes on Streamlit page files to retain navigation order.
- Keep Streamlit widgets declarative and colocate layout code with its callbacks; multi-step workflows belong in helper functions in `utils.py`.

## Testing Guidelines
- Favor deterministic tests around data prep, tokenization, and scoring logic using `pytest`; mock external API calls via fixtures to avoid hitting OpenAI-compatible endpoints.
- Mirror file names with `test_*.py` modules under `tests/` and assert on user-facing strings or dataframe shapes seen in the demo.
- Treat the interactive app as a smoke test: after changes, run through each sidebar page and confirm expected outputs render without exceptions.

## Commit & Pull Request Guidelines
- Match the existing history (`Improve sentiment model, dashboard, and UI scaling`) by writing imperative, descriptive subject lines (~72 chars) without prefixes.
- Each PR should summarize motivation, list impacted pages/utilities, and include screenshots or console snippets when UI/UX changes are involved.
- Link relevant issues, call out config or secret updates, and ensure CI (or manual `pytest` runs) is referenced in the PR description before requesting review.

## Security & Configuration Tips
- Never commit `.env`, `secrets.toml`, or raw API keys; rely on `.gitignore` defaults and encourage reviewers to verify secrets stay local.
- Parameterize model names via environment variables so deployments on Streamlit Cloud vs. Bedrock merely change configuration, not code.
