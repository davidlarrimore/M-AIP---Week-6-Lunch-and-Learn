#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo ".env not found, copying .env.example -> .env"
  cp .env.example .env
  echo "Update .env with your secrets (including the Bedrock base URL) and rerun the script."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

set -o allexport
source .env
set +o allexport

streamlit run demo_app/Home.py
