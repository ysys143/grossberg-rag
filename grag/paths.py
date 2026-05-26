"""Single source of truth for filesystem paths.

Every module resolves config, assets, and runtime data through here instead of
re-deriving `Path(__file__).parent`, so moving files never silently breaks a path.

Layout:
  PROJECT_ROOT/                 repo root (config.yaml, .env live here)
    grag/                       package (PKG_DIR)
      prompts/  static/         code-adjacent assets (travel with the package)
    data/                       runtime data (gitignored)
      rag_storage/ output/ sessions/ logs/ pdfs/
"""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"
APIKEY_ENV = Path.home() / ".oh-my-zsh/custom/apikey.env"  # shared shell secrets

PROMPTS_DIR = PKG_DIR / "prompts"
STATIC_DIR = PKG_DIR / "static"

DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"  # persisted conversation history
LOGS_DIR = DATA_DIR / "logs"          # llm_calls.jsonl etc.
