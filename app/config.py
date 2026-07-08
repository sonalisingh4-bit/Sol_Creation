"""Central configuration, loaded from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- API / models ---------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_PARSE_MODEL = os.getenv("GEMINI_PARSE_MODEL", "gemini-2.5-flash")
GEMINI_GEN_MODEL = os.getenv("GEMINI_GEN_MODEL", "gemini-2.5-pro")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "32768"))
# gemini-embedding-001 supports Matryoshka dims (768/1536/3072). 1536 is a good
# quality/footprint balance; embeddings are normalised before cosine search.
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

# --- Retrieval ------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))

# --- Storage --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (BASE_DIR / os.getenv("DATA_DIR", "data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
INDEX_DIR = DATA_DIR / "index"

for _d in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Supported answer languages -------------------------------------------
# value -> human label shown in the UI. The value is passed to Gemini.
LANGUAGES: dict[str, str] = {
    "English": "English",
    "Hindi": "Hindi (हिन्दी)",
    "Bengali": "Bangla (বাংলা)",
    "Marathi": "Marathi (मराठी)",
    "Gujarati": "Gujarati (ગુજરાતી)",
    "Kannada": "Kannada (ಕನ್ನಡ)",
    "Telugu": "Telugu (తెలుగు)",
    "Tamil": "Tamil (தமிழ்)",
    "Malayalam": "Malayalam (മലയാളം)",
    "Punjabi": "Punjabi (ਪੰਜਾਬੀ)",
    "Odia": "Odia (ଓଡ଼ିଆ)",
    "Urdu": "Urdu (اردو)",
}


# --- Subjects -------------------------------------------------------------
# value -> human label. The value selects the subject-specific answer-writing
# style in solver._build_prompt. "General" is a safe default for anything else.
SUBJECTS: dict[str, str] = {
    "General": "General / Other",
    "Mathematics": "Mathematics",
    "Physics": "Physics",
    "Chemistry": "Chemistry",
    "Biology": "Biology",
}


def require_api_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env to .env and add your key."
        )
    return GEMINI_API_KEY
