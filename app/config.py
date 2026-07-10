"""Central configuration, loaded from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Access / models ------------------------------------------------------
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
APP_SESSION_DAYS = int(os.getenv("APP_SESSION_DAYS", "7"))
GEMINI_PARSE_MODEL = os.getenv("PW_PARSE_MODEL", "gemini-2.5-flash")
GEMINI_GEN_MODEL = os.getenv("PW_GEN_MODEL", "gemini-2.5-pro")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("PW_MAX_OUTPUT_TOKENS", "32768"))
# Local lexical vectors are only for the non-AI knowledge-base index.
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
KB_CACHE_DIR = DATA_DIR / "kb_cache"  # downloaded Drive sources (never committed)

for _d in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, INDEX_DIR, KB_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Knowledge base from Google Drive --------------------------------------
# The foundation source files live in a shared Google Drive folder (free hosting,
# nothing large committed to git). The admin builds the index once with
# scripts/build_kb.py; faculty machines only download the prebuilt index.
#
# KB_DRIVE_FOLDER  admin only: the shared Drive FOLDER holding the source files.
# KB_INDEX_URL     everyone: Drive share link of the prebuilt index zip produced
#                  by build_kb.py. If data/index is empty at startup, the app
#                  downloads and unpacks it automatically.
# DRIVE_API_KEY    optional, admin only: a Google API key with the Drive API
#                  enabled. Lists folders completely (no 50-files-per-folder
#                  limit of the keyless downloader).
KB_DRIVE_FOLDER = os.getenv("KB_DRIVE_FOLDER", "").strip()
KB_INDEX_URL = os.getenv("KB_INDEX_URL", "").strip()
DRIVE_API_KEY = os.getenv("DRIVE_API_KEY", "").strip()

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


# --- Boards & classes/levels ------------------------------------------------
# value -> human label. Board/class tag every knowledge-base chunk and filter
# retrieval, so a Class 8 ICSE paper is answered from Class 8 ICSE material.
# Retrieval falls back gracefully, so any combination works even when the
# knowledge base has no material tagged for it. NEET/JEE also switch the
# solver into its brief MCQ answer style.
BOARDS: dict[str, str] = {
    "CBSE": "CBSE",
    "ICSE": "ICSE",
    "State Board": "State Board",
}

CLASSES: dict[str, str] = {
    "Class 6": "Class 6",
    "Class 7": "Class 7",
    "Class 8": "Class 8",
    "Class 9": "Class 9",
    "Class 10": "Class 10",
    "Class 11": "Class 11",
    "Class 12": "Class 12",
    "Class 11+12": "Class 11 + 12",
    "NEET": "NEET",
    "JEE": "JEE",
}


# --- Subjects -------------------------------------------------------------
# value -> human label. The value selects the subject-specific answer-writing
# style in solver._build_prompt. "General" is a safe default for anything else.
SUBJECTS: dict[str, str] = {
    "General": "General / Other",
    "Mathematics": "Mathematics",
    "Science": "Science (combined)",
    "Physics": "Physics",
    "Chemistry": "Chemistry",
    "Biology": "Biology",
    "Social Science": "Social Science / SST",
    "English": "English",
}
