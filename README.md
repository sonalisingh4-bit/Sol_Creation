# Solution Creation Tool

Upload a **question paper**, ground the answers in **your own sources** (textbooks,
notes, references), and export a complete **solution document** (`.docx` + `.pdf`)
with a marks-appropriate model answer for every question — in the language you choose.

Built with **FastAPI + HTMX** and **Google Gemini** (parsing, embeddings, answers).

## How it works

```
Sources ──▶ extract text ──▶ chunk ──▶ Gemini embeddings ──▶ vector store (RAG)
                                                                     │
Question paper ──▶ Gemini parses questions + marks ──▶ per question: retrieve
   relevant passages ──▶ Gemini writes a marks-aware answer in chosen language
                                                                     │
                                                          .docx  +  .pdf
```

- **RAG knowledge base** — sources are chunked, embedded with Gemini, and stored in a
  lightweight on-disk NumPy vector index (no native build tools required). Each answer
  retrieves the most relevant passages so answers stay grounded in *your* material.
- **Marks-aware** — a 1-mark question gets a crisp sentence; a 10-mark question gets a
  structured long answer. Length and depth scale with the detected marks.
- **Multilingual** — English, Hindi, Bangla, Marathi, Gujarati, Kannada, Telugu, Tamil,
  Malayalam, Punjabi, Odia, Urdu (edit `LANGUAGES` in `app/config.py` to add more).
- **Scanned papers/images** — PDFs with no extractable text and image uploads are OCR'd
  by Gemini automatically.

## Setup

```bash
# 1. install dependencies (into the existing .venv)
.venv/Scripts/python -m pip install -r requirements.txt

# 2. add your Gemini API key
copy .env .env          # then edit .env and paste your key
# get a key at https://aistudio.google.com/apikey

# 3. run
python main.py                  # http://127.0.0.1:8000
```

## Usage

1. **Knowledge base** (left) — upload your sources and click *Add to knowledge base*.
   Supported: PDF, DOCX, TXT, MD, CSV, images. Add as many as you like; remove or clear
   any time. The index persists in `data/index/`.
2. **Generate** (right) — upload the question paper, pick the answer language, and click
   *Generate*. Progress is shown live; when done you get **.docx** and **.pdf** downloads.

## PDF export

The `.docx` is always produced. The `.pdf` is rendered from it using **Microsoft Word**
(`docx2pdf`) or **LibreOffice** (`soffice`), whichever is installed — both shape Indic
scripts correctly. If neither is present, you still get the `.docx` (which opens and
prints everywhere) and a note explaining how to enable PDF.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Your Gemini key. |
| `GEMINI_PARSE_MODEL` | `gemini-2.5-flash` | Parsing the paper / OCR. |
| `GEMINI_GEN_MODEL` | `gemini-2.5-pro` | Writing the answers. |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Embedding sources & queries. |
| `GEMINI_MAX_OUTPUT_TOKENS` | `32768` | Room for full-paper OCR and JSON parsing. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Source chunking. |
| `TOP_K` | `6` | Passages retrieved per question. |

## Project layout

```
app/
  config.py         settings + supported languages
  gemini_client.py  embeddings, generation, file upload (google-genai)
  extract.py        text extraction (pdf/docx/txt/image) + OCR fallback
  chunking.py       boundary-aware overlapping chunker
  vectorstore.py    persistent NumPy cosine-search index (swappable)
  ingest.py         extract → chunk → embed → store one source
  paper_parser.py   Gemini parses paper → structured questions + marks
  solver.py         retrieve + marks-aware multilingual answer per question
  document.py       render SolvedPaper → .docx (+ pdf)
  pdf_convert.py    docx → pdf via Word or LibreOffice
  jobs.py           background generation jobs with live progress
  main.py           FastAPI routes
templates/  static/ HTMX UI
```

> Answers are AI-generated. Review them before use.
