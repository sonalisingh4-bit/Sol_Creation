# Solution Creation Tool

Upload a question paper and export a complete solution document (`.docx` plus
optional `.pdf`) with a marks-appropriate model answer for every question.

The app is onboarded to the PW shared proxy:

- App name: `Solution Creation`
- User access is checked through the proxy allowlist before generation.
- Gemini calls go through the proxy; provider credentials are not stored here.
- A single `UsageSession` is used per generation job so usage is logged as a
  combined provider row for the task.

## Faculty Setup

```bash
.venv/Scripts/python -m pip install -r requirements.txt
copy .env.example .env
python main.py
```

Set these values in `.env`:

| Variable | Purpose |
|---|---|
| `GOOGLE_CLIENT_ID` | Google Identity Services client ID for user sign-in. |
| `KB_INDEX_URL` | Drive share link for the prebuilt knowledge-base index zip. |
| `PW_PARSE_MODEL` | Gemini model name used by the proxy for parsing/OCR. |
| `PW_GEN_MODEL` | Gemini model name used by the proxy for answer generation. |
| `PW_MAX_OUTPUT_TOKENS` | Output-token cap sent in proxy Gemini requests. |

On first start, the app downloads the knowledge base in the background if
`KB_INDEX_URL` is set. Users sign in with a whitelisted `@pw.live` Google
account, upload the paper, choose course or exam level, subject, language, and
school board when relevant, then generate.

## Admin Knowledge Base

```bash
python scripts/build_kb.py --dry-run
python scripts/build_kb.py
```

The script mirrors source books from `KB_DRIVE_FOLDER`, chunks them, builds the
local index, and writes `data/kb_foundation_index.zip`. Upload that zip to
Drive, share it as viewer, and give the link to faculty for `KB_INDEX_URL`.

Scanned/image source files still need proxy-backed OCR, so run admin indexing in
a context where a valid PW user token is available if those inputs are used.

## Project Layout

```text
app/
  config.py         settings, boards/classes/subjects, languages
  drive_sync.py     Google Drive download and index bootstrap
  gemini_client.py  PW proxy Gemini wrapper plus local lexical vectors
  extract.py        text extraction and proxy OCR fallback
  chunking.py       boundary-aware chunker
  vectorstore.py    persistent NumPy index plus local text search
  ingest.py         extract -> chunk -> store source material
  paper_parser.py   proxy-assisted paper parsing
  solver.py         retrieve + marks-aware multilingual answers
  document.py       render SolvedPaper -> .docx and optional .pdf
  jobs.py           background generation jobs with live progress
  main.py           FastAPI routes and access gate
scripts/
  build_kb.py       admin Drive-folder to index workflow
templates/ static/ HTMX UI
```

Answers are AI-generated. Review them before use.
